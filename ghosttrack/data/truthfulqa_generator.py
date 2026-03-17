"""
TruthfulQA data generator.

Loads TruthfulQA from HuggingFace, runs the wrapped model to generate a
completion for each question, then scores it by string-matching against
the known correct answers.  Produces one :class:`HallucinationExample`
per question.

Key design decisions
--------------------
- ``completion`` stores only the *generated* tokens, not the prompt.  The
  tracking phase concatenates ``prompt + " " + completion`` itself.  Storing
  the full ``prompt + generated`` in the completion field would cause the
  prompt to appear twice during hidden-state extraction.
- Results are cached to disk after the first run so that subsequent calls
  (e.g. for seeds 1-4) skip the expensive generation step entirely.

Usage::

    from ghosttrack.models import create_model
    from ghosttrack.data import TruthfulQAGenerator

    model = create_model("gpt2-medium", device="cuda")
    gen = TruthfulQAGenerator(model, cache_dir="/data/datasets")
    dataset = gen.generate(
        max_new_tokens=64,
        cache_path="/data/datasets/truthfulqa_gpt2-medium.json",
    )
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

import torch
from datasets import load_dataset

from .dataset import HallucinationDataset, HallucinationExample


def _normalise(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def _is_correct(generated_only: str, correct_answers: List[str]) -> bool:
    """
    Return True if *generated_only* (prompt excluded) contains any correct answer.

    Args:
        generated_only: The model's generated text with the prompt stripped.
        correct_answers: List of acceptable answer strings from TruthfulQA.
    """
    norm = _normalise(generated_only)
    for ans in correct_answers:
        if _normalise(ans) in norm:
            return True
    return False


def _generate_completion_only(model, prompt: str, max_new_tokens: int) -> str:
    """
    Run greedy decoding and return *only* the newly generated tokens.

    Unlike ``BaseModelWrapper.generate_text``, this decodes only the tokens
    produced after the prompt so that prompt and completion can be stored
    separately without duplication.

    Args:
        model: A :class:`~ghosttrack.models.base.BaseModelWrapper` instance.
        prompt: Input text (question).
        max_new_tokens: How many tokens to generate beyond the prompt.

    Returns:
        Generated completion string (prompt tokens excluded).
    """
    encoded = model.encode_text(prompt)
    input_len = encoded["input_ids"].shape[1]
    with torch.no_grad():
        out = model.model.generate(
            encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=model.tokenizer.eos_token_id,
        )
    new_tokens = out[0][input_len:]
    return model.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


class TruthfulQAGenerator:
    """
    Generate labelled completions from TruthfulQA using a model wrapper.

    The generated completions are the model's *own* outputs — genuine model
    hallucinations rather than human-crafted false answers (as in HaluEval).
    This satisfies the methodological requirement that the detector is trained
    on the model's own factual vs. hallucinated behaviour.

    Args:
        model: A :class:`~ghosttrack.models.base.BaseModelWrapper` instance.
        cache_dir: HuggingFace dataset cache directory.
    """

    def __init__(self, model, cache_dir: Optional[str] = None):
        self.model = model
        self.cache_dir = cache_dir

    def generate(
        self,
        max_new_tokens: int = 64,
        max_examples: Optional[int] = None,
        hf_split: str = "validation",
        cache_path: Optional[str] = None,
    ) -> HallucinationDataset:
        """
        Run the model on every TruthfulQA question and label the output.

        If *cache_path* is provided and the file already exists, the saved
        dataset is returned immediately without re-generating.  Otherwise,
        completions are generated and written to *cache_path* if provided.

        Args:
            max_new_tokens: Number of tokens to generate per question.
            max_examples: Stop after this many questions (``None`` = all 817).
            hf_split: HuggingFace dataset split (``"validation"`` for TruthfulQA).
            cache_path: Optional path to a JSON file.  On cache hit, generation
                is skipped.  On cache miss, the result is saved here.

        Returns:
            :class:`~ghosttrack.data.dataset.HallucinationDataset` with one
            example per question.  ``example.prompt`` is the question;
            ``example.completion`` is the generated answer only (no prompt).
        """
        # --- Cache hit ---
        if cache_path and Path(cache_path).exists():
            print(f"[TruthfulQAGenerator] Loading cached dataset from {cache_path}")
            return HallucinationDataset.load(cache_path)

        # --- Generate ---
        raw = load_dataset(
            "truthful_qa", "generation",
            split=hf_split,
            cache_dir=self.cache_dir,
        )

        dataset = HallucinationDataset()
        n_factual = 0
        n_halluc = 0

        for idx, row in enumerate(raw):
            if max_examples is not None and idx >= max_examples:
                break

            question: str = row["question"]
            correct_answers: List[str] = row.get(
                "correct_answers", [row.get("best_answer", "")]
            )
            category: str = row.get("category", "")

            try:
                completion = _generate_completion_only(
                    self.model, question, max_new_tokens
                )
            except Exception as exc:
                print(f"[TruthfulQAGenerator] Skipping question {idx}: {exc}")
                continue

            label = 0 if _is_correct(completion, correct_answers) else 1
            if label == 0:
                n_factual += 1
            else:
                n_halluc += 1

            dataset.add(HallucinationExample(
                id=f"truthfulqa_{idx}",
                prompt=question,
                completion=completion,
                label=label,
                source="truthful_qa",
                category=category,
                metadata={"correct_answers": correct_answers},
            ))

            if (idx + 1) % 50 == 0:
                print(
                    f"[TruthfulQAGenerator] {idx + 1} questions  "
                    f"({n_factual} factual, {n_halluc} hallucinated)"
                )

        print(
            f"[TruthfulQAGenerator] Done: {len(dataset)} examples  "
            f"({n_factual} factual, {n_halluc} hallucinated)"
        )

        # --- Cache write ---
        if cache_path:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            dataset.save(cache_path)
            print(f"[TruthfulQAGenerator] Saved to {cache_path}")

        return dataset
