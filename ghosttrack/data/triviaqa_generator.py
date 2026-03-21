"""
TriviaQA data generator (rc.nocontext split).

Loads TriviaQA from HuggingFace, prompts the model with a Q&A template,
and labels each completion correct/hallucinated by checking against the
answer value and normalized aliases.

Why TriviaQA over TruthfulQA for small models
----------------------------------------------
TruthfulQA is calibrated for large models (GPT-3+); GPT-2 Medium, Qwen 1.5B,
and Phi-2 hallucinate on ~92-95% of its questions, leaving fewer than 40
factual examples out of 817 — far too few for a meaningful detector.
TriviaQA covers common factoid knowledge; smaller models answer 15-30%
correctly, giving hundreds of factual examples from max_examples=3000.

Design decisions
----------------
- Prompt template: "Question: {q}\\nAnswer:" — standard few-shot format that
  cues the model to produce a short, direct answer.
- Only the generated tokens (no prompt) are stored in ``completion``.
- Correctness uses ``answer['value']`` and ``answer['normalized_aliases']``
  (NOT ``answer['aliases']``, which can contain noisy Wikipedia article titles).
- Results are cached to disk; subsequent seed runs skip generation entirely.

Usage::

    from ghosttrack.models import create_model
    from ghosttrack.data import TriviaQAGenerator

    model = create_model("gpt2-medium", device="cuda")
    gen = TriviaQAGenerator(model, cache_dir="/data/datasets")
    dataset = gen.generate(
        max_examples=3000,
        cache_path="/data/datasets/triviaqa_gpt2-medium.json",
    )
    print(dataset.n_factual, dataset.n_hallucinated)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

import torch
from datasets import load_dataset

from .dataset import HallucinationDataset, HallucinationExample

# Prompt template — cues the model to produce a short factual answer.
_PROMPT_TEMPLATE = "Question: {question}\nAnswer:"


def _normalise(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def _is_correct(generated: str, answer_value: str, normalized_aliases: List[str]) -> bool:
    """
    Check if the generated text contains the correct answer.

    Uses ``answer_value`` and ``normalized_aliases`` (cleaner than raw aliases
    which can include noisy Wikipedia article titles).
    """
    norm_gen = _normalise(generated)
    candidates = [answer_value] + normalized_aliases
    for ans in candidates:
        if ans and _normalise(ans) in norm_gen:
            return True
    return False


def _generate_completion_only(model, prompt: str, max_new_tokens: int) -> str:
    """Generate and return only the newly produced tokens (prompt excluded)."""
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


class TriviaQAGenerator:
    """
    Generate labelled completions from TriviaQA (rc.nocontext) using a model.

    Args:
        model: A :class:`~ghosttrack.models.base.BaseModelWrapper` instance.
        cache_dir: HuggingFace dataset cache directory.
    """

    def __init__(self, model, cache_dir: Optional[str] = None):
        self.model = model
        self.cache_dir = cache_dir

    def generate(
        self,
        max_new_tokens: int = 32,
        max_examples: int = 3000,
        hf_split: str = "validation",
        cache_path: Optional[str] = None,
    ) -> HallucinationDataset:
        """
        Run the model on TriviaQA questions and label the outputs.

        Args:
            max_new_tokens: Tokens to generate per question. 32 is enough
                for a short factual answer; longer hurts correctness scoring.
            max_examples: Number of questions to process. 3000 gives ~450-900
                factual examples even at low (15-30%) model accuracy.
            hf_split: HuggingFace split name (``"validation"`` recommended).
            cache_path: If set and file exists, load from cache (skip generation).
                On cache miss, save the result here for future runs.

        Returns:
            :class:`~ghosttrack.data.dataset.HallucinationDataset` with one
            example per question. ``prompt`` is the formatted question;
            ``completion`` is the generated answer only.
        """
        if cache_path and Path(cache_path).exists():
            print(f"[TriviaQAGenerator] Loading cached dataset from {cache_path}")
            return HallucinationDataset.load(cache_path)

        raw = load_dataset(
            "trivia_qa", "rc.nocontext",
            split=hf_split,
            cache_dir=self.cache_dir,
        )

        dataset = HallucinationDataset()
        n_factual = 0
        n_halluc = 0

        for idx, row in enumerate(raw):
            if idx >= max_examples:
                break

            question: str = row["question"]
            answer_value: str = row["answer"]["value"]
            normalized_aliases: List[str] = row["answer"].get("normalized_aliases", [])

            prompt = _PROMPT_TEMPLATE.format(question=question)

            try:
                completion = _generate_completion_only(
                    self.model, prompt, max_new_tokens
                )
            except Exception as exc:
                print(f"[TriviaQAGenerator] Skipping question {idx}: {exc}")
                continue

            label = 0 if _is_correct(completion, answer_value, normalized_aliases) else 1
            if label == 0:
                n_factual += 1
            else:
                n_halluc += 1

            dataset.add(HallucinationExample(
                id=f"triviaqa_{idx}",
                prompt=prompt,
                completion=completion,
                label=label,
                source="trivia_qa",
                category=row.get("type", ""),
                metadata={"answer_value": answer_value},
            ))

            if (idx + 1) % 100 == 0:
                pct = 100 * n_factual / (idx + 1)
                print(
                    f"[TriviaQAGenerator] {idx + 1}/{max_examples}  "
                    f"factual={n_factual} ({pct:.1f}%)  halluc={n_halluc}"
                )

        pct = 100 * n_factual / max(len(dataset), 1)
        print(
            f"[TriviaQAGenerator] Done: {len(dataset)} examples  "
            f"({n_factual} factual [{pct:.1f}%], {n_halluc} hallucinated)"
        )

        if cache_path:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            dataset.save(cache_path)
            print(f"[TriviaQAGenerator] Saved to {cache_path}")

        return dataset
