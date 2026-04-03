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

Instruct vs base models
------------------------
Instruction-tuned models (names containing "Instruct", "instruct", or "chat")
have a ``tokenizer.chat_template`` and are designed to respond to messages in
chat format.  Using the raw "Question: X\nAnswer:" prompt on these models
causes them to acknowledge the question rather than answer it directly.

When ``instruct_mode`` is True (auto-detected by default for instruct models):
- The prompt is built with ``tokenizer.apply_chat_template`` so the model
  sees a proper ``<|user|>`` / ``<|assistant|>`` structure.
- The instruction asks for a "one-phrase answer" which forces a committed,
  short response.
- ``max_new_tokens`` defaults to 16 (vs 32 for base models) — instruct models
  give concise answers without rambling, making string-match scoring reliable.

Design decisions
----------------
- Only the generated tokens (no prompt) are stored in ``completion``.
- Correctness uses ``answer['value']`` and ``answer['normalized_aliases']``
  (NOT ``answer['aliases']``, which can contain noisy Wikipedia article titles).
- Results are cached to disk; subsequent seed runs skip generation entirely.

Usage::

    from ghosttrack.models import create_model
    from ghosttrack.data import TriviaQAGenerator

    # Base model — auto-uses "Question: X\\nAnswer:" prompt
    model = create_model("gpt2-medium", device="cuda")
    gen = TriviaQAGenerator(model, cache_dir="/data/datasets")
    dataset = gen.generate(max_examples=3000,
                           cache_path="/data/datasets/triviaqa_gpt2-medium.json")

    # Instruct model — auto-uses chat_template prompt
    model = create_model("Qwen/Qwen2.5-3B-Instruct", device="cuda")
    gen = TriviaQAGenerator(model, cache_dir="/data/datasets")
    dataset = gen.generate(max_examples=3000,
                           cache_path="/data/datasets/triviaqa_Qwen-Qwen2.5-3B-Instruct.json")

    print(dataset.n_factual, dataset.n_hallucinated)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

import torch
from datasets import load_dataset

from .dataset import HallucinationDataset, HallucinationExample

# Prompt template for base (non-instruct) models.
_PROMPT_TEMPLATE = "Question: {question}\nAnswer:"

# Instruction for instruct models — requests a concise, committed answer.
_INSTRUCT_CONTENT = "Answer in one phrase or word: {question}"


def _is_instruct_model(model) -> bool:
    """Detect instruct models by tokenizer chat_template or model name."""
    name = getattr(model, "model_name", "").lower()
    if any(k in name for k in ("instruct", "chat", "it")):
        return True
    tok = getattr(model, "tokenizer", None)
    if tok is not None and getattr(tok, "chat_template", None) is not None:
        return True
    return False


def _build_prompt(model, question: str, instruct_mode: bool) -> str:
    """Build the prompt string for a question."""
    if not instruct_mode:
        return _PROMPT_TEMPLATE.format(question=question)
    tok = model.tokenizer
    content = _INSTRUCT_CONTENT.format(question=question)
    if hasattr(tok, "apply_chat_template"):
        return tok.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )
    # Fallback if no chat_template method (shouldn't happen for instruct models)
    return content + "\n"


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

    Automatically detects instruct models (via tokenizer.chat_template or model
    name) and switches to chat-format prompts with shorter ``max_new_tokens``.

    Args:
        model: A :class:`~ghosttrack.models.base.BaseModelWrapper` instance.
        cache_dir: HuggingFace dataset cache directory.
        instruct_mode: Override auto-detection. ``True`` forces instruct prompts;
            ``False`` forces the base "Question: X\\nAnswer:" template;
            ``None`` (default) auto-detects from the model.
    """

    def __init__(self, model, cache_dir: Optional[str] = None,
                 instruct_mode: Optional[bool] = None):
        self.model = model
        self.cache_dir = cache_dir
        if instruct_mode is None:
            self.instruct_mode = _is_instruct_model(model)
        else:
            self.instruct_mode = instruct_mode

    def generate(
        self,
        max_new_tokens: Optional[int] = None,
        max_examples: int = 3000,
        hf_split: str = "validation",
        cache_path: Optional[str] = None,
    ) -> HallucinationDataset:
        """
        Run the model on TriviaQA questions and label the outputs.

        Args:
            max_new_tokens: Tokens to generate per question.  Defaults to 16
                for instruct models (concise direct answers) and 32 for base
                models.  Shorter is better: long completions produce substring
                matches that LLM judges reject as incorrect.
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

        # Default token budget: instruct models give concise answers; base
        # models need more tokens to eventually reach the answer in their
        # continuation-style generation.
        if max_new_tokens is None:
            max_new_tokens = 16 if self.instruct_mode else 32

        print(f"[TriviaQAGenerator] instruct_mode={self.instruct_mode}  "
              f"max_new_tokens={max_new_tokens}")

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

            prompt = _build_prompt(self.model, question, self.instruct_mode)

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
