"""
score_triviaqa.py — Re-label a TriviaQA completion cache using an LLM judge.

String-match labeling has ~10-20% false negatives (e.g., "Ross Bagdasarian Sr."
for gold answer "David Seville").  This script replaces those labels by calling
qwen3-small on NRP's LLM endpoint, which can resolve alternate names and partial
matches.

Usage (cluster, after triviaqa generation cache exists on PVC):

    python scripts/score_triviaqa.py \\
        --model gpt2-medium \\
        --triviaqa-cache /data/datasets/triviaqa_gpt2-medium.json \\
        --output-dir /data/experiments/phase2_mechanism/gpt2_medium_s0_triviaqa_judged \\
        --api-key E0fTkLAw3ZqcsYuugDKbwh5VcTMmto8x

Outputs:
    {output_dir}/dataset.json    — same format as HallucinationDataset, corrected labels
    {output_dir}/judge_stats.json — label flip statistics

Performance:
    batch_size=10, max_workers=10 → ~300 API calls per 3000 examples,
    ~20s/call with 10 concurrent workers → ~600s (~10 min) per model.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

API_BASE = "https://ellm.nrp-nautilus.io/v1"
JUDGE_MODEL = "qwen3-small"
BATCH_SIZE = 10
MAX_WORKERS = 10
MAX_TOKENS = 2048  # must be large enough for reasoning chain to complete
MAX_RETRIES = 3


def _build_batch_prompt(examples: List[dict]) -> str:
    """Build a single prompt asking for N verdicts, one per line."""
    lines = [
        "For each item below output ONLY correct or incorrect, one per line, in order.\n"
    ]
    for i, ex in enumerate(examples):
        q = ex["prompt"].replace("Question: ", "").replace("\nAnswer:", "").strip()
        gold = ex["metadata"].get("answer_value", "")
        resp = ex["completion"]
        lines.append(f"{i+1}. Q: {q} | Gold: {gold} | Response: {resp}")
    return "\n".join(lines)


def _parse_verdicts(content: str, n: int) -> List[bool]:
    """Parse n verdicts from a content string. Returns list of bool (True=correct)."""
    verdicts: List[bool] = []
    for line in content.strip().splitlines():
        word = line.strip().lower().lstrip("0123456789). ")
        if word.startswith("correct"):
            verdicts.append(True)
        elif word.startswith("incorrect"):
            verdicts.append(False)
        if len(verdicts) == n:
            break
    # If parsing failed, fall back to all-incorrect (conservative)
    while len(verdicts) < n:
        verdicts.append(False)
    return verdicts


def judge_batch(client, examples: List[dict], retries: int = MAX_RETRIES) -> List[bool]:
    """Call the LLM judge for a batch of examples. Returns list of bool (correct/not)."""
    prompt = _build_batch_prompt(examples)
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=MAX_TOKENS,
                temperature=0,
            )
            content = resp.choices[0].message.content or ""
            verdicts = _parse_verdicts(content, len(examples))
            return verdicts
        except Exception as exc:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"[judge] Batch failed after {retries} attempts: {exc}")
                return [False] * len(examples)


def main():
    p = argparse.ArgumentParser(description="Re-label TriviaQA cache with LLM judge")
    p.add_argument("--triviaqa-cache", required=True,
                   help="Path to triviaqa_{model}.json (generated completions)")
    p.add_argument("--output-dir", required=True,
                   help="Directory to write dataset.json and judge_stats.json")
    p.add_argument("--api-key", required=True)
    p.add_argument("--api-base", default=API_BASE)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    args = p.parse_args()

    from openai import OpenAI
    from ghosttrack.data.dataset import HallucinationDataset, HallucinationExample

    client = OpenAI(api_key=args.api_key, base_url=args.api_base)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load existing string-match labeled dataset
    print(f"[judge] Loading {args.triviaqa_cache}")
    with open(args.triviaqa_cache) as f:
        raw = json.load(f)
    examples = raw["examples"]
    print(f"[judge] {len(examples)} examples  "
          f"(string-match: {sum(1 for e in examples if e['label']==0)} factual, "
          f"{sum(1 for e in examples if e['label']==1)} hallucinated)")

    # Build batches
    batches: List[Tuple[int, List[dict]]] = []
    for start in range(0, len(examples), args.batch_size):
        batch = examples[start:start + args.batch_size]
        batches.append((start, batch))

    print(f"[judge] {len(batches)} batches × {args.batch_size}  "
          f"({args.max_workers} concurrent workers)  model={JUDGE_MODEL}")

    # Judge all batches concurrently
    new_labels = [None] * len(examples)
    done = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {
            pool.submit(judge_batch, client, batch): (start, batch)
            for start, batch in batches
        }
        for fut in as_completed(futures):
            start, batch = futures[fut]
            verdicts = fut.result()
            for offset, is_correct in enumerate(verdicts):
                new_labels[start + offset] = 0 if is_correct else 1
            done += len(batch)
            elapsed = time.time() - t0
            rate = done / elapsed
            remaining = (len(examples) - done) / rate if rate > 0 else 0
            print(f"[judge] {done}/{len(examples)}  "
                  f"{elapsed:.0f}s elapsed  ~{remaining:.0f}s remaining")

    # Build corrected dataset
    old_factual = sum(1 for e in examples if e["label"] == 0)
    new_factual = sum(1 for l in new_labels if l == 0)

    flipped_to_factual   = sum(1 for e, nl in zip(examples, new_labels) if e["label"]==1 and nl==0)
    flipped_to_halluc    = sum(1 for e, nl in zip(examples, new_labels) if e["label"]==0 and nl==1)

    print(f"\n[judge] Label correction summary:")
    print(f"  String-match factual:  {old_factual} ({100*old_factual/len(examples):.1f}%)")
    print(f"  LLM-judged factual:    {new_factual} ({100*new_factual/len(examples):.1f}%)")
    print(f"  Flipped incorrect→correct: {flipped_to_factual}")
    print(f"  Flipped correct→incorrect: {flipped_to_halluc}")

    corrected_dataset = HallucinationDataset()
    for ex, new_label in zip(examples, new_labels):
        corrected_dataset.add(HallucinationExample(
            id=ex["id"],
            prompt=ex["prompt"],
            completion=ex["completion"],
            label=new_label,
            source=ex.get("source", "trivia_qa"),
            category=ex.get("category", ""),
            metadata=ex.get("metadata", {}),
        ))

    out_path = out_dir / "dataset.json"
    corrected_dataset.save(str(out_path))
    print(f"[judge] Saved corrected dataset → {out_path}")

    stats = {
        "n_total": len(examples),
        "string_match_factual": old_factual,
        "string_match_hallucinated": len(examples) - old_factual,
        "llm_judged_factual": new_factual,
        "llm_judged_hallucinated": len(examples) - new_factual,
        "flipped_to_factual": flipped_to_factual,
        "flipped_to_hallucinated": flipped_to_halluc,
        "judge_model": JUDGE_MODEL,
        "elapsed_seconds": time.time() - t0,
    }
    with open(out_dir / "judge_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[judge] Saved stats → {out_dir / 'judge_stats.json'}")


if __name__ == "__main__":
    main()
