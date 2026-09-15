"""
data_prep.py
============
Loads FreedomIntelligence/Medical-Dialogue-Dataset and
medalpaca/medical_meadow_medqa, unifies them into a single
instruction -> {reasoning, answer} schema with a Gemma-2 chat template,
and writes train/val/test JSONL splits plus a dataset_card.json for
reproducibility.

Usage:
    python data/data_prep.py --output_dir ./data/processed \
        --max_medqa 8000 --max_dialogue 4000
"""
import argparse
import json
import os
import random
import re
from pathlib import Path

from datasets import load_dataset, concatenate_datasets, Dataset

SYSTEM_INSTRUCTION = (
    "You are a careful clinical reasoning assistant. Given a medical "
    "question or patient case, think step by step through the relevant "
    "history, findings, and differential considerations inside a "
    "<reasoning>...</reasoning> block, then give a concise final answer "
    "inside an <answer>...</answer> block. Never state clinical facts you "
    "are not confident about; if uncertain, say so explicitly."
)

GEMMA_TEMPLATE = (
    "<start_of_turn>user\n{system}\n\n{question}<end_of_turn>\n"
    "<start_of_turn>model\n{response}<end_of_turn>\n"
)


def synthesize_cot(question: str, options: dict, correct_letter: str, correct_text: str) -> str:
    """
    Build a lightweight, rule-based chain-of-thought scaffold for MedQA
    items that ship only an answer key (no physician-authored rationale).

    This is a *weak supervision* signal: it mentions the key clinical
    clues present in the stem and walks through elimination of the other
    listed options before committing to the correct one. Swap this
    function out for a teacher-model-generated rationale (e.g. by
    prompting a larger model offline and caching results) for
    higher-quality supervision.
    """
    stem_clues = re.split(r"(?<=[.?!])\s+", question.strip())
    lead_clue = stem_clues[0] if stem_clues else question[:120]

    lines = [
        f"The key clinical clue in the stem is: \"{lead_clue}\"",
        "Reviewing the answer options against this presentation:",
    ]
    for letter, text in options.items():
        if letter == correct_letter:
            continue
        lines.append(f"- Option {letter} ({text}) is less consistent with the described presentation than the best answer, so it is less likely.")
    lines.append(
        f"- Option {correct_letter} ({correct_text}) best fits the clinical picture described in the stem."
    )
    lines.append(f"Therefore, the most likely answer is {correct_letter}: {correct_text}.")
    return " ".join(lines)


def parse_medqa_options(input_field: str):
    """
    medical_meadow_medqa's `input` field typically embeds the multiple
    choice options as lines like 'A. text' / 'B. text' etc. Parse them
    defensively; return {} if the format doesn't match (free-text item).
    """
    options = {}
    for match in re.finditer(r"\b([A-E])[\.\)]\s*([^\n]+)", input_field or ""):
        options[match.group(1)] = match.group(2).strip()
    return options


def build_medqa_examples(max_n: int):
    print("Loading medalpaca/medical_meadow_medqa ...")
    ds = load_dataset("medalpaca/medical_meadow_medqa", split="train")
    if max_n:
        ds = ds.shuffle(seed=42).select(range(min(max_n, len(ds))))

    examples = []
    for row in ds:
        instruction = row.get("instruction", "").strip()
        input_field = row.get("input", "").strip()
        output_field = row.get("output", "").strip()
        if not output_field:
            continue

        question = f"{instruction}\n\n{input_field}".strip()
        options = parse_medqa_options(input_field)

        # try to recover the correct letter from the output text
        letter_match = re.match(r"\s*([A-E])[\.\):]", output_field)
        correct_letter = letter_match.group(1) if letter_match else None
        correct_text = output_field

        if options and correct_letter and correct_letter in options:
            reasoning = synthesize_cot(question, options, correct_letter, options[correct_letter])
            final_answer = f"{correct_letter}: {options[correct_letter]}"
        else:
            # Free-text / short-answer style item: lighter scaffold.
            reasoning = (
                f"Analyzing the clinical stem: \"{question[:200]}...\" "
                "the presentation and findings point toward the diagnosis/management "
                "described in the reference answer."
            )
            final_answer = correct_text

        examples.append({
            "source": "medical_meadow_medqa",
            "question": question,
            "reasoning": reasoning,
            "answer": final_answer,
        })
    print(f"  -> built {len(examples)} MedQA CoT examples")
    return examples


def build_dialogue_examples(max_n: int):
    print("Loading FreedomIntelligence/Medical-Dialogue-Dataset ...")
    try:
        ds = load_dataset("FreedomIntelligence/Medical-Dialogue-Dataset", split="train")
    except Exception as e:
        print(f"  WARNING: could not load Medical-Dialogue-Dataset ({e}); skipping this source.")
        return []

    if max_n:
        ds = ds.shuffle(seed=42).select(range(min(max_n, len(ds))))

    examples = []
    for row in ds:
        # The dataset's exact column names vary by config/version, so we
        # defensively look for the most likely fields.
        patient_turn = row.get("description") or row.get("utterances", [None])[0] if isinstance(row.get("utterances"), list) else row.get("patient") or ""
        doctor_turn = row.get("doctor") or (row.get("utterances", [None, None])[1] if isinstance(row.get("utterances"), list) and len(row.get("utterances", [])) > 1 else None)

        if not patient_turn or not doctor_turn:
            continue

        question = str(patient_turn).strip()
        answer_text = str(doctor_turn).strip()
        if len(question) < 10 or len(answer_text) < 10:
            continue

        reasoning = (
            "The patient's message describes their symptoms/history. "
            "Considering the described complaint, relevant differentials, "
            "and typical first-line guidance for this presentation, "
            "a safe and informative response can be formed before advising the patient."
        )
        examples.append({
            "source": "medical_dialogue_dataset",
            "question": question,
            "reasoning": reasoning,
            "answer": answer_text,
        })
    print(f"  -> built {len(examples)} dialogue CoT examples")
    return examples


def format_gemma(example: dict) -> str:
    response = f"<reasoning>{example['reasoning']}</reasoning>\n<answer>{example['answer']}</answer>"
    return GEMMA_TEMPLATE.format(
        system=SYSTEM_INSTRUCTION,
        question=example["question"],
        response=response,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", type=str, default="./data/processed")
    ap.add_argument("--max_medqa", type=int, default=8000)
    ap.add_argument("--max_dialogue", type=int, default=4000)
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--test_frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    medqa_examples = build_medqa_examples(args.max_medqa)
    dialogue_examples = build_dialogue_examples(args.max_dialogue)
    all_examples = medqa_examples + dialogue_examples
    random.shuffle(all_examples)

    for ex in all_examples:
        ex["text"] = format_gemma(ex)

    n = len(all_examples)
    n_val = int(n * args.val_frac)
    n_test = int(n * args.test_frac)
    n_train = n - n_val - n_test

    train_ex = all_examples[:n_train]
    val_ex = all_examples[n_train:n_train + n_val]
    test_ex = all_examples[n_train + n_val:]

    def write_jsonl(path, rows):
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    write_jsonl(out_dir / "train.jsonl", train_ex)
    write_jsonl(out_dir / "val.jsonl", val_ex)
    write_jsonl(out_dir / "test.jsonl", test_ex)

    card = {
        "datasets": {
            "medalpaca/medical_meadow_medqa": {"n_used": len(medqa_examples), "cap": args.max_medqa},
            "FreedomIntelligence/Medical-Dialogue-Dataset": {"n_used": len(dialogue_examples), "cap": args.max_dialogue},
        },
        "splits": {"train": len(train_ex), "val": len(val_ex), "test": len(test_ex)},
        "seed": args.seed,
        "prompt_template": GEMMA_TEMPLATE,
        "system_instruction": SYSTEM_INSTRUCTION,
        "notes": "MedQA reasoning traces are synthetically scaffolded (rule-based), not physician-authored. See README limitations.",
    }
    with open(out_dir / "dataset_card.json", "w", encoding="utf-8") as f:
        json.dump(card, f, indent=2)

    print(f"\nDone. train={len(train_ex)} val={len(val_ex)} test={len(test_ex)}")
    print(f"Wrote files to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
