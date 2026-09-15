"""
evaluate.py
===========
Multi-metric evaluation harness for ClinReason-2B.

Computes, over a held-out test JSONL (same schema as data_prep.py output):
  1. Exact-match accuracy on the extracted final answer (MCQA items)
  2. ROUGE-1/2/L and BLEU between generated text and reference
  3. BERTScore (P/R/F1) for semantic similarity
  4. CoT diagnostics: reasoning-presence rate, avg reasoning length
  5. Hallucination proxy: NLI-based unsupported-claim rate of reasoning
     sentences against the source question (entailment/neutral/contradiction)
  6. Latency / throughput

Each metric block is wrapped so a missing optional dependency only
disables that block (with a printed warning) instead of crashing the run.

Usage:
    python eval/evaluate.py \
        --model_path outputs/clinreason-2b/final_adapter \
        --test_file data/processed/test.jsonl \
        --report_path eval/report.json \
        --n_samples 200
"""
import argparse
import json
import re
import time
from pathlib import Path

SYSTEM_INSTRUCTION = (
    "You are a careful clinical reasoning assistant. Given a medical "
    "question or patient case, think step by step through the relevant "
    "history, findings, and differential considerations inside a "
    "<reasoning>...</reasoning> block, then give a concise final answer "
    "inside an <answer>...</answer> block. Never state clinical facts you "
    "are not confident about; if uncertain, say so explicitly."
)


def build_prompt(question: str) -> str:
    return (
        f"<start_of_turn>user\n{SYSTEM_INSTRUCTION}\n\n{question}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )


def extract_block(text: str, tag: str) -> str:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return m.group(1).strip() if m else ""


def normalize_answer(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9 ]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def load_test_set(path: str, n_samples: int):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    if n_samples:
        rows = rows[:n_samples]
    return rows


def generate_batch(model, tokenizer, questions, max_new_tokens=512, batch_size=8):
    """Yields (raw_generated_text, latency_seconds) pairs."""
    import torch
    from unsloth import FastLanguageModel

    FastLanguageModel.for_inference(model)
    outputs = []
    for i in range(0, len(questions), batch_size):
        batch = questions[i:i + batch_size]
        prompts = [build_prompt(q) for q in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        start = time.time()
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        elapsed = time.time() - start
        for j in range(len(batch)):
            input_len = inputs["input_ids"].shape[1]
            text = tokenizer.decode(gen[j][input_len:], skip_special_tokens=True)
            outputs.append((text, elapsed / len(batch)))
    return outputs


# ---------------------------------------------------------------------------
# Metric blocks (each independently optional)
# ---------------------------------------------------------------------------

def compute_exact_match(preds, refs):
    correct = 0
    total = 0
    for p, r in zip(preds, refs):
        p_ans = extract_block(p, "answer") or p
        # reference answer may be "A: text" or free text
        total += 1
        if normalize_answer(r) and normalize_answer(r) in normalize_answer(p_ans):
            correct += 1
        elif normalize_answer(p_ans) and normalize_answer(p_ans) in normalize_answer(r):
            correct += 1
    return {"exact_or_substring_match_accuracy": correct / total if total else 0.0, "n": total}


def compute_rouge_bleu(preds, refs):
    result = {}
    try:
        import evaluate as hf_evaluate
        rouge = hf_evaluate.load("rouge")
        rouge_scores = rouge.compute(predictions=preds, references=refs)
        result["rouge"] = rouge_scores
    except Exception as e:
        print(f"[warn] ROUGE unavailable: {e}")

    try:
        import sacrebleu
        bleu = sacrebleu.corpus_bleu(preds, [refs])
        result["bleu"] = bleu.score
    except Exception as e:
        print(f"[warn] BLEU unavailable: {e}")

    return result


def compute_bertscore(preds, refs):
    try:
        from bert_score import score as bert_score_fn
        P, R, F1 = bert_score_fn(preds, refs, lang="en", rescale_with_baseline=True)
        return {
            "precision": float(P.mean()),
            "recall": float(R.mean()),
            "f1": float(F1.mean()),
        }
    except Exception as e:
        print(f"[warn] BERTScore unavailable: {e}")
        return None


def compute_cot_diagnostics(preds):
    n = len(preds)
    n_with_reasoning = 0
    lengths = []
    for p in preds:
        r = extract_block(p, "reasoning")
        if r:
            n_with_reasoning += 1
            lengths.append(len(r.split()))
    avg_len = sum(lengths) / len(lengths) if lengths else 0.0
    return {
        "cot_presence_rate": n_with_reasoning / n if n else 0.0,
        "avg_reasoning_length_words": avg_len,
    }


def compute_hallucination_proxy(preds, questions, max_examples=100):
    """
    Lightweight, automatic hallucination signal: for each generated
    reasoning block, split into sentences and check NLI entailment of
    each sentence against the source question/vignette. Sentences
    labeled 'contradiction' (or, as a proxy for unsupported claims,
    'neutral' with low entailment probability) are counted as
    potentially unsupported/hallucinated claims.

    NOTE: this is a heuristic proxy, not a clinical fact-checker -- it
    flags reasoning that isn't textually grounded in the given vignette,
    which correlates with (but isn't identical to) factual hallucination.
    """
    try:
        from transformers import pipeline
    except Exception as e:
        print(f"[warn] transformers pipeline unavailable for hallucination proxy: {e}")
        return None

    try:
        nli = pipeline("text-classification", model="roberta-large-mnli")
    except Exception as e:
        print(f"[warn] could not load NLI model for hallucination proxy ({e}); skipping.")
        return None

    total_sentences = 0
    unsupported = 0
    for p, q in list(zip(preds, questions))[:max_examples]:
        reasoning = extract_block(p, "reasoning")
        if not reasoning:
            continue
        sentences = re.split(r"(?<=[.!?])\s+", reasoning)
        for s in sentences:
            s = s.strip()
            if len(s.split()) < 4:
                continue
            total_sentences += 1
            try:
                pred = nli(f"{q} </s></s> {s}", truncation=True)[0]
                if pred["label"].upper() in ("CONTRADICTION",) or (
                    pred["label"].upper() == "NEUTRAL" and pred["score"] > 0.6
                ):
                    unsupported += 1
            except Exception:
                continue

    rate = unsupported / total_sentences if total_sentences else None
    return {
        "unsupported_claim_rate": rate,
        "sentences_checked": total_sentences,
        "examples_checked": min(max_examples, len(preds)),
    }


def compute_latency(latencies):
    if not latencies:
        return {}
    return {
        "avg_latency_sec_per_example": sum(latencies) / len(latencies),
        "n": len(latencies),
    }


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", type=str, default="google/gemma-2-2b-it")
    ap.add_argument("--model_path", type=str, required=True,
                     help="Path to LoRA adapter directory (or merged model dir)")
    ap.add_argument("--test_file", type=str, required=True)
    ap.add_argument("--report_path", type=str, default="eval/report.json")
    ap.add_argument("--n_samples", type=int, default=200)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--skip_hallucination_proxy", action="store_true")
    args = ap.parse_args()

    rows = load_test_set(args.test_file, args.n_samples)
    questions = [r["question"] for r in rows]
    references = [r["answer"] for r in rows]
    reference_texts = [f"<reasoning>{r['reasoning']}</reasoning>\n<answer>{r['answer']}</answer>" for r in rows]

    print(f"Loaded {len(rows)} test examples.")
    print("Loading model for evaluation ...")

    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=2048,
        load_in_4bit=True,
    )
    model.load_adapter(args.model_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Generating predictions ...")
    gen_results = generate_batch(
        model, tokenizer, questions,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
    )
    preds = [g[0] for g in gen_results]
    latencies = [g[1] for g in gen_results]

    print("Computing metrics ...")
    report = {
        "n_examples": len(rows),
        "exact_match": compute_exact_match(preds, references),
        "lexical_overlap": compute_rouge_bleu(preds, reference_texts),
        "semantic_similarity_bertscore": compute_bertscore(preds, reference_texts),
        "cot_diagnostics": compute_cot_diagnostics(preds),
        "latency": compute_latency(latencies),
    }

    if not args.skip_hallucination_proxy:
        print("Computing hallucination proxy (NLI-based) ...")
        report["hallucination_proxy"] = compute_hallucination_proxy(preds, questions)

    Path(args.report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n=== EVALUATION SUMMARY ===")
    print(json.dumps(report, indent=2))
    print(f"\nFull report written to {args.report_path}")


if __name__ == "__main__":
    main()
