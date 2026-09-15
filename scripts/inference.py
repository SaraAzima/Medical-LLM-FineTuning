"""
inference.py
============
Quick single-question inference against a trained LoRA adapter, using
the same CoT prompt format used at training time.

Usage:
    python scripts/inference.py --adapter outputs/clinreason-2b/final_adapter \
        --question "A 58-year-old woman presents with sudden-onset severe headache..."
"""
import argparse

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", type=str, default="google/gemma-2-2b-it")
    ap.add_argument("--adapter", type=str, required=True)
    ap.add_argument("--question", type=str, required=True)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--top_p", type=float, default=0.9)
    args = ap.parse_args()

    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=2048,
        load_in_4bit=True,
    )
    model.load_adapter(args.adapter)
    FastLanguageModel.for_inference(model)

    prompt = build_prompt(args.question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=True,
    )
    text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print("\n=== MODEL OUTPUT ===\n")
    print(text)


if __name__ == "__main__":
    main()
