"""
merge_and_export.py
====================
Merges a trained LoRA adapter into the base Gemma-2-2B-it weights to
produce a standalone, deployment-ready model directory (no PEFT
dependency needed at serving time). Optionally also saves a
16-bit-merged checkpoint suitable for further GGUF conversion via
llama.cpp if you want to serve with llama.cpp / ollama.

Usage:
    python scripts/merge_and_export.py \
        --adapter outputs/clinreason-2b/final_adapter \
        --out merged/clinreason-2b-merged
"""
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", type=str, default="google/gemma-2-2b-it")
    ap.add_argument("--adapter", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--save_method", type=str, default="merged_16bit",
                     choices=["merged_16bit", "merged_4bit", "lora"],
                     help="unsloth save_pretrained_merged save_method")
    args = ap.parse_args()

    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=2048,
        load_in_4bit=(args.save_method != "merged_16bit"),
    )
    model.load_adapter(args.adapter)

    print(f"Merging and saving to {args.out} (method={args.save_method}) ...")
    model.save_pretrained_merged(args.out, tokenizer, save_method=args.save_method)
    print("Done.")


if __name__ == "__main__":
    main()
