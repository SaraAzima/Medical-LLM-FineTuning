"""
train.py
========
Fine-tunes google/gemma-2-2b-it with QLoRA using Unsloth for fast,
memory-efficient training on the CoT-formatted clinical QA data
produced by data/data_prep.py.

Usage:
    python scripts/train.py --config configs/train_config.yaml
"""
import argparse
import os

import yaml
from datasets import load_dataset


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/train_config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    # Imports are inside main() so `python scripts/train.py --help`
    # doesn't require a GPU / unsloth installed just to read args.
    import torch
    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig

    m_cfg = cfg["model"]
    l_cfg = cfg["lora"]
    d_cfg = cfg["data"]
    t_cfg = cfg["training"]

    print(f"Loading base model {m_cfg['base_model']} (4bit={m_cfg['load_in_4bit']}) ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=m_cfg["base_model"],
        max_seq_length=m_cfg["max_seq_length"],
        dtype=m_cfg["dtype"],
        load_in_4bit=m_cfg["load_in_4bit"],
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=l_cfg["r"],
        target_modules=l_cfg["target_modules"],
        lora_alpha=l_cfg["lora_alpha"],
        lora_dropout=l_cfg["lora_dropout"],
        bias=l_cfg["bias"],
        use_gradient_checkpointing=l_cfg["use_gradient_checkpointing"],
        random_state=l_cfg["random_state"],
    )

    print("Loading formatted datasets ...")
    data_files = {"train": d_cfg["train_file"], "validation": d_cfg["val_file"]}
    dataset = load_dataset("json", data_files=data_files)

    os.makedirs(t_cfg["output_dir"], exist_ok=True)

    sft_config = SFTConfig(
        output_dir=t_cfg["output_dir"],
        num_train_epochs=t_cfg["num_train_epochs"],
        per_device_train_batch_size=t_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=t_cfg["gradient_accumulation_steps"],
        per_device_eval_batch_size=t_cfg["per_device_eval_batch_size"],
        eval_strategy=t_cfg["eval_strategy"],
        eval_steps=t_cfg["eval_steps"],
        save_strategy=t_cfg["save_strategy"],
        save_steps=t_cfg["save_steps"],
        save_total_limit=t_cfg["save_total_limit"],
        logging_steps=t_cfg["logging_steps"],
        learning_rate=t_cfg["learning_rate"],
        lr_scheduler_type=t_cfg["lr_scheduler_type"],
        warmup_ratio=t_cfg["warmup_ratio"],
        weight_decay=t_cfg["weight_decay"],
        optim=t_cfg["optim"],
        seed=t_cfg["seed"],
        bf16=t_cfg["bf16"],
        fp16=t_cfg["fp16"],
        report_to=t_cfg["report_to"],
        dataset_text_field=d_cfg["text_field"],
        max_seq_length=m_cfg["max_seq_length"],
        packing=d_cfg["packing"],
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        args=sft_config,
    )

    print("Starting training ...")
    trainer.train()

    final_dir = os.path.join(t_cfg["output_dir"], "final_adapter")
    print(f"Saving final LoRA adapter to {final_dir}")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print("Training complete.")


if __name__ == "__main__":
    main()
