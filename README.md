# ClinReason-2B
**Chain-of-Thought Clinical Reasoning & QA via QLoRA Fine-Tuning of Gemma-2-2B-it**

## 1. Project Overview

ClinReason-2B adapts `google/gemma-2-2b-it` into a domain-specialized clinical
reasoning assistant. The model is fine-tuned with **QLoRA** (4-bit
quantization + LoRA adapters, via **Unsloth**) to answer medical questions
using explicit **chain-of-thought (CoT)** reasoning before committing to a
final answer.

- **Academic value**: studies domain adaptation of a small (2B) general LLM
  into a specialized clinical setting, and whether structured CoT
  supervision reduces hallucination / improves calibration versus
  direct-answer fine-tuning.
- **Industrial value**: the resulting adapter is a candidate backbone for
  online medical Q&A assistants, preliminary/triage screening chat flows,
  and clinical note / dialogue summarization tools.

- **Base model**: `google/gemma-2-2b-it`
- **Training method**: QLoRA (4-bit NF4) with Unsloth's fast kernels
- **Datasets**:
  - `medalpaca/medical_meadow_medqa` — USMLE-style multiple-choice medical
    exam questions (primary CoT reasoning + final-answer supervision source)
  - `FreedomIntelligence/Medical-Dialogue-Dataset` — real doctor–patient
    dialogues (used for conversational/history-taking style and
    patient-facing question answering)

> ⚠️ **Safety disclaimer**: This project is for research/educational
> purposes only. The resulting model is **not** a certified medical device
> and must not be used for real diagnosis or treatment decisions without a
> licensed clinician in the loop.

## 2. Project Structure

```
clinreason/
├── README.md
├── requirements.txt
├── configs/
│   └── train_config.yaml        # all hyperparameters in one place
├── data/
│   └── data_prep.py              # download, clean, unify, CoT-format datasets
├── scripts/
│   ├── train.py                  # Unsloth + QLoRA fine-tuning
│   ├── inference.py               # quick single-prompt / batch inference
│   └── merge_and_export.py       # merge LoRA adapter -> full model / GGUF
└── eval/
    └── evaluate.py                # multi-metric evaluation harness
```

## 3. Setup

```bash
pip install -r requirements.txt
```

`requirements.txt` pins Unsloth, bitsandbytes, PEFT, TRL, transformers,
datasets, and the evaluation libraries (evaluate, rouge-score, bert-score,
scikit-learn).

A single consumer GPU with **≥12–16GB VRAM** (e.g. T4/L4/RTX 3090) is
sufficient for QLoRA fine-tuning of the 2B model thanks to Unsloth's 4-bit
loading and fused kernels.

## 4. Pipeline

### Step 1 — Data preparation
```bash
python data/data_prep.py --output_dir ./data/processed
```
This:
1. Loads `medalpaca/medical_meadow_medqa` and
   `FreedomIntelligence/Medical-Dialogue-Dataset` from the HF Hub.
2. Normalizes both into a single instruction/CoT schema:
   `{"instruction", "input", "reasoning", "answer"}`.
3. For MedQA (which ships answer-only), synthesizes a structured
   **CoT scaffold** from the available options/answer metadata (rule-based
   reasoning trace: restate question → eliminate distractors → state
   answer) so every training example has a `<reasoning>...</reasoning>`
   block. This is documented clearly as a **weak/synthetic CoT signal**
   (see README §7 limitations) — you may swap in a teacher-model-generated
   CoT (e.g. via a larger model) by editing `synthesize_cot()`.
4. Formats every example with Gemma's chat template and a fixed
   system-style instruction that asks the model to output
   `<reasoning>...</reasoning>` then `<answer>...</answer>`.
5. Splits into train/val/test (90/5/5) and saves as JSONL.

### Step 2 — Fine-tuning
```bash
python scripts/train.py --config configs/train_config.yaml
```
Key details (see `configs/train_config.yaml` for all values):
- 4-bit QLoRA via `unsloth.FastLanguageModel`
- LoRA rank 16, alpha 16, dropout 0.05 on all attention + MLP projections
- Packed sequences, `max_seq_length=2048`
- Trains with TRL's `SFTTrainer`, loss masked to only the assistant turn
- Saves LoRA adapter checkpoints + final adapter to `outputs/clinreason-2b/`

### Step 3 — Inference
```bash
python scripts/inference.py \
  --adapter outputs/clinreason-2b \
  --question "A 45-year-old man presents with crushing substernal chest pain radiating to the left arm..."
```

### Step 4 — Merge (optional, for deployment)
```bash
python scripts/merge_and_export.py --adapter outputs/clinreason-2b --out merged/clinreason-2b-merged
```

### Step 5 — Evaluation
```bash
python eval/evaluate.py \
  --model_path outputs/clinreason-2b \
  --test_file data/processed/test.jsonl \
  --report_path eval/report.json
```

## 5. Evaluation Metrics

`eval/evaluate.py` reports, per example and aggregated:

| Category | Metric | What it captures |
|---|---|---|
| Answer correctness (MCQA) | Exact-match accuracy on extracted final answer | Task accuracy on MedQA-style items |
| Lexical overlap | ROUGE-1/2/L, BLEU | Surface-level similarity of generated reasoning/answer to reference |
| Semantic similarity | BERTScore (P/R/F1) | Meaning-level similarity, robust to paraphrase |
| Reasoning quality | CoT-presence rate, avg reasoning length | Whether/how much the model actually reasons before answering |
| Hallucination proxy | Unsupported-claim rate (NLI-based entailment check of each reasoning sentence against the source question/options) | Flags reasoning steps not entailed by the given clinical vignette — a lightweight, automatic hallucination signal |
| Calibration | Answer/no-answer refusal rate, confidence-vs-correctness (if logprobs available) | Whether the model appropriately hedges on uncertain items |
| Efficiency | Tokens/sec, latency | Practical deployability |

All metrics are computed with open, license-friendly libraries
(`evaluate`, `rouge-score`, `bert-score`, `scikit-learn`, plus a small
NLI model such as `roberta-large-mnli` used only for the hallucination
proxy). The script degrades gracefully — if a metric's dependency isn't
importable, it's skipped with a warning rather than crashing the run.

## 6. Reproducibility

- `configs/train_config.yaml` fixes the random seed and all hyperparameters.
- `data/data_prep.py` writes a `dataset_card.json` recording dataset
  versions/revisions, split sizes, and the exact prompt template used.

## 7. Limitations & Responsible-Use Notes

- MedQA CoT traces are synthetically scaffolded, not physician-authored;
  treat the reasoning-quality numbers as an *upper bound* on what
  supervised rationale quality could look like, not ground truth.
- Medical-Dialogue-Dataset conversations are used for style/coverage, not
  as a source of verified clinical fact — the model can still hallucinate.
- This is a research artifact. Do not use outputs for real clinical
  decision-making.

## 8. License
Respect the individual licenses of `google/gemma-2-2b-it`,
`medalpaca/medical_meadow_medqa`, and
`FreedomIntelligence/Medical-Dialogue-Dataset`. This code is provided
as-is for research use.
