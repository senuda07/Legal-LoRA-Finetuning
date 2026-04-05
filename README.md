# Legal LoRA Fine-Tuning

Fine-tuning pipeline for **Mistral-Small-24B** on legislative summarisation tasks using QLoRA. Three training strategies are implemented and compared: supervised fine-tuning (SFT), direct preference optimisation (DPO), and a combined SFT → DPO pipeline.

---

## Project Structure

```
legal-lora-finetuning/
├── data_engineering/
│   ├── build_hf_dataset.py          # Push processed dataset to HuggingFace Hub
│   ├── chunking_test_data.py        # Chunk raw test documents into fixed-size segments
│   ├── chunking_train_data.py       # Chunk raw train documents into fixed-size segments
│   ├── extract_english_data.py      # Filter and extract English-language documents
│   └── split_dataset.py             # Train/test split
│
├── model_finetuning/
│   ├── lora_finetuning/
│   │   └── mistral24B_qlora_finetune.py      # QLoRA SFT pipeline
│   ├── dpo_finetuning/
│   │   └── mistral24B_dpo_finetune.py        # QLoRA DPO (IPO) pipeline
│   └── lora-dpo_finetuning/
│       └── mistral24B_lora-dpo_finetune.py   # Combined SFT → DPO pipeline
│
├── performance_metrics/
│   ├── bertscore_calculation/       # BERTScore evaluation notebooks per model
│   └── rouge_score_calculation/     # ROUGE score evaluation notebooks per model
│
└── summary_generation/              # Generated summaries per model variant
    ├── mistral24B_summary_generation/
    ├── lora_mistral24B_summary_generation/
    ├── dpo_mistral24B_summary_generation/
    ├── lora-dpo_mistral24B_summary_generation/
    ├── openai_summary_generation/
    └── qwen32B_summary_generation/
```

---

## Training Pipelines

### 1. QLoRA SFT — `mistral24B_qlora_finetune.py`

Supervised fine-tuning using paired (input chunk → preferred summary) text files. The model learns to imitate the preferred responses directly.

**Data format:** Two directories of `.txt` files matched by filename stem.
```
inputs/doc001.txt        → chunked legislative text
preferred_responses/doc001.txt  → GPT-4o-mini generated summary
```

**Key config:**
| Parameter | Value |
|-----------|-------|
| Base model | `mistralai/Mistral-Small-24B-Instruct-2501` |
| Quantisation | 4-bit NF4 + double quant |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Learning rate | 2e-4 |
| Effective batch size | 16 (2 × 8 grad accum) |
| Epochs | 3 |
| Max sequence length | 2048 |

---

### 2. QLoRA DPO (IPO) — `mistral24B_dpo_finetune.py`

Preference optimisation using triplets of (input, preferred response, rejected response). Uses **Identity Preference Optimisation (IPO)** over standard DPO for more stable training on small legal datasets.

> IPO regularises on the preference probability directly rather than on the log-ratio, which prevents reward over-optimisation on domain-specific data with subtle preference margins (Azar et al., 2023).

**Data format:** Three directories of `.txt` files matched by filename stem.
```
chunked_train_data/doc001.txt           → input chunk
openai_summaries_train_chunked/doc001.txt   → preferred (GPT-4o-mini)
mistral24B_summaries_train_chunked/doc001.txt → rejected (base Mistral)
```

**Key config:**
| Parameter | Value |
|-----------|-------|
| Base model | `mistralai/Mistral-Small-24B-Instruct-2501` |
| Loss type | `ipo` |
| Beta | 0.1 |
| Learning rate | 5e-5 |
| Reference model | Shared (saves ~12 GB VRAM) |

---

### 3. SFT → DPO — `mistral24B_lora-dpo_finetune.py`

Two-stage pipeline: loads the SFT adapter from HuggingFace Hub, merges it into the base model with `merge_and_unload()`, then runs DPO on top. This gives the DPO stage a stronger starting policy than the raw base model.

```
Base Model
    └── merge SFT adapter (senuda07/legal-mistral-qlora)
            └── DPO fine-tuning (IPO loss)
                    └── senuda07/legal-mistral-sft-dpo
```

---

## Data Engineering

| Script | Purpose |
|--------|---------|
| `extract_english_data.py` | Filters source documents to English only |
| `chunking_train_data.py` | Splits train documents into fixed-size chunks |
| `chunking_test_data.py` | Splits test documents into fixed-size chunks |
| `split_dataset.py` | Produces train/test splits |
| `build_hf_dataset.py` | Packages and pushes the dataset to HuggingFace Hub |

---

## Evaluation

Models are evaluated on held-out test summaries using two metrics:

### ROUGE Score
Measures n-gram overlap between generated and reference summaries. Notebooks under `performance_metrics/rouge_score_calculation/`.

| Notebook | Model |
|----------|-------|
| `mistral24B_rouge_score.ipynb` | Base Mistral-Small-24B |
| `lora_mistral24B_rouge_score.ipynb` | + SFT adapter |
| `dpo_mistral24B_rouge_score.ipynb` | + DPO adapter |
| `lora-dpo_mistral24B_rouge_score.ipynb` | + SFT → DPO adapter |
| `qwen32B_rouge_score.ipynb` | Qwen-32B baseline |

### BERTScore
Measures semantic similarity using contextual embeddings. Notebooks under `performance_metrics/bertscore_calculation/` — same set of models.

---

## Setup

### Requirements

```bash
pip install transformers peft trl bitsandbytes datasets accelerate huggingface_hub
```

### HuggingFace Authentication

All scripts read the token from the `HF_TOKEN` environment variable:

```bash
export HF_TOKEN=your_token_here
```

### VRAM Requirements

Tested on 80 GB A100. Approximate VRAM usage per pipeline:

| Component | VRAM |
|-----------|------|
| 4-bit NF4 base model (24B) | ~12–14 GB |
| LoRA adapters + optimiser | ~4–6 GB |
| Activations (grad checkpointing on) | ~8–10 GB |
| **Total** | **~28–32 GB** |

> Gradient checkpointing and `paged_adamw_8bit` are enabled by default to keep peak usage well within 80 GB.

---

## Running a Pipeline

### SFT

```bash
python model_finetuning/lora_finetuning/mistral24B_qlora_finetune.py
```

### DPO

```bash
python model_finetuning/dpo_finetuning/mistral24B_dpo_finetune.py
```

### SFT → DPO

```bash
# Ensure the SFT adapter is already pushed to Hub before running this
python model_finetuning/lora-dpo_finetuning/mistral24B_lora-dpo_finetune.py
```

All pipelines end with an inference smoke test that prints a sample generation to confirm the adapter is working correctly.

---

## HuggingFace Hub

| Adapter | Repo |
|---------|------|
| SFT | `senuda07/legal-mistral-qlora` |
| SFT → DPO | `senuda07/legal-mistral-sft-dpo` |

### Loading a saved adapter

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

base = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-Small-24B-Instruct-2501",
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

model = PeftModel.from_pretrained(base, "senuda07/legal-mistral-sft-dpo")
tokenizer = AutoTokenizer.from_pretrained("senuda07/legal-mistral-sft-dpo")
```

---

## Models Compared

| Model | Description |
|-------|-------------|
| `mistral24B` | Base Mistral-Small-24B, no fine-tuning |
| `lora_mistral24B` | + QLoRA SFT on preferred summaries |
| `dpo_mistral24B` | + QLoRA DPO (IPO) on preference triplets |
| `lora-dpo_mistral24B` | + SFT adapter merged, then DPO |
| `openai` | GPT-4o-mini generated summaries (reference) |
| `qwen32B` | Qwen-32B baseline |