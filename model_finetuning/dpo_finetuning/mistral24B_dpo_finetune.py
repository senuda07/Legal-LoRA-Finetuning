# Imports

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch
from datasets import Dataset
from huggingface_hub import login
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import DPOConfig, DPOTrainer

# Logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# 1. Central configuration

@dataclass
class Config:

    # Model 
    # HuggingFace model repo used as the base
    base_model_id: str = "mistralai/Mistral-Small-24B-Instruct-2501"

    # Data 
    # Each directory must contain .txt files with matching filenames across all three
    inputs_dir: str = "chunked_train_data"
    preferred_dir: str = "openai_summaries_train_chunked"
    rejected_dir: str = "mistral24B_summaries_train_chunked"

    # Hub 
    # HuggingFace Hub destination for the fine-tuned adapter
    hf_repo_id: str = "senuda07/legal-mistral-dpo"

    # Sequence lengths 
    # Keep max_prompt_length well below max_length to leave room for
    # both chosen and rejected completions inside the DPO context window.
    max_prompt_length: int = 1024
    max_length: int = 2048

    # LoRA
    lora_r: int = 16            # rank — raise to 32 for more capacity
    lora_alpha: int = 32        # scaling factor, convention: 2 × lora_r
    lora_dropout: float = 0.05  # dropout for regularisation
    # Target all attention + SwiGLU-FFN projections for best task coverage
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # Training
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8   # → effective batch size = 16
    num_train_epochs: int = 3
    learning_rate: float = 5e-5            # IPO prefers a lower LR than SFT
    warmup_ratio: float = 0.05

    # IPO / DPO hyper-params
    # loss_type="ipo" selects Identity Preference Optimisation.
    # beta=0.1 is conservative and stable for legal summarisation tasks.
    beta: float = 0.1
    loss_type: str = "ipo"

    # Validation split
    val_split: float = 0.2   # 20% held out for validation

    # Misc 
    push_private: bool = False
    seed: int = 42


# Instantiate the global config used throughout the script
cfg = Config()



# 2. HuggingFace Hub authentication

def authenticate_hub(token: Optional[str] = None) -> None:
    # Log into HuggingFace Hub. WRITE permission is required to push adapters.
    # Store your token in HF_TOKEN env var — never hard-code it here.
    resolved_token = token or os.environ.get("HF_TOKEN")
    login(token=resolved_token)
    logger.info("Authenticated with HuggingFace Hub.")


# 3. Dataset loading — triplet file pairs

# Load (input, preferred, rejected) triplets from the specified directories.
def load_triplet_files(
    inputs_dir: str,
    preferred_dir: str,
    rejected_dir: str,
) -> list[dict]:
    input_paths = sorted(Path(inputs_dir).glob("*.txt"))
    if not input_paths:
        raise FileNotFoundError(
            f"No .txt input files found in '{inputs_dir}'. "
            "Verify the directory path is correct."
        )

    triplets, missing = [], []
    for inp_path in input_paths:
        pref_path = Path(preferred_dir) / inp_path.name
        rej_path = Path(rejected_dir) / inp_path.name

        # Skip this file if either counterpart is missing
        if not pref_path.exists() or not rej_path.exists():
            missing.append(inp_path.name)
            continue

        triplets.append({
            "input_text": inp_path.read_text(encoding="utf-8").strip(),
            "preferred_text": pref_path.read_text(encoding="utf-8").strip(),
            "rejected_text": rej_path.read_text(encoding="utf-8").strip(),
        })

    if missing:
        logger.warning(
            "%d input file(s) have incomplete triplets and will be skipped: %s",
            len(missing), missing[:10],
        )

    if not triplets:
        raise RuntimeError(
            "No valid (input, preferred, rejected) triplets were found. "
            "Ensure filenames match exactly across all three directories."
        )

    logger.info("Loaded %d input-preferred-rejected triplets.", len(triplets))
    return triplets


# 4. Prompt formatting — Mistral chat template

def build_mistral_prompt(input_text: str, system_prompt: Optional[str] = None) -> str:

    if system_prompt is None:
        system_prompt = (
            """You are a legal research assistant.

            You are a legal research assistant specialized in summarizing legislative documents.
            The included text is part of a larger summary.

            Create accurate, concise, and short summaries in plain English.
            Preserve key legal clauses, definitions, obligations, and relationships.
            Do not introduce new information.
            Avoid unnecessary simplification.
            Maintain legal terminology where important.
            
            Text:
            """
        )

    return (
        f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"
        f"{input_text} [/INST]"
    )


def build_dpo_dataset(
    inputs_dir: str,
    preferred_dir: str,
    rejected_dir: str,
) -> Dataset:

    triplets = load_triplet_files(inputs_dir, preferred_dir, rejected_dir)

    records = []
    for t in triplets:
        records.append({
            "prompt": build_mistral_prompt(t["input_text"]),
            "chosen": t["preferred_text"],
            "rejected": t["rejected_text"],
        })

    dataset = Dataset.from_list(records)
    logger.info("DPO dataset created — %d examples total.", len(dataset))
    return dataset


# 5. Train / validation split

def split_dataset(dataset: Dataset, val_split: float, seed: int):
    # Split dataset into train and validation subsets with reproducible shuffling
    split = dataset.train_test_split(test_size=val_split, seed=seed, shuffle=True)
    train_ds, val_ds = split["train"], split["test"]
    logger.info(
        "Split — Train: %d samples  |  Validation: %d samples",
        len(train_ds), len(val_ds),
    )
    return train_ds, val_ds


# 6. Tokenizer

def load_tokenizer(model_id: str) -> AutoTokenizer:
    # Load and configure the tokenizer.
    # Mistral has no dedicated pad token, so EOS is reused to satisfy
    # the data collator. Right-padding is safer for causal attention masks.

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        use_fast=False,
    )

    # Reuse EOS as the pad token since Mistral has none
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Right-padding is safer than left-padding for causal attention masks
    tokenizer.padding_side = "right"

    logger.info(
        "Tokenizer loaded — vocab size: %d  |  pad token: '%s'",
        tokenizer.vocab_size,
        tokenizer.pad_token,
    )
    return tokenizer



# 7. BitsAndBytes 4-bit config

def build_bnb_config() -> BitsAndBytesConfig:
    # Build the 4-bit NF4 quantisation config for VRAM efficiency.
    #
    # Key choices:
    #   nf4          — Normal Float 4-bit; optimal for neural network weights
    #   double_quant — Quantises the quant constants too, saves ~0.4 GB
    #   bfloat16     — Dequantises to BF16 for stable forward/backward pass

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


# 8. Model loading (4-bit NF4 + BF16 compute)

def load_model(model_id: str, bnb_config: BitsAndBytesConfig) -> AutoModelForCausalLM:

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    model.config.use_cache = False
    model.config.pretraining_tp = 1
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True
    )

    vram_gb = model.get_memory_footprint() / (1024 ** 3)
    logger.info("Model loaded — VRAM footprint: %.2f GB", vram_gb)
    return model


# 9. LoRA adapter configuration

# Build the LoRA adapter config for QLoRA DPO.

def build_lora_config(config: Config) -> LoraConfig:

    lora_cfg = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    logger.info(
        "LoRA config — r=%d  alpha=%d  dropout=%.2f  modules=%s",
        config.lora_r, config.lora_alpha, config.lora_dropout,
        config.target_modules,
    )
    return lora_cfg



# 10. DPO training arguments

def build_dpo_config(config: Config) -> DPOConfig:

    return DPOConfig(
        # — Output and logging —
        output_dir=config.output_dir,
        logging_steps=10,
        report_to="none",

        # — Training schedule —
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=config.warmup_ratio,

        # — VRAM optimisation —
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        dataloader_num_workers=4,
        dataloader_pin_memory=True,

        # — Checkpointing — save every 50 steps, keep best 3 by eval_loss —
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",

        # — IPO / DPO hyper-params —
        beta=config.beta,
        loss_type=config.loss_type,    # "ipo"

        # — Misc —
        seed=config.seed,
        push_to_hub=False,
        remove_unused_columns=False,
    )


# 11. DPO training loop

def run_dpo_training(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    lora_cfg: LoraConfig,
    train_dataset: Dataset,
    val_dataset: Dataset,
    dpo_cfg: DPOConfig,
) -> DPOTrainer:

    trainer = DPOTrainer(
        model=model,
        ref_model=None,         # share base as reference (saves ~12 GB VRAM)
        args=dpo_cfg,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        peft_config=lora_cfg,   # DPOTrainer applies LoRA internally
    )

    # Log trainable parameter count after LoRA injection
    trainer.model.print_trainable_parameters()

    logger.info("Starting IPO/DPO training …")
    train_result = trainer.train()
    logger.info("Training complete.  Metrics: %s", train_result.metrics)
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)

    return trainer


# 12. Validation evaluation

# Run evaluation on the validation set and log metrics.
def evaluate(trainer: DPOTrainer) -> dict:
    logger.info("Evaluating on validation set …")
    metrics = trainer.evaluate()
    logger.info("Eval metrics: %s", metrics)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    return metrics


# 13. Save adapter locally

def save_adapter(
    trainer: DPOTrainer,
    tokenizer: AutoTokenizer,
    output_dir: str,
) -> str:
    # Save the LoRA adapter weights and tokenizer to a local directory.
    # Only the adapter delta weights are saved (not the full 24B base),
    # keeping the checkpoint compact — typically 100–400 MB.

    adapter_path = os.path.join(output_dir, "final_adapter")
    trainer.model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    logger.info("Adapter saved locally to '%s'.", adapter_path)
    return adapter_path



# 14. Push to HuggingFace Hub

 # Push the fine-tuned LoRA adapter and tokenizer to HuggingFace Hub.

def push_to_hub(
    trainer: DPOTrainer,
    tokenizer: AutoTokenizer,
    config: Config,
) -> None:

    logger.info(
        "Pushing adapter to HuggingFace Hub — repo: %s", config.hf_repo_id
    )
    trainer.model.push_to_hub(
        config.hf_repo_id,
        use_auth_token=True,
        private=config.push_private,
    )
    tokenizer.push_to_hub(
        config.hf_repo_id,
        use_auth_token=True,
        private=config.push_private,
    )
    logger.info(
        "Upload complete.  View at: https://huggingface.co/%s", config.hf_repo_id
    )


# 15. Inference smoke test

def run_inference_smoke_test(
    base_model_id: str,
    adapter_path: str,
    bnb_config: BitsAndBytesConfig,
    max_new_tokens: int = 256,
) -> str:
    # Load the saved adapter and run a sample inference to verify the output.

    tok = AutoTokenizer.from_pretrained(adapter_path)

    base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    peft_model = PeftModel.from_pretrained(base, adapter_path)
    peft_model.eval()

    sample_prompt = build_mistral_prompt(
        "Summarise the key obligations of the lessee under a standard "
        "commercial lease, including rent payment, maintenance duties, "
        "and obligations upon lease termination."
    )

    inputs = tok(sample_prompt, return_tensors="pt").to(peft_model.device)
    with torch.no_grad():
        output_ids = peft_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,    # greedy decoding for a deterministic smoke test
            temperature=1.0,
        )

    # Slice off the prompt tokens and decode only the generated portion
    generated = tok.decode(
        output_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    logger.info("Smoke test generation complete.")
    return generated


# 16. Main entry point

def main():

    # Step 1 – Authentication
    authenticate_hub()

    # Step 2 – Dataset
    dataset = build_dpo_dataset(
        cfg.inputs_dir,
        cfg.preferred_dir,
        cfg.rejected_dir,
    )

    # Step 3 – Train / val split
    train_dataset, val_dataset = split_dataset(dataset, cfg.val_split, cfg.seed)

    # Step 4 – Tokenizer
    tokenizer = load_tokenizer(cfg.base_model_id)

    # Step 5 & 6 – Build quant config then load model in 4-bit
    bnb_config = build_bnb_config()
    model = load_model(cfg.base_model_id, bnb_config)

    # Step 7 – LoRA config
    lora_cfg = build_lora_config(cfg)

    # Step 8 – DPO training arguments
    dpo_cfg = build_dpo_config(cfg)

    # Step 9 – Train
    trainer = run_dpo_training(
        model=model,
        tokenizer=tokenizer,
        lora_cfg=lora_cfg,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        dpo_cfg=dpo_cfg,
    )

    # Step 10 – Evaluate
    evaluate(trainer)

    # Step 11 – Save locally
    adapter_path = save_adapter(trainer, tokenizer, cfg.output_dir)

    # Step 12 – Push to Hub
    push_to_hub(trainer, tokenizer, cfg)

    # Step 13 – Smoke test
    logger.info("Running inference smoke test …")
    response = run_inference_smoke_test(cfg.base_model_id, adapter_path, bnb_config)
    print("\n=== Smoke Test Response ===")
    print(response)
    print("===========================\n")
    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()