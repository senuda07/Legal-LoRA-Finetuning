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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# 1. Central configuration

@dataclass
class Config:
    # All hyper-parameters and file paths live here — nothing is hard-coded elsewhere.

    # Model
    base_model_id: str = "mistralai/Mistral-Small-24B-Instruct-2501"
    adapter_name: str = "senuda07/legal-mistral-qlora"  # SFT adapter to merge before DPO

    # Data — all three dirs must contain .txt files with matching filenames
    inputs_dir: str = "chunked_train_data"
    preferred_dir: str = "openai_summaries_train_chunked"
    rejected_dir: str = "mistral24B_summaries_train_chunked"

    # Hub
    hf_repo_id: str = "senuda07/legal-mistral-sft-dpo"

    # Sequence lengths — keep max_prompt_length well below max_length to leave room for completions
    max_prompt_length: int = 1024
    max_length: int = 2048

    # LoRA
    lora_r: int = 16            
    lora_alpha: int = 32        
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # Training
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8   
    num_train_epochs: int = 3
    learning_rate: float = 5e-5          
    warmup_ratio: float = 0.05

    # IPO / DPO hyper-params
    beta: float = 0.1       # conservative regularisation, stable for legal tasks
    loss_type: str = "ipo"  # use "sigmoid" for standard DPO, "hinge", or "kto_pair"

    # Validation
    val_split: float = 0.2  # 20% held out for validation

    # Misc
    output_dir: str = "./dpo_checkpoints"
    push_private: bool = False
    seed: int = 42


# Global config instance used throughout the script
cfg = Config()


# 2. HuggingFace Hub authentication

def authenticate_hub(token: Optional[str] = None) -> None:
    resolved_token = token or os.environ.get("HF_TOKEN")
    login(token=resolved_token)
    logger.info("Authenticated with HuggingFace Hub.")


# 3. Dataset loading — triplet file pairs

def load_triplet_files(
    inputs_dir: str,
    preferred_dir: str,
    rejected_dir: str,
) -> list[dict]:
    # Match files across all three dirs by filename stem — skips incomplete triplets with a warning
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

        # Skip if either counterpart is missing
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
    # DPOTrainer receives (prompt, chosen, rejected) separately and concatenates internally.
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
    # Builds a HuggingFace Dataset with the three columns DPOTrainer expects: prompt, chosen, rejected
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
    split = dataset.train_test_split(test_size=val_split, seed=seed, shuffle=True)
    train_ds, val_ds = split["train"], split["test"]
    logger.info(
        "Split — Train: %d samples  |  Validation: %d samples",
        len(train_ds), len(val_ds),
    )
    return train_ds, val_ds


# 6. Tokenizer

def load_tokenizer(model_id: str) -> AutoTokenizer:
    # Mistral has no dedicated pad token, so EOS is reused to satisfy the data collator
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        use_fast=False,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.padding_side = "right"  # safer than left-padding for causal attention masks

    logger.info(
        "Tokenizer loaded — vocab size: %d  |  pad token: '%s'",
        tokenizer.vocab_size,
        tokenizer.pad_token,
    )
    return tokenizer


# 7. BitsAndBytes 4-bit config

def build_bnb_config() -> BitsAndBytesConfig:
    # NF4 + double quantisation + BF16 compute — optimal quality/VRAM tradeoff for 24B
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, 
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


# 8. Model loading (4-bit NF4 + BF16 compute)

def load_model(model_id: str, bnb_config: BitsAndBytesConfig, adapter_name: str = None) -> AutoModelForCausalLM:
    # Loads the base model in 4-bit. If adapter_name is given, merges the SFT adapter before DPO.
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",          # spreads layers across all available GPUs
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    if adapter_name:
        logger.info("Loading SFT adapter '%s' and merging into base model …", adapter_name)
        model = PeftModel.from_pretrained(model, adapter_name)
        model = model.merge_and_unload()  # returns a plain AutoModelForCausalLM
        logger.info("SFT adapter merged and unloaded.")

    model.config.use_cache = False      # disable KV cache during training
    model.config.pretraining_tp = 1

    # Enables gradient checkpointing and casts layer-norms to full precision
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True
    )

    vram_gb = model.get_memory_footprint() / (1024 ** 3)
    logger.info("Model loaded — VRAM footprint: %.2f GB", vram_gb)
    return model


# 9. LoRA adapter configuration

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
        optim="paged_adamw_8bit",       # paged optimiser prevents OOM spikes
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
        loss_type=config.loss_type,

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
    # ref_model=None reuses the base model as reference in a frozen no_grad pass — saves ~12 GB VRAM
    # LoRA is applied internally by DPOTrainer via peft_config
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_cfg,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        peft_config=lora_cfg,
    )

    trainer.model.print_trainable_parameters()

    logger.info("Starting IPO/DPO training …")
    train_result = trainer.train()
    logger.info("Training complete.  Metrics: %s", train_result.metrics)
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)

    return trainer


# 12. Validation evaluation

def evaluate(trainer: DPOTrainer) -> dict:
    # Reports eval_loss, reward margins, and accuracy (chosen > rejected rate)
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
    # Only the adapter delta weights are saved (not the full 24B base) — typically 100–400 MB
    adapter_path = os.path.join(output_dir, "final_adapter")
    trainer.model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    logger.info("Adapter saved locally to '%s'.", adapter_path)
    return adapter_path


# 14. Push to HuggingFace Hub

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
    # Quick sanity check — loads the saved adapter and runs one sample generation
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

    # Slice off the prompt tokens, decode only the newly generated part
    generated = tok.decode(
        output_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    logger.info("Smoke test generation complete.")
    return generated


# 16. Main

def main():
    # Step 1 – Authentication
    authenticate_hub()

    # Step 2 – Build dataset from triplet txt files
    dataset = build_dpo_dataset(
        cfg.inputs_dir,
        cfg.preferred_dir,
        cfg.rejected_dir,
    )

    # Step 3 – Train / val split
    train_dataset, val_dataset = split_dataset(dataset, cfg.val_split, cfg.seed)

    # Step 4 – Tokenizer
    tokenizer = load_tokenizer(cfg.base_model_id)

    # Step 5 & 6 – Quantisation config + model (merges SFT adapter if cfg.adapter_name is set)
    bnb_config = build_bnb_config()
    model = load_model(cfg.base_model_id, bnb_config, cfg.adapter_name)

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

    # Step 10 – Evaluate on validation set
    evaluate(trainer)

    # Step 11 – Save adapter locally
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