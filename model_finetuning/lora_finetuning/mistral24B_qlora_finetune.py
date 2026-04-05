import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
)
from trl import SFTTrainer, SFTConfig
from huggingface_hub import login

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

    # Data — both dirs must contain .txt files with matching filenames
    inputs_dir: str = "inputs"                      # original dataset chunks
    responses_dir: str = "preferred_responses"      # GPT-4o-mini train outputs

    # Hub
    hf_repo_id: str = "senuda07/legal-mistral-qlora"
    max_seq_length: int = 2048

    # LoRA
    lora_r: int = 16            # rank — raise to 32 for more capacity
    lora_alpha: int = 32        # scaling factor, convention: 2 × lora_r
    lora_dropout: float = 0.05
    # All attention + SwiGLU-FFN projections — better than attention-only targeting
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # Training
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8   # effective batch size = 16
    num_train_epochs: int = 3
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03

    # Validation
    val_split: float = 0.2  # 20% held out for validation

    # Misc
    output_dir: str = "./checkpoints"
    push_private: bool = False
    seed: int = 42


# Global config instance used throughout the script
cfg = Config()


# 2. HuggingFace Hub authentication

def authenticate_hub(token: Optional[str] = None) -> None:
    resolved_token = token or os.environ.get("HF_TOKEN")
    login(token=resolved_token)
    logger.info("Authenticated with HuggingFace Hub.")


# 3. Dataset loading and prompt formatting

def load_paired_files(inputs_dir: str, responses_dir: str) -> list[dict]:
    # Match files across both dirs by filename stem — skips incomplete pairs with a warning
    input_paths = sorted(Path(inputs_dir).glob("*.txt"))
    if not input_paths:
        raise FileNotFoundError(
            f"No .txt input files found in '{inputs_dir}'. "
            "Check that the directory path is correct."
        )

    pairs, missing = [], []
    for inp_path in input_paths:
        resp_path = Path(responses_dir) / inp_path.name
        if not resp_path.exists():
            missing.append(inp_path.name)
            continue
        pairs.append({
            "input_text": inp_path.read_text(encoding="utf-8").strip(),
            "response_text": resp_path.read_text(encoding="utf-8").strip(),
        })

    if missing:
        logger.warning(
            "%d input file(s) have no matching response and will be skipped: %s",
            len(missing), missing[:10],
        )

    if not pairs:
        raise RuntimeError(
            "No valid (input, response) pairs were found. "
            "Ensure filenames match exactly across both directories."
        )

    logger.info("Loaded %d input-response pairs.", len(pairs))
    return pairs


def build_mistral_chat_prompt(pair: dict, system_prompt: Optional[str] = None) -> str:
    # Formats a single (input, response) pair using Mistral's [INST] chat template.
    # The full assistant turn is included here since this is SFT (not DPO).
    if system_prompt is None:
        system_prompt = (
            "You are an expert legal assistant with deep knowledge of contract law, "
            "litigation, statutory interpretation, and legal drafting. "
            "Analyse the provided legal text carefully and respond accurately, "
            "citing relevant clauses, statutes, or case law where applicable. "
            "Be precise, concise, and professionally rigorous."
        )

    return (
        f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"
        f"{pair['input_text']} [/INST] {pair['response_text']} </s>"
    )


def build_dataset(inputs_dir: str, responses_dir: str) -> Dataset:
    # Builds a single-column Dataset ("text") with all formatted prompts
    pairs = load_paired_files(inputs_dir, responses_dir)
    formatted = [build_mistral_chat_prompt(p) for p in pairs]
    dataset = Dataset.from_dict({"text": formatted})
    logger.info("Dataset created — %d examples total.", len(dataset))
    return dataset


# 4. Train / validation split

def split_dataset(dataset: Dataset, val_split: float, seed: int):
    split = dataset.train_test_split(test_size=val_split, seed=seed, shuffle=True)
    train_ds, val_ds = split["train"], split["test"]
    logger.info(
        "Split — Train: %d samples  |  Validation: %d samples",
        len(train_ds), len(val_ds),
    )
    return train_ds, val_ds


# 5. Tokenizer

def load_tokenizer(model_id: str) -> AutoTokenizer:
    # Mistral has no dedicated pad token, so EOS is reused to satisfy the data collator
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        use_fast=True,
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


# 6. Model loading (4-bit NF4 + BF16 compute)

def build_bnb_config() -> BitsAndBytesConfig:
    # NF4 + double quantisation + BF16 compute — optimal quality/VRAM tradeoff for 24B
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,  # quantises the quant constants too, saves ~0.4 GB
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def load_model(model_id: str, bnb_config: BitsAndBytesConfig) -> AutoModelForCausalLM:
    # Loads the base model in 4-bit and prepares it for QLoRA training
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",          # spreads layers across all available GPUs
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        # attn_implementation="flash_attention_2",  # uncomment if FA-2 is installed
    )

    model.config.use_cache = False      # disable KV cache during training
    model.config.pretraining_tp = 1

    # Enables gradient checkpointing and casts layer-norms to full precision
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True
    )

    vram_gb = model.get_memory_footprint() / (1024 ** 3)
    logger.info("Model loaded — VRAM footprint: %.2f GB", vram_gb)
    return model


# 7. LoRA adapter injection

def apply_lora(model: AutoModelForCausalLM, config: Config):
    # Injects LoRA adapters into all attention + FFN projections and freezes base weights
    lora_cfg = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model, lora_cfg


# 8. Training arguments

def build_training_args(config: Config) -> SFTConfig:
    return SFTConfig(
        # — Training schedule —
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",       # paged optimiser prevents OOM spikes
        learning_rate=config.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=config.warmup_ratio,
        bf16=True,
        fp16=False,

        # — Logging and checkpointing —
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        seed=config.seed,
        push_to_hub=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,

        # — SFT-specific —
        dataset_text_field="text",
        max_length=config.max_seq_length,
        packing=False,
    )


# 9. SFT training loop

def run_training(
    model,
    tokenizer: AutoTokenizer,
    lora_cfg: LoraConfig,
    train_dataset: Dataset,
    val_dataset: Dataset,
    training_args: SFTConfig,
    config: Config,
) -> SFTTrainer:
    # Apply LoRA exactly once before handing the model to the trainer
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
    )

    logger.info("Starting SFT training …")
    train_result = trainer.train()
    logger.info("Training complete. Metrics: %s", train_result.metrics)
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    return trainer


# 10. Evaluation

def evaluate(trainer: SFTTrainer) -> dict:
    # Reports eval_loss and eval_perplexity on the validation set
    logger.info("Evaluating on validation set …")
    metrics = trainer.evaluate()
    logger.info("Eval metrics: %s", metrics)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    return metrics


# 11. Save adapter locally

def save_adapter(trainer: SFTTrainer, tokenizer: AutoTokenizer, output_dir: str) -> str:
    # Only the adapter delta weights are saved (not the full 24B base) — typically 100–400 MB
    adapter_path = os.path.join(output_dir, "final_adapter")
    trainer.model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    logger.info("Adapter saved locally to '%s'.", adapter_path)
    return adapter_path


# 12. Push to HuggingFace Hub

def push_to_hub(trainer: SFTTrainer, tokenizer: AutoTokenizer, config: Config) -> None:
    logger.info("Pushing adapter to HuggingFace Hub — repo: %s", config.hf_repo_id)
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
        "Upload complete. View at: https://huggingface.co/%s", config.hf_repo_id
    )


# 13. Inference smoke test

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

    sample_prompt = (
        "<s>[INST] <<SYS>>\n"
        "You are an expert legal assistant.\n"
        "<</SYS>>\n\n"
        "Summarise the key obligations of the lessee under a standard commercial lease. [/INST]"
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
    return generated


# 14. Main

def main():
    # Step 1 – Authentication
    authenticate_hub()

    # Step 2 – Build dataset from paired txt files
    dataset = build_dataset(cfg.inputs_dir, cfg.responses_dir)

    # Step 3 – Train / val split
    train_dataset, val_dataset = split_dataset(dataset, cfg.val_split, cfg.seed)

    # Step 4 – Tokenizer
    tokenizer = load_tokenizer(cfg.base_model_id)

    # Step 5 – Quantisation config + model
    bnb_config = build_bnb_config()
    model = load_model(cfg.base_model_id, bnb_config)

    # Step 6 – LoRA config (applied inside run_training)
    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # Step 7 – Training arguments
    training_args = build_training_args(cfg)

    # Step 8 – Train
    trainer = run_training(
        model=model,
        tokenizer=tokenizer,
        lora_cfg=lora_cfg,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        training_args=training_args,
        config=cfg,
    )

    # Step 9 – Evaluate on validation set
    evaluate(trainer)

    # Step 10 – Save adapter locally
    adapter_path = save_adapter(trainer, tokenizer, cfg.output_dir)

    # Step 11 – Push to Hub
    push_to_hub(trainer, tokenizer, cfg)

    # Step 12 – Smoke test
    logger.info("Running inference smoke test …")
    response = run_inference_smoke_test(cfg.base_model_id, adapter_path, bnb_config)
    print("\n=== Smoke Test Response ===")
    print(response)
    print("===========================\n")
    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()