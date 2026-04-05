import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch
from datasets import Dataset
from huggingface_hub import login
from peft import LoraConfig, PeftModel, TaskType, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class Config:
    # Base model to fine-tune; swap this out if using a different Mistral variant
    base_model_id: str = "mistralai/Mistral-Small-24B-Instruct-2501"

    # Each directory must contain .txt files with matching filenames across all three folders
    inputs_dir: str = "chunked_train_data"
    preferred_dir: str = "openai_summaries_train_chunked"
    rejected_dir: str = "mistral24B_summaries_train_chunked"

    # Hub repo where the final LoRA adapter will be uploaded
    hf_repo_id: str = "senuda07/legal-mistral-dpo"

    # Keep max_prompt_length well below max_length to leave room for both chosen and rejected completions
    max_prompt_length: int = 1024
    max_length: int = 2048

    # LoRA rank and alpha control adapter capacity; alpha=2*r is a standard convention
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    # Targeting all attention + SwiGLU-FFN projections gives better coverage than attention-only
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    per_device_train_batch_size: int = 2
    # Effective batch size = per_device_train_batch_size * gradient_accumulation_steps = 16
    gradient_accumulation_steps: int = 8
    num_train_epochs: int = 3
    # IPO benefits from a lower LR than standard SFT to avoid over-optimising the implicit reward
    learning_rate: float = 5e-5
    warmup_ratio: float = 0.05

    # IPO beta controls regularisation strength; 0.1 is conservative and stable for legal tasks
    beta: float = 0.1
    # "ipo" selects Identity Preference Optimisation; alternatives: "sigmoid" (standard DPO), "hinge", "kto_pair"
    loss_type: str = "ipo"

    val_split: float = 0.2

    output_dir: str = "./dpo_checkpoints"
    push_private: bool = False
    seed: int = 42


cfg = Config()


def authenticate_hub(token: Optional[str] = None) -> None:
    # Reads HF_TOKEN from environment if no token is passed explicitly; requires write permission for Hub push
    resolved_token = token or os.environ.get("HF_TOKEN")
    login(token=resolved_token)
    logger.info("Authenticated with HuggingFace Hub.")


def load_triplet_files(inputs_dir: str, preferred_dir: str, rejected_dir: str) -> list[dict]:
    input_paths = sorted(Path(inputs_dir).glob("*.txt"))
    if not input_paths:
        raise FileNotFoundError(f"No .txt input files found in '{inputs_dir}'.")

    triplets, missing = [], []
    for inp_path in input_paths:
        # Matching is done by filename stem, so all three directories must use identical filenames
        pref_path = Path(preferred_dir) / inp_path.name
        rej_path = Path(rejected_dir) / inp_path.name

        if not pref_path.exists() or not rej_path.exists():
            missing.append(inp_path.name)
            continue

        triplets.append({
            "input_text": inp_path.read_text(encoding="utf-8").strip(),
            "preferred_text": pref_path.read_text(encoding="utf-8").strip(),
            "rejected_text": rej_path.read_text(encoding="utf-8").strip(),
        })

    if missing:
        # Incomplete triplets are skipped rather than failing the entire run
        logger.warning("%d file(s) have incomplete triplets and will be skipped: %s", len(missing), missing[:10])

    if not triplets:
        raise RuntimeError("No valid (input, preferred, rejected) triplets found.")

    logger.info("Loaded %d triplets.", len(triplets))
    return triplets


def build_mistral_prompt(input_text: str, system_prompt: Optional[str] = None) -> str:
    # Prompt contains only system + user turn; DPOTrainer appends chosen/rejected completions internally
    if system_prompt is None:
        system_prompt = (
            "You are a legal research assistant specialized in summarizing legislative documents. "
            "The included text is part of a larger summary. "
            "Create accurate, concise, and short summaries in plain English. "
            "Preserve key legal clauses, definitions, obligations, and relationships. "
            "Do not introduce new information. Avoid unnecessary simplification. "
            "Maintain legal terminology where important.\n\nText:"
        )
    # Mistral chat format: <s>[INST] <<SYS>> system <</SYS>> user [/INST]
    return f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{input_text} [/INST]"


def build_dpo_dataset(inputs_dir: str, preferred_dir: str, rejected_dir: str) -> Dataset:
    triplets = load_triplet_files(inputs_dir, preferred_dir, rejected_dir)
    # DPOTrainer expects exactly three columns: prompt, chosen, rejected
    records = [
        {
            "prompt": build_mistral_prompt(t["input_text"]),
            "chosen": t["preferred_text"],
            "rejected": t["rejected_text"],
        }
        for t in triplets
    ]
    dataset = Dataset.from_list(records)
    logger.info("DPO dataset created — %d examples.", len(dataset))
    return dataset


def split_dataset(dataset: Dataset, val_split: float, seed: int):
    split = dataset.train_test_split(test_size=val_split, seed=seed, shuffle=True)
    train_ds, val_ds = split["train"], split["test"]
    logger.info("Train: %d  |  Validation: %d", len(train_ds), len(val_ds))
    return train_ds, val_ds


def load_tokenizer(model_id: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)

    # Mistral has no dedicated pad token; reusing EOS satisfies the data collator without introducing a new embedding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # Right-padding keeps attention masks correct for causal LM training; left-padding can corrupt loss computation
    tokenizer.padding_side = "right"

    logger.info("Tokenizer loaded — vocab size: %d", tokenizer.vocab_size)
    return tokenizer


def build_bnb_config() -> BitsAndBytesConfig:
    # NF4 (Normal Float 4-bit) is better suited than INT4 for normally-distributed neural network weights
    # double_quant quantises the quantisation constants themselves, saving ~0.4 GB on a 24B model
    # bfloat16 compute dtype dequantises to BF16 for the forward/backward pass; numerically stable on A100/H100
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def load_model(model_id: str, bnb_config: BitsAndBytesConfig) -> AutoModelForCausalLM:
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",          # spreads layers across all available GPUs automatically
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        # attn_implementation="flash_attention_2",  # uncomment if FlashAttention-2 is installed; halves attention VRAM
    )
    model.config.use_cache = False      # KV cache is not needed during training
    model.config.pretraining_tp = 1     # disables tensor parallelism used during pretraining

    # prepare_model_for_kbit_training enables gradient checkpointing and upcasts layer norms
    # to full precision, both of which are required for numerically stable QLoRA training
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    logger.info("Model loaded — VRAM: %.2f GB", model.get_memory_footprint() / (1024 ** 3))
    return model


def build_lora_config(config: Config) -> LoraConfig:
    lora_cfg = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",                # training bias terms alongside LoRA adds overhead with minimal benefit
        task_type=TaskType.CAUSAL_LM,
    )
    logger.info("LoRA — r=%d  alpha=%d  dropout=%.2f", config.lora_r, config.lora_alpha, config.lora_dropout)
    return lora_cfg


def build_dpo_config(config: Config) -> DPOConfig:
    return DPOConfig(
        output_dir=config.output_dir,
        logging_steps=10,
        report_to="none",           # disable W&B / TensorBoard; change to "wandb" if tracking is needed

        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=config.warmup_ratio,

        bf16=True,
        fp16=False,                 # fp16 and bf16 are mutually exclusive; bf16 is preferred on Ampere/Hopper
        gradient_checkpointing=True,            # trades ~20% throughput for ~40% VRAM reduction
        optim="paged_adamw_8bit",               # paged optimiser prevents OOM spikes from optimiser state
        dataloader_num_workers=4,
        dataloader_pin_memory=True,

        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,         # keep only the 3 most recent checkpoints to save disk space
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",

        beta=config.beta,
        loss_type=config.loss_type,

        seed=config.seed,
        push_to_hub=False,
        remove_unused_columns=False,    # must be False; DPOTrainer needs all three dataset columns
    )


def run_dpo_training(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    lora_cfg: LoraConfig,
    train_dataset: Dataset,
    val_dataset: Dataset,
    dpo_cfg: DPOConfig,
) -> DPOTrainer:
    # ref_model=None instructs DPOTrainer to compute reference log-probs from the base model in a no-grad pass,
    # avoiding a second full model load and saving ~12 GB VRAM with no measurable quality loss for QLoRA DPO
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_cfg,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        peft_config=lora_cfg,       # LoRA is injected here by DPOTrainer so it can manage the policy/reference split
    )

    trainer.model.print_trainable_parameters()
    logger.info("Starting IPO/DPO training …")
    train_result = trainer.train()
    logger.info("Training complete — %s", train_result.metrics)
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    return trainer


def evaluate(trainer: DPOTrainer) -> dict:
    # Key metrics: eval_loss, rewards/chosen, rewards/rejected, rewards/accuracies, rewards/margins
    logger.info("Evaluating on validation set …")
    metrics = trainer.evaluate()
    logger.info("Eval metrics: %s", metrics)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    return metrics


def save_adapter(trainer: DPOTrainer, tokenizer: AutoTokenizer, output_dir: str) -> str:
    # Only the LoRA delta weights are saved (not the full 24B base), keeping the checkpoint to ~100–400 MB
    adapter_path = os.path.join(output_dir, "final_adapter")
    trainer.model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    logger.info("Adapter saved to '%s'.", adapter_path)
    return adapter_path


def push_to_hub(trainer: DPOTrainer, tokenizer: AutoTokenizer, config: Config) -> None:
    logger.info("Pushing adapter to Hub — %s", config.hf_repo_id)
    trainer.model.push_to_hub(config.hf_repo_id, use_auth_token=True, private=config.push_private)
    tokenizer.push_to_hub(config.hf_repo_id, use_auth_token=True, private=config.push_private)
    logger.info("Upload complete — https://huggingface.co/%s", config.hf_repo_id)


def run_inference_smoke_test(
    base_model_id: str,
    adapter_path: str,
    bnb_config: BitsAndBytesConfig,
    max_new_tokens: int = 256,
) -> str:
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
        "Summarise the key obligations of the lessee under a standard commercial lease, "
        "including rent payment, maintenance duties, and obligations upon lease termination."
    )

    inputs = tok(sample_prompt, return_tensors="pt").to(peft_model.device)
    with torch.no_grad():
        output_ids = peft_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,        # greedy decoding for a deterministic, reproducible smoke test
            temperature=1.0,
        )

    # Slice off the prompt tokens so only the generated completion is decoded
    generated = tok.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    logger.info("Smoke test complete.")
    return generated


def main():
    authenticate_hub()

    dataset = build_dpo_dataset(cfg.inputs_dir, cfg.preferred_dir, cfg.rejected_dir)
    train_dataset, val_dataset = split_dataset(dataset, cfg.val_split, cfg.seed)

    tokenizer = load_tokenizer(cfg.base_model_id)

    bnb_config = build_bnb_config()
    model = load_model(cfg.base_model_id, bnb_config)

    lora_cfg = build_lora_config(cfg)
    dpo_cfg = build_dpo_config(cfg)

    trainer = run_dpo_training(
        model=model,
        tokenizer=tokenizer,
        lora_cfg=lora_cfg,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        dpo_cfg=dpo_cfg,
    )

    evaluate(trainer)
    adapter_path = save_adapter(trainer, tokenizer, cfg.output_dir)
    push_to_hub(trainer, tokenizer, cfg)

    logger.info("Running inference smoke test …")
    response = run_inference_smoke_test(cfg.base_model_id, adapter_path, bnb_config)
    print("\n=== Smoke Test Response ===")
    print(response)
    print("===========================\n")
    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()