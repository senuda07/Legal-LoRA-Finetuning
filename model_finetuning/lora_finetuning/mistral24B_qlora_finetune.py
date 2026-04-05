import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
)
from trl import SFTTrainer, SFTConfig
from huggingface_hub import login


# Setup logging to track progress and debug issues
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class Config:
    # Base pretrained model used for fine-tuning
    base_model_id: str = "mistralai/Mistral-Small-24B-Instruct-2501"

    # Dataset directories
    inputs_dir: str = "inputs"
    responses_dir: str = "preferred_responses"

    # HuggingFace repo for saving trained adapter
    hf_repo_id: str = "senuda07/legal-mistral-qlora"
    push_private: bool = False

    # Maximum sequence length 
    max_seq_length: int = 2048

    # LoRA parameters 
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # Target layers for LoRA injection 
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # Training hyperparameters
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8  # simulate larger batch size
    num_train_epochs: int = 3
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03  # stabilizes early training

    # Validation split ratio
    val_split: float = 0.2

    # Output directory for checkpoints
    output_dir: str = "./checkpoints"

    # Seed for reproducibility
    seed: int = 42


cfg = Config()


def authenticate_hub(token: Optional[str] = None):
    # Login to HuggingFace Hub 
    login(token=token)
    logger.info("Authenticated with HuggingFace Hub.")


def load_paired_files(inputs_dir: str, responses_dir: str) -> list[dict]:
    # Load input-response file pairs based on matching filenames
    input_paths = sorted(Path(inputs_dir).glob("*.txt"))

    if not input_paths:
        raise FileNotFoundError(f"No input files found in {inputs_dir}")

    pairs = []

    for inp in input_paths:
        resp = Path(responses_dir) / inp.name

        # Skip if matching response file is missing
        if not resp.exists():
            continue

        pairs.append({
            "input_text": inp.read_text().strip(),
            "response_text": resp.read_text().strip(),
        })

    if not pairs:
        raise RuntimeError("No matching input-response pairs found.")

    logger.info("Loaded %d training pairs", len(pairs))
    return pairs


def format_prompt(pair: dict) -> str:
    # Format data into Mistral instruction style
    system_prompt = (
        "You are an expert legal assistant. Provide accurate and concise responses."
    )

    return (
        f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"
        f"{pair['input_text']} [/INST] {pair['response_text']} </s>"
    )


def build_dataset(inputs_dir: str, responses_dir: str) -> Dataset:
    # Convert raw text files into a HuggingFace dataset
    pairs = load_paired_files(inputs_dir, responses_dir)

    # Apply formatting to each pair
    formatted = [format_prompt(p) for p in pairs]

    return Dataset.from_dict({"text": formatted})


def split_dataset(dataset: Dataset, val_split: float, seed: int):
    # Split dataset into training and validation sets
    split = dataset.train_test_split(test_size=val_split, seed=seed)
    return split["train"], split["test"]


def load_tokenizer(model_id: str) -> AutoTokenizer:
    # Load tokenizer and fix padding 
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    tokenizer.pad_token = tokenizer.eos_token  # reuse EOS as padding
    tokenizer.padding_side = "right"  # required for causal models

    return tokenizer


def build_bnb_config() -> BitsAndBytesConfig:
    # 4-bit quantization reduces memory usage significantly
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4", 
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def load_model(model_id: str, bnb_config: BitsAndBytesConfig):
    # Load model with quantization and device auto-mapping
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    model.config.use_cache = False  # disable cache to save memory

    # Enable gradient checkpointing
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    return model


def apply_lora(model, config: Config):
    # Inject LoRA adapters instead of updating full model weights
    lora_cfg = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_cfg)

    # Print trainable vs frozen parameters
    model.print_trainable_parameters()

    return model


def build_training_args(config: Config) -> SFTConfig:
    # Define training settings optimized for QLoRA
    return SFTConfig(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_steps=50,
        save_total_limit=3,
        bf16=True,  
        optim="paged_adamw_8bit",  
        dataset_text_field="text",
        max_length=config.max_seq_length,
        seed=config.seed,
    )


def run_training(model, tokenizer, train_ds, val_ds, training_args):
    # Initialize trainer for supervised fine-tuning
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=training_args,
    )

    logger.info("Starting training...")
    trainer.train()

    return trainer


def evaluate(trainer):
    # Evaluate model on validation set
    metrics = trainer.evaluate()
    logger.info("Evaluation results: %s", metrics)
    return metrics


def save_adapter(trainer, tokenizer, output_dir: str):
    # Save only LoRA adapter 
    path = os.path.join(output_dir, "final_adapter")

    trainer.model.save_pretrained(path)
    tokenizer.save_pretrained(path)

    return path


def push_to_hub(trainer, tokenizer, config: Config):
    # Upload adapter and tokenizer to HuggingFace Hub
    trainer.model.push_to_hub(config.hf_repo_id, private=config.push_private)
    tokenizer.push_to_hub(config.hf_repo_id, private=config.push_private)


def run_inference(base_model_id, adapter_path, bnb_config):
    # Load tokenizer and base model for inference
    tok = AutoTokenizer.from_pretrained(adapter_path)

    base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    # Attach trained LoRA adapter
    model = PeftModel.from_pretrained(base, adapter_path)

    # Simple test prompt
    prompt = "<s>[INST] Summarise a lease agreement. [/INST]"
    inputs = tok(prompt, return_tensors="pt").to(model.device)

    # Generate output
    output = model.generate(**inputs, max_new_tokens=200)

    return tok.decode(output[0], skip_special_tokens=True)


def main():
    # Authenticate before pushing to HuggingFace
    authenticate_hub()

    # Prepare dataset
    dataset = build_dataset(cfg.inputs_dir, cfg.responses_dir)

    # Split dataset
    train_ds, val_ds = split_dataset(dataset, cfg.val_split, cfg.seed)

    # Load tokenizer and model
    tokenizer = load_tokenizer(cfg.base_model_id)
    bnb_config = build_bnb_config()
    model = load_model(cfg.base_model_id, bnb_config)

    # Apply LoRA
    model = apply_lora(model, cfg)

    # Setup training configuration
    training_args = build_training_args(cfg)

    # Train model
    trainer = run_training(model, tokenizer, train_ds, val_ds, training_args)

    # Evaluate model
    evaluate(trainer)

    # Save adapter locally
    adapter_path = save_adapter(trainer, tokenizer, cfg.output_dir)

    # Push to HuggingFace Hub
    push_to_hub(trainer, tokenizer, cfg)

    # Run quick inference test
    print(run_inference(cfg.base_model_id, adapter_path, bnb_config))


if __name__ == "__main__":
    main()