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
    TrainingArguments,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
)
from trl import SFTTrainer
from huggingface_hub import login

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    # Remove TrainingArguments — SFTConfig replaces it
)
from trl import SFTTrainer, SFTConfig  # Add SFTConfig
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1.  Central configuration

@dataclass
class Config:


    # Model 
    base_model_id: str = "mistralai/Mistral-Small-24B-Instruct-2501"

    # Data 
    inputs_dir: str = "inputs" # original dataset chunks
    responses_dir: str = "preferred_responses" # GPT-5-mini train outputs

    # Hub 
    hf_repo_id: str = "senuda07/legal-mistral-qlora"
    max_seq_length: int = 2048

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
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03

    # Validation split
    val_split: float = 0.2

    # Misc
    output_dir: str = "./checkpoints"
    push_private: bool = False
    seed: int = 42


cfg = Config()


# 2. HuggingFace Hub authentication

def authenticate_hub(token: Optional[str] = None) -> None:

    resolved_token = "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"
    login(token=resolved_token)  # prompts interactively when resolved_token is None
    logger.info("Authenticated with HuggingFace Hub.")


# 3. Dataset loading and prompt formatting

def load_paired_files(inputs_dir: str, responses_dir: str) -> list[dict]:

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
    """
    Format a single (input, response) pair using Mistral's chat format.

    Mistral Instruct models are trained with a specific [INST] ... [/INST]
    prompt structure.  We include an explicit system message prepended to the
    user turn, which is the recommended approach for Mistral-Small-3.x models.

    Format
    ------
    ::

        <s>[INST] <<SYS>>
        {system_prompt}
        <</SYS>>

        {input_text} [/INST] {response_text} </s>

    Parameters
    ----------
    pair : dict
        Dict with keys ``input_text`` and ``response_text``.
    system_prompt : str, optional
        Custom system instruction.  Defaults to a domain-specific legal
        assistant prompt.

    Returns
    -------
    str
        A fully formatted Mistral-style chat string ready for tokenisation.
    """
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
    """
    Build a HuggingFace Dataset from paired input / response files.

    Parameters
    ----------
    inputs_dir : str
        Path to the folder containing input chunk text files.
    responses_dir : str
        Path to the folder containing preferred response text files.

    Returns
    -------
    datasets.Dataset
        A single-column Dataset (``text``) with all formatted prompts.
    """
    pairs = load_paired_files(inputs_dir, responses_dir)
    formatted = [build_mistral_chat_prompt(p) for p in pairs]
    dataset = Dataset.from_dict({"text": formatted})
    logger.info("Dataset created — %d examples total.", len(dataset))
    return dataset


# 4.  Train / validation split

def split_dataset(dataset: Dataset, val_split: float, seed: int):
    """
    Perform a stratified 80/20 (or custom) train / validation split.

    Parameters
    ----------
    dataset : Dataset
        Full HuggingFace Dataset to split.
    val_split : float
        Fraction reserved for validation, e.g. 0.2 for 20 %.
    seed : int
        Random seed for reproducible shuffling.

    Returns
    -------
    tuple[Dataset, Dataset]
        ``(train_dataset, val_dataset)``
    """
    split = dataset.train_test_split(test_size=val_split, seed=seed, shuffle=True)
    train_ds, val_ds = split["train"], split["test"]
    logger.info(
        "Split — Train: %d samples  |  Validation: %d samples",
        len(train_ds), len(val_ds),
    )
    return train_ds, val_ds


# 5.  Tokenizer

def load_tokenizer(model_id: str) -> AutoTokenizer:
    """
    Load and configure the tokenizer for the given model.

    Mistral uses a SentencePiece / Tekken tokenizer (LlamaTokenizerFast
    under the hood for 3.x models).  The pad token is set to eos_token
    because Mistral has no dedicated pad token by default.

    Parameters
    ----------
    model_id : str
        HuggingFace model identifier, e.g. "mistralai/Mistral-Small-3.1-24B-Instruct-2503".

    Returns
    -------
    AutoTokenizer
        Configured tokenizer instance.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        use_fast=True,
    )

    # Mistral tokenizers have no pad token; reuse EOS to satisfy the data collator.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Right-padding is safer than left-padding for causal attention masks.
    tokenizer.padding_side = "right"

    logger.info(
        "Tokenizer loaded — vocab size: %d  |  pad token: '%s'",
        tokenizer.vocab_size,
        tokenizer.pad_token,
    )
    return tokenizer


# 6.  Model loading (4-bit NF4 + BF16 compute)

def build_bnb_config() -> BitsAndBytesConfig:
    """
    Build the BitsAndBytes 4-bit quantisation configuration.

    Choices made for VRAM efficiency
    ---------------------------------
    * ``nf4``           — Normal Float 4-bit, optimal for normally-distributed
                          neural network weights (better quality than INT4).
    * ``double_quant``  — Quantises the quantisation constants themselves,
                          saving ~0.4 GB on a 24B model.
    * ``bfloat16``      — Dequantises to BF16 for forward/backward passes;
                          numerically stable on Ampere/Hopper GPUs (A100, H100).

    Returns
    -------
    BitsAndBytesConfig
    """
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def load_model(model_id: str, bnb_config: BitsAndBytesConfig) -> AutoModelForCausalLM:
    """
    Load the base model in 4-bit quantised form.

    Additional VRAM knobs
    ---------------------
    * ``device_map="auto"``     — Spreads layers across all available GPUs
                                  (and CPU if needed); ideal for multi-GPU nodes.
    * ``use_cache=False``       — Disables the KV cache during training (unused
                                  and wastes memory).
    * ``gradient_checkpointing``— Re-computes activations during the backward
                                  pass instead of storing them; saves ~40 % VRAM
                                  at ~20 % throughput cost.
    * ``flash_attention_2``     — Fused attention kernel; cuts attention-layer
                                  VRAM roughly in half. Remove the
                                  ``attn_implementation`` kwarg if FlashAttention-2
                                  is not installed in your environment.

    Parameters
    ----------
    model_id : str
        HuggingFace model identifier.
    bnb_config : BitsAndBytesConfig
        4-bit quantisation settings.

    Returns
    -------
    AutoModelForCausalLM
        Quantised model prepared for k-bit (QLoRA) training.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        # attn_implementation="flash_attention_2",  # remove if FA-2 not installed
    )

    # KV-cache is not needed during training and wastes VRAM.
    model.config.use_cache = False
    # Ensure the model is not sharded internally (safe default for Mistral).
    model.config.pretraining_tp = 1

    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True
    )

    vram_gb = model.get_memory_footprint() / (1024 ** 3)
    logger.info("Model loaded — VRAM footprint: %.2f GB", vram_gb)
    return model


# 7.  LoRA adapter injection

def apply_lora(model: AutoModelForCausalLM, config: Config) -> AutoModelForCausalLM:
    """
    Inject LoRA adapters into the quantised model (QLoRA).

    Targeting all seven linear projections (attention + SwiGLU FFN) is the
    recommended strategy for instruction-following tasks on Mistral-family
    models, as it outperforms attention-only targeting.

    Parameters
    ----------
    model : AutoModelForCausalLM
        Base model already prepared for k-bit training.
    config : Config
        Pipeline configuration (reads lora_r, lora_alpha, lora_dropout,
        target_modules).

    Returns
    -------
    AutoModelForCausalLM
        Model with LoRA adapters attached and all base weights frozen.
    """
    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model, lora_cfg


# 8.  Training arguments

def build_training_args(config: Config) -> SFTConfig:
    """
    Build HuggingFace TrainingArguments optimised for QLoRA SFT.

    Key VRAM-saving choices
    -----------------------
    * ``paged_adamw_8bit``       — Paged memory for optimiser states avoids
                                   OOM spikes caused by large Adam momentum
                                   tensors on 24B+ models.
    * ``gradient_checkpointing`` — Re-computes activations on the backward pass.
    * ``bf16=True``              — Mixed-precision in BF16 (stable on A100/H100).
    * ``group_by_length=True``   — Batches similar-length sequences, reducing
                                   padding waste and improving throughput.

    Parameters
    ----------
    config : Config
        Pipeline configuration object.

    Returns
    -------
    TrainingArguments
    """
    return SFTConfig(
        # — Training args —
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        learning_rate=config.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=config.warmup_ratio,
        bf16=True,
        fp16=False,
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
        # — SFT-specific args (moved here from SFTTrainer) —
        dataset_text_field="text",
        max_length=config.max_seq_length,
        packing=False,
    )


# 9.  SFT training loop

def run_training(
    model,
    tokenizer: AutoTokenizer,
    lora_cfg: LoraConfig,
    train_dataset: Dataset,
    val_dataset: Dataset,
    training_args: SFTConfig,
    config: Config,
) -> SFTTrainer:
    # Apply LoRA here — exactly once
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        # peft_config removed — LoRA already applied above
        args=training_args,
    )

    logger.info("Starting SFT training …")
    train_result = trainer.train()
    logger.info("Training complete. Metrics: %s", train_result.metrics)
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    return trainer


# 10.  Evaluation

def evaluate(trainer: SFTTrainer) -> dict:
    """
    Evaluate the trained model on the validation set.

    Parameters
    ----------
    trainer : SFTTrainer
        Trainer instance after training.

    Returns
    -------
    dict
        Evaluation metrics (includes ``eval_loss`` and ``eval_perplexity``
        when available).
    """
    logger.info("Evaluating on validation set …")
    metrics = trainer.evaluate()
    logger.info("Eval metrics: %s", metrics)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    return metrics


# 11.  Save adapter locally

def save_adapter(trainer: SFTTrainer, tokenizer: AutoTokenizer, output_dir: str) -> str:
    """
    Save the LoRA adapter weights and tokenizer to a local directory.

    Only the adapter delta weights are saved (not the full 24B base model),
    keeping the checkpoint compact — typically 100–400 MB.

    Parameters
    ----------
    trainer : SFTTrainer
        Trained SFTTrainer instance.
    tokenizer : AutoTokenizer
        Tokenizer to save alongside the adapter.
    output_dir : str
        Parent checkpoint directory (the adapter is saved to a sub-folder).

    Returns
    -------
    str
        Absolute path to the saved adapter directory.
    """
    adapter_path = os.path.join(output_dir, "final_adapter")
    trainer.model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    logger.info("Adapter saved locally to '%s'.", adapter_path)
    return adapter_path


# 12.  Push to HuggingFace Hub

def push_to_hub(trainer: SFTTrainer, tokenizer: AutoTokenizer, config: Config) -> None:
    """
    Push the fine-tuned LoRA adapter and tokenizer to HuggingFace Hub.

    After this step users can load the full pipeline with:

    .. code-block:: python

        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        base = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-Small-3.1-24B-Instruct-2503", ...)
        model = PeftModel.from_pretrained(base, "<hf_repo_id>")
        tokenizer = AutoTokenizer.from_pretrained("<hf_repo_id>")

    Parameters
    ----------
    trainer : SFTTrainer
        Trained SFTTrainer instance.
    tokenizer : AutoTokenizer
        Tokenizer to upload.
    config : Config
        Pipeline configuration (reads hf_repo_id and push_private).
    """
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


# 13.  Inference smoke test

def run_inference_smoke_test(
    base_model_id: str,
    adapter_path: str,
    bnb_config: BitsAndBytesConfig,
    max_new_tokens: int = 256,
) -> str:
    """
    Load the saved adapter and run a sample inference to verify the output.

    This is a quick sanity check — not a benchmark.  Run after saving the
    adapter to confirm the model produces coherent legal text.

    Parameters
    ----------
    base_model_id : str
        HuggingFace base model identifier (must match training).
    adapter_path : str
        Local path to the saved LoRA adapter directory.
    bnb_config : BitsAndBytesConfig
        Quantisation config (reuse the same object from training).
    max_new_tokens : int
        Maximum tokens to generate (default 256).

    Returns
    -------
    str
        Generated text string (prompt excluded).
    """
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
            do_sample=False, 
            temperature=1.0,
        )

    generated = tok.decode(
        output_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return generated


# 14.  Main

def main():

    # Step 1 – Authentication
    authenticate_hub()

    # Step 2 – Dataset
    dataset = build_dataset(cfg.inputs_dir, cfg.responses_dir)

    # Step 3 – Train / val split
    train_dataset, val_dataset = split_dataset(dataset, cfg.val_split, cfg.seed)

    # Step 4 – Tokenizer
    tokenizer = load_tokenizer(cfg.base_model_id)

    # Step 5 – Model (4-bit)
    bnb_config = build_bnb_config()
    model = load_model(cfg.base_model_id, bnb_config)

    # Step 6 – LoRA adapters
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

    # Step 9 – Evaluate
    evaluate(trainer)

    # Step 10 – Save locally
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
