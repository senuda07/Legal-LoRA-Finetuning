"""
mistral24B_qlora_dpo.py
=======================
QLoRA-based DPO (Identity Preference Optimisation / IPO) fine-tuning pipeline
for ``mistralai/Mistral-Small-3.1-24B-Instruct-2503`` on a paired legal dataset.

Why IPO over standard DPO?
--------------------------
Standard DPO can over-optimise the implicit reward, especially on small or
domain-specific datasets where the margin between preferred and rejected
responses is subtle — which is common in legal text.  IPO (Azar et al., 2023)
regularises the optimisation directly on the preference probability rather
than on the log-ratio, yielding more stable training and better calibration
on legal summarisation tasks.

Pipeline overview
-----------------
1.  Authenticate with HuggingFace Hub.
2.  Load three folders of .txt files: inputs, preferred responses, rejected
    responses.  Files are matched by filename stem.
3.  Format each triple into Mistral chat-style prompt/chosen/rejected dicts.
4.  Build a HuggingFace Dataset and split 80 / 20 into train / validation.
5.  Load the tokenizer (pad = EOS, right-padded).
6.  Load the base model in 4-bit NF4 QLoRA quantisation.
7.  Inject LoRA adapters across all attention + FFN projections.
8.  Configure and run DPOTrainer with IPO loss.
9.  Evaluate on the validation set and log metrics.
10. Save the LoRA adapter weights locally.
11. Push adapter + tokenizer to HuggingFace Hub.
12. Run an inference smoke test with the saved adapter.

VRAM budget (80 GB)
-------------------
* 4-bit NF4 double-quant          →  ~12–14 GB for the 24B base model
* BF16 compute dtype              →  stable on A100 / H100
* paged_adamw_8bit optimiser      →  minimises optimiser-state spikes
* gradient_checkpointing=True     →  ~40 % VRAM saving on activations
* per_device_train_batch_size=2   →  tune upward if headroom allows
* Reference model sharing         →  DPOTrainer reuses the same 4-bit base
                                     as the reference; avoids loading a second
                                     full copy (saves ~12 GB vs default).

Dependencies
------------
    pip install transformers peft trl bitsandbytes datasets accelerate \
                huggingface_hub
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1.  Central configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """
    Master configuration for the QLoRA IPO/DPO pipeline.

    All hyper-parameters and file paths live here so that nothing is
    hard-coded elsewhere in the script.

    Attributes
    ----------
    base_model_id : str
        HuggingFace model repo used as the frozen base.
    inputs_dir : str
        Folder containing input text chunk files (*.txt).
    preferred_dir : str
        Folder containing preferred (golden) response files (*.txt).
        Each file must share the exact filename with its input counterpart.
    rejected_dir : str
        Folder containing rejected (base-model) response files (*.txt).
        Each file must share the exact filename with its input counterpart.
    hf_repo_id : str
        HuggingFace Hub destination for the fine-tuned adapter.
    max_prompt_length : int
        Maximum token length for the prompt portion of each sample.
        Sequences are truncated to this value.
    max_length : int
        Maximum total token length (prompt + response).  Keep ≤ 4096 on
        memory-constrained setups; Mistral-Small supports 128k context.
    lora_r : int
        LoRA rank.  16 is a good default; raise to 32 for more capacity.
    lora_alpha : int
        LoRA scaling factor.  Convention: 2 × lora_r.
    lora_dropout : float
        Dropout applied to LoRA layers for regularisation.
    target_modules : List[str]
        Linear sub-layers that receive LoRA adapters.
    per_device_train_batch_size : int
        Batch size per GPU.  Reduce to 1 if OOM occurs.
    gradient_accumulation_steps : int
        Effective batch = per_device_train_batch_size ×
        gradient_accumulation_steps.
    num_train_epochs : int
        Full passes over the training dataset.
    learning_rate : float
        Peak learning rate for the paged AdamW optimiser.
    warmup_ratio : float
        Fraction of total steps used for linear LR warm-up.
    beta : float
        IPO / DPO temperature coefficient.  Lower values → more conservative
        updates.  IPO is less sensitive to beta than standard DPO; 0.1 is a
        robust default for legal tasks.
    loss_type : str
        DPOTrainer loss type.  ``"ipo"`` selects Identity Preference
        Optimisation.  Other options: ``"sigmoid"`` (standard DPO),
        ``"hinge"``, ``"kto_pair"``.
    val_split : float
        Fraction of the dataset held out for validation (0.2 = 20 %).
    output_dir : str
        Local directory for training checkpoints.
    push_private : bool
        If True the HuggingFace Hub repo is created as private.
    seed : int
        Global random seed for reproducibility.
    """

    # ── Model ──────────────────────────────────────────────────────────────
    base_model_id: str = "mistralai/Mistral-Small-24B-Instruct-2501"
    adapter_name: str = "senuda07/legal-mistral-qlora"

    # ── Data ───────────────────────────────────────────────────────────────
    inputs_dir: str = "chunked_train_data"
    preferred_dir: str = "openai_summaries_train_chunked"
    rejected_dir: str = "mistral24B_summaries_train_chunked"

    # ── Hub ────────────────────────────────────────────────────────────────
    hf_repo_id: str = "senuda07/legal-mistral-sft-dpo"

    # ── Sequence lengths ───────────────────────────────────────────────────
    # Keep max_prompt_length well below max_length to leave room for
    # both chosen and rejected completions inside the DPO context window.
    max_prompt_length: int = 1024
    max_length: int = 2048

    # ── LoRA ───────────────────────────────────────────────────────────────
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    # Target all attention + SwiGLU-FFN projections for best task coverage.
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # ── Training ───────────────────────────────────────────────────────────
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8   # → effective batch size = 16
    num_train_epochs: int = 3
    learning_rate: float = 5e-5            # IPO prefers a lower LR than SFT
    warmup_ratio: float = 0.05

    # ── IPO / DPO hyper-params ─────────────────────────────────────────────
    # IPO is selected via loss_type="ipo".  Beta controls the regularisation
    # strength; 0.1 is conservative and stable for legal summarisation.
    beta: float = 0.1
    loss_type: str = "ipo"

    # ── Validation split ───────────────────────────────────────────────────
    val_split: float = 0.2

    # ── Misc ───────────────────────────────────────────────────────────────
    output_dir: str = "./dpo_checkpoints"
    push_private: bool = False
    seed: int = 42


# Instantiate the global config used throughout the script.
cfg = Config()


# ---------------------------------------------------------------------------
# 2.  HuggingFace Hub authentication
# ---------------------------------------------------------------------------

def authenticate_hub(token: Optional[str] = None) -> None:
    """
    Log into HuggingFace Hub using an environment variable or explicit token.

    The WRITE permission is required to push adapters to the Hub.
    Store your token in the ``HF_TOKEN`` environment variable rather than
    hard-coding it here.

    Parameters
    ----------
    token : str, optional
        Explicit HF token.  If None, the ``HF_TOKEN`` environment variable
        is used.  Falls back to an interactive prompt when both are absent.
    """
    resolved_token = token or os.environ.get("HF_TOKEN")
    login(token=resolved_token)
    logger.info("Authenticated with HuggingFace Hub.")


# ---------------------------------------------------------------------------
# 3.  Dataset loading — triplet file pairs
# ---------------------------------------------------------------------------

def load_triplet_files(
    inputs_dir: str,
    preferred_dir: str,
    rejected_dir: str,
) -> list[dict]:
    """
    Load all (input, preferred, rejected) file triplets from disk.

    Files are matched by filename stem: ``inputs/doc01.txt`` is paired with
    ``preferred_responses/doc01.txt`` and ``rejected_responses/doc01.txt``.
    Triplets with any missing file are skipped with a warning so that a few
    absent files do not abort the entire run.

    Parameters
    ----------
    inputs_dir : str
        Directory containing input prompt text files (*.txt).
    preferred_dir : str
        Directory containing preferred (golden) response text files (*.txt).
    rejected_dir : str
        Directory containing rejected (base-model) response text files (*.txt).

    Returns
    -------
    list[dict]
        List of dicts, each with keys ``input_text``, ``preferred_text``,
        and ``rejected_text``.

    Raises
    ------
    FileNotFoundError
        Raised when no *.txt files are found in ``inputs_dir``.
    RuntimeError
        Raised when no valid triplets are found after matching.
    """
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


# ---------------------------------------------------------------------------
# 4.  Prompt formatting — Mistral chat template
# ---------------------------------------------------------------------------

def build_mistral_prompt(input_text: str, system_prompt: Optional[str] = None) -> str:
    """
    Format a prompt string using Mistral's [INST] chat template.

    For DPO the prompt and response must be kept separate — the DPOTrainer
    receives three distinct fields (``prompt``, ``chosen``, ``rejected``) and
    handles concatenation internally.

    The prompt here contains only the system message + user turn (no
    assistant turn), so that DPOTrainer can append chosen / rejected
    completions correctly.

    Format produced
    ---------------
    ::

        <s>[INST] <<SYS>>
        {system_prompt}
        <</SYS>>

        {input_text} [/INST]

    Parameters
    ----------
    input_text : str
        The user's input / legal document chunk.
    system_prompt : str, optional
        Custom system instruction.  Defaults to a domain-specific legal
        assistant prompt.

    Returns
    -------
    str
        A formatted Mistral-style prompt string (no assistant turn).
    """
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
    """
    Build a HuggingFace Dataset formatted for DPOTrainer.

    DPOTrainer expects a dataset with three columns:
    * ``prompt``   — the formatted user-turn prompt (no assistant response).
    * ``chosen``   — the preferred (golden) completion text.
    * ``rejected`` — the rejected (base-model) completion text.

    Parameters
    ----------
    inputs_dir : str
        Path to the folder containing input chunk text files.
    preferred_dir : str
        Path to the folder containing preferred response text files.
    rejected_dir : str
        Path to the folder containing rejected response text files.

    Returns
    -------
    datasets.Dataset
        Three-column Dataset ready for DPOTrainer.
    """
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


# ---------------------------------------------------------------------------
# 5.  Train / validation split
# ---------------------------------------------------------------------------

def split_dataset(dataset: Dataset, val_split: float, seed: int):
    """
    Split the dataset into train and validation subsets.

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


# ---------------------------------------------------------------------------
# 6.  Tokenizer
# ---------------------------------------------------------------------------

def load_tokenizer(model_id: str) -> AutoTokenizer:
    """
    Load and configure the tokenizer for the given model.

    Mistral models have no dedicated pad token; EOS is reused to satisfy
    the data collator.  Right-padding is safer than left-padding for causal
    attention masks during DPO training.

    Parameters
    ----------
    model_id : str
        HuggingFace model identifier.

    Returns
    -------
    AutoTokenizer
        Configured tokenizer instance.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        use_fast=False,
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


# ---------------------------------------------------------------------------
# 7.  BitsAndBytes 4-bit config
# ---------------------------------------------------------------------------

def build_bnb_config() -> BitsAndBytesConfig:
    """
    Build the BitsAndBytes 4-bit NF4 quantisation configuration.

    Design choices for VRAM efficiency
    ------------------------------------
    * ``nf4``          — Normal Float 4-bit; optimal for normally-distributed
                         neural network weights (better quality than INT4).
    * ``double_quant`` — Quantises the quantisation constants themselves,
                         saving ~0.4 GB on a 24B model.
    * ``bfloat16``     — Dequantises to BF16 for the forward/backward pass;
                         numerically stable on Ampere/Hopper GPUs.

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


# ---------------------------------------------------------------------------
# 8.  Model loading (4-bit NF4 + BF16 compute)
# ---------------------------------------------------------------------------

def load_model(model_id: str, bnb_config: BitsAndBytesConfig, adapter_name: str = None) -> AutoModelForCausalLM:
    """
    Load the base model in 4-bit quantised form and prepare it for QLoRA.

    VRAM-saving knobs applied
    -------------------------
    * ``device_map="auto"``        — Spreads layers across all available GPUs
                                     (and CPU offload if required).
    * ``use_cache=False``          — Disables the KV cache during training.
    * ``gradient_checkpointing``   — Re-computes activations on the backward
                                     pass; saves ~40 % VRAM at ~20 % throughput
                                     cost.
    * ``attn_implementation``      — Uncomment flash_attention_2 if FA-2 is
                                     installed; halves attention-layer VRAM.

    Note on the DPO reference model
    --------------------------------
    DPOTrainer by default creates a frozen copy of the policy model to use as
    the reference model.  When ``ref_model=None`` is passed to DPOTrainer, it
    instead computes reference log-probs from the same model in a no-grad pass,
    which avoids loading a second 24B copy and saves ~12 GB VRAM.

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
    )
    
    if adapter_name:
        logger.info("Loading SFT adapter '%s' and merging into base model …", adapter_name)
        model = PeftModel.from_pretrained(model, adapter_name)
        model = model.merge_and_unload()   # returns a plain AutoModelForCausalLM
        logger.info("SFT adapter merged and unloaded.")
    
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True
    )

    vram_gb = model.get_memory_footprint() / (1024 ** 3)
    logger.info("Model loaded — VRAM footprint: %.2f GB", vram_gb)
    return model


# ---------------------------------------------------------------------------
# 9.  LoRA adapter configuration
# ---------------------------------------------------------------------------

def build_lora_config(config: Config) -> LoraConfig:
    """
    Build the LoRA adapter configuration for QLoRA DPO.

    Targeting all seven linear projections (attention + SwiGLU FFN) is
    recommended for instruction-following / preference-learning tasks on
    Mistral-family models; it consistently outperforms attention-only
    targeting.

    Parameters
    ----------
    config : Config
        Pipeline configuration (reads lora_r, lora_alpha, lora_dropout,
        target_modules).

    Returns
    -------
    LoraConfig
        PEFT LoRA configuration object ready to pass to DPOTrainer.
    """
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


# ---------------------------------------------------------------------------
# 10.  DPO training arguments
# ---------------------------------------------------------------------------

def build_dpo_config(config: Config) -> DPOConfig:
    """
    Build the DPOConfig (training arguments) for the DPOTrainer.

    Key choices
    -----------
    * ``loss_type="ipo"``          — Identity Preference Optimisation loss.
    * ``beta``                     — IPO regularisation coefficient (0.1).
    * ``paged_adamw_8bit``         — Paged optimiser states prevent OOM spikes.
    * ``gradient_checkpointing``   — Saves ~40 % VRAM on activation storage.
    * ``bf16=True``                — Mixed-precision in BF16 (A100/H100 safe).
    * ``group_by_length=True``     — Bins similar-length samples to reduce
                                     padding waste and improve throughput.
    * ``generate_during_eval``     — Disabled (True adds latency and VRAM).

    Parameters
    ----------
    config : Config
        Pipeline configuration object.

    Returns
    -------
    DPOConfig
    """
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

        # — Checkpointing —
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


# ---------------------------------------------------------------------------
# 11.  DPO training loop
# ---------------------------------------------------------------------------

def run_dpo_training(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    lora_cfg: LoraConfig,
    train_dataset: Dataset,
    val_dataset: Dataset,
    dpo_cfg: DPOConfig,
) -> DPOTrainer:
    """
    Initialise and run the DPOTrainer with IPO loss.

    Reference model strategy
    ------------------------
    Passing ``ref_model=None`` instructs DPOTrainer to compute reference
    log-probabilities using the **same** model in a frozen (no_grad) pass
    rather than instantiating a separate copy.  This saves ~12 GB VRAM on
    a 24B model with no measurable quality loss for QLoRA DPO.

    LoRA is applied inside this function (via ``peft_config``) so that
    DPOTrainer can correctly handle the policy / reference split.

    Parameters
    ----------
    model : AutoModelForCausalLM
        Base model loaded and prepared for k-bit training (no LoRA yet).
    tokenizer : AutoTokenizer
        Configured tokenizer.
    lora_cfg : LoraConfig
        LoRA adapter configuration.
    train_dataset : Dataset
        Training split with columns: prompt, chosen, rejected.
    val_dataset : Dataset
        Validation split with columns: prompt, chosen, rejected.
    dpo_cfg : DPOConfig
        Training arguments and DPO hyper-parameters.

    Returns
    -------
    DPOTrainer
        Trained DPOTrainer instance.
    """
    trainer = DPOTrainer(
        model=model,
        ref_model=None,         # share base as reference (saves ~12 GB VRAM)
        args=dpo_cfg,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        peft_config=lora_cfg,   # DPOTrainer applies LoRA internally
    )

    # Log trainable parameter count after LoRA injection.
    trainer.model.print_trainable_parameters()

    logger.info("Starting IPO/DPO training …")
    train_result = trainer.train()
    logger.info("Training complete.  Metrics: %s", train_result.metrics)
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)

    return trainer


# ---------------------------------------------------------------------------
# 12.  Validation evaluation
# ---------------------------------------------------------------------------

def evaluate(trainer: DPOTrainer) -> dict:
    """
    Run evaluation on the validation set and log metrics.

    Metrics reported by DPOTrainer
    -------------------------------
    * ``eval_loss``           — IPO loss on the validation set.
    * ``eval_rewards/chosen`` — Mean reward margin for chosen responses.
    * ``eval_rewards/rejected``
    * ``eval_rewards/accuracies`` — Fraction of pairs where chosen > rejected.
    * ``eval_rewards/margins``    — Mean (chosen_reward − rejected_reward).

    Parameters
    ----------
    trainer : DPOTrainer
        Trained DPOTrainer instance.

    Returns
    -------
    dict
        Evaluation metric dictionary.
    """
    logger.info("Evaluating on validation set …")
    metrics = trainer.evaluate()
    logger.info("Eval metrics: %s", metrics)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    return metrics


# ---------------------------------------------------------------------------
# 13.  Save adapter locally
# ---------------------------------------------------------------------------

def save_adapter(
    trainer: DPOTrainer,
    tokenizer: AutoTokenizer,
    output_dir: str,
) -> str:
    """
    Save the LoRA adapter weights and tokenizer to a local directory.

    Only the adapter delta weights are persisted (not the full 24B base),
    keeping the checkpoint compact — typically 100–400 MB.

    Parameters
    ----------
    trainer : DPOTrainer
        Trained DPOTrainer instance.
    tokenizer : AutoTokenizer
        Tokenizer to save alongside the adapter.
    output_dir : str
        Parent checkpoint directory; adapter is saved to a sub-folder.

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


# ---------------------------------------------------------------------------
# 14.  Push to HuggingFace Hub
# ---------------------------------------------------------------------------

def push_to_hub(
    trainer: DPOTrainer,
    tokenizer: AutoTokenizer,
    config: Config,
) -> None:
    """
    Push the fine-tuned LoRA adapter and tokenizer to HuggingFace Hub.

    After upload, users can reload the full pipeline as follows:

    .. code-block:: python

        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        base = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-Small-3.1-24B-Instruct-2503", ...
        )
        model = PeftModel.from_pretrained(base, "senuda07/legal-mistral-dpo")
        tokenizer = AutoTokenizer.from_pretrained("senuda07/legal-mistral-dpo")

    Parameters
    ----------
    trainer : DPOTrainer
        Trained DPOTrainer instance.
    tokenizer : AutoTokenizer
        Tokenizer to upload.
    config : Config
        Pipeline configuration (reads hf_repo_id, push_private).
    """
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


# ---------------------------------------------------------------------------
# 15.  Inference smoke test
# ---------------------------------------------------------------------------

def run_inference_smoke_test(
    base_model_id: str,
    adapter_path: str,
    bnb_config: BitsAndBytesConfig,
    max_new_tokens: int = 256,
) -> str:
    """
    Load the saved adapter and run a sample inference to verify the output.

    This is a quick sanity check, not a benchmark.  Run after saving the
    adapter to confirm the model produces coherent, legally-flavoured text.

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

    generated = tok.decode(
        output_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    logger.info("Smoke test generation complete.")
    return generated


# ---------------------------------------------------------------------------
# 16.  Main entry point
# ---------------------------------------------------------------------------

def main():
    """
    Orchestrate the full QLoRA IPO/DPO pipeline end-to-end.

    Execution order
    ---------------
    1.  Authenticate with HuggingFace Hub (reads HF_TOKEN env var).
    2.  Load triplet .txt files and build the DPO dataset.
    3.  Split 80 / 20 into train / validation sets.
    4.  Load and configure the tokenizer.
    5.  Build the 4-bit NF4 BitsAndBytes config.
    6.  Load the base model in 4-bit quantisation.
    7.  Build the LoRA adapter configuration.
    8.  Build the DPOConfig (training arguments + IPO hyper-params).
    9.  Run the DPO training loop.
    10. Evaluate on the validation set.
    11. Save the adapter locally.
    12. Push adapter + tokenizer to HuggingFace Hub.
    13. Run an inference smoke test.
    """
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

    # Step 5 & 6 – Model (4-bit)
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