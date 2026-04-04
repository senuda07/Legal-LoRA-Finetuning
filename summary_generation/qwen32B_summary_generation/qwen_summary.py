import os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Configuration

model = "Qwen/Qwen2.5-32B-Instruct"

input = "chunked_test_data"
output = "qwen32B_summaries_test_chunked"

max_new_tokens = 300
batch_size = 4   

os.makedirs(output, exist_ok=True)

# BNB Quantization Configuration for 4-bit quantization with NF4 and double quantization

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# Load Model and Tokenizer

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    model,
    trust_remote_code=True
)
tokenizer.padding_side = "left"

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading model with 4-bit BNB quantization...")

model = AutoModelForCausalLM.from_pretrained(
    model,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
)

model.eval()

# Prompt Template

def build_prompt(text):

    prompt = f"""
You are a legal research assistant specialized in summarizing legislative documents.
The included text is part of a larger summary.

Create accurate, concise, and short summaries in plain English.
Preserve key legal clauses, definitions, obligations, and relationships.
Do not introduce new information.
Avoid unnecessary simplification.
Maintain legal terminology where important.

Text:
{text}

Summary:
"""

    return prompt.strip()


# Batch Summarization Function

def summarize_batch(texts):

    prompts = [build_prompt(t) for t in texts]

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.2,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.batch_decode(
        outputs,
        skip_special_tokens=True
    )

    summaries = []

    for text in decoded:
        summary = text.split("Summary:")[-1].strip()
        summaries.append(summary)

    return summaries


# Load Files 

files = sorted([f for f in os.listdir(input) if f.endswith(".txt")])

texts = []
file_names = []

for file in files:

    with open(os.path.join(input, file), "r", encoding="utf-8") as f:
        texts.append(f.read())

    file_names.append(file)


# Batch Processing

for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):

    batch_texts = texts[i:i+batch_size]
    batch_files = file_names[i:i+batch_size]

    summaries = summarize_batch(batch_texts)

    for file, summary in zip(batch_files, summaries):

        output_path = os.path.join(output, file)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(summary)

print("All summaries generated.")