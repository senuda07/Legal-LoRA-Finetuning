import os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model = "mistralai/Mistral-Small-24B-Instruct-2501"

input = "chunked_train_data"
output = "mistral24B_summaries_train_chunked"

max_new_tokens = 200

os.makedirs(output, exist_ok=True)

# 4-bit quantization configuration with NF4 and double quantization

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model,
    device_map="auto",
    quantization_config=bnb_config,
    torch_dtype=torch.float16
)

model.eval()

# Summarization function
def summarize(text):

    prompt = f"""You are a legal research assistant.

You are a legal research assistant specialized in summarizing legislative documents.
The included text is part of a larger summary.

Create accurate, concise, and short summaries in plain English.
Preserve key legal clauses, definitions, obligations, and relationships.
Do not introduce new information.
Avoid unnecessary simplification.
Maintain legal terminology where important.


Text:
{text}

Summary:"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.2,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    # extract ONLY generated tokens
    generated_tokens = outputs[0][inputs.input_ids.shape[1]:]

    summary = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

    return summary

# Process files
files = sorted(os.listdir(input))

for file in tqdm(files):
    if not file.endswith(".txt"):
        continue
    with open(os.path.join(input, file), "r", encoding="utf-8") as f:
        text = f.read()
    summary = summarize(text)
    with open(os.path.join(output, file), "w", encoding="utf-8") as f:
        f.write(summary)

print("All summaries generated.")