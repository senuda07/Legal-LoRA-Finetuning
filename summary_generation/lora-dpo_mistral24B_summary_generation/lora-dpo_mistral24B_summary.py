import os
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

base_model  = "mistralai/Mistral-Small-24B-Instruct-2501"
adapter_id  = "senuda07/legal-mistral-qlora-dpo"

input_dir  = "chunked_test_data"
output_dir = "lora-dpo_mistral24B_results"

max_new_tokens = 200

os.makedirs(output_dir, exist_ok=True)

# Quantization config for 4-bit loading
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# Load tokenizer and model
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(adapter_id)  # use adapter's tokenizer
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="auto",
    quantization_config=bnb_config,
    dtype=torch.float16
)

print("Loading adapter...")
model = PeftModel.from_pretrained(base_model, adapter_id)
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

    generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
    summary = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    return summary

# Process files
files = sorted(os.listdir(input_dir))

for file in tqdm(files):
    if not file.endswith(".txt"):
        continue
    with open(os.path.join(input_dir, file), "r", encoding="utf-8") as f:
        text = f.read()
    summary = summarize(text)
    with open(os.path.join(output_dir, file), "w", encoding="utf-8") as f:
        f.write(summary)

print("All summaries generated.")