import os
import tiktoken


input = "Data/english_data_clean_split/train"
output = "Data/chunked_train_data"
chunk_size = 2000
overlap_percent = 0.15
model_name = "gpt-5-mini"

os.makedirs(output, exist_ok=True)

encoding = tiktoken.encoding_for_model(model_name)

# Function to chunk tokens with specified chunk size and overlap
def chunk_tokens(tokens, chunk_size, overlap_percent):
    overlap_tokens = int(chunk_size * overlap_percent)
    step = chunk_size - overlap_tokens
    
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk = tokens[start:end]
        chunks.append(chunk)
        start += step
    return chunks

# Looping through all text files in the input, chunking them, and saving to output
for root, dirs, files in os.walk(input):
    for filename in files:
        if not filename.endswith(".txt"):
            continue
        
        file_path = os.path.join(root, filename)

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        tokens = encoding.encode(text)
        token_chunks = chunk_tokens(tokens, chunk_size, overlap_percent)

        base_name = os.path.splitext(filename)[0]

        for i, chunk in enumerate(token_chunks, start=1):
            chunk_text = encoding.decode(chunk)
            output_filename = f"{base_name}_chunk_{i}.txt"
            output_path = os.path.join(output, output_filename)

            with open(output_path, "w", encoding="utf-8") as out_f:
                out_f.write(chunk_text)

print("Train Data Chunking completed successfully.")