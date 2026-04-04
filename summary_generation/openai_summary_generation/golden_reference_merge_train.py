import os
import re
from collections import defaultdict

input = "results/openai_results/openai_summaries_train_chunked"
ouput = "results/openai_results/openai_summaries_train_merged"

os.makedirs(ouput, exist_ok=True)

# Pattern to capture document id and chunk number
pattern = re.compile(r"(.*)_chunk_(\d+)\.txt")

documents = defaultdict(list)

# Collect files
for filename in os.listdir(input):
    if not filename.endswith(".txt"):
        continue

    match = pattern.match(filename)
    if match:
        doc_id = match.group(1)
        chunk_num = int(match.group(2))

        documents[doc_id].append((chunk_num, filename))

# Merge chunks
for doc_id, chunks in documents.items():

    # Sort by chunk number
    chunks.sort(key=lambda x: x[0])

    merged_text = []

    for chunk_num, filename in chunks:
        path = os.path.join(input, filename)

        with open(path, "r", encoding="utf-8") as f:
            merged_text.append(f.read())

    output_path = os.path.join(ouput, f"{doc_id}.txt")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(merged_text))

print(f"Merged {len(documents)} documents.")