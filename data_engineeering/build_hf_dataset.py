from pathlib import Path
from datasets import Dataset, DatasetDict
import pandas as pd

data_dir = Path("/Users/senudaliyanage/Downloads/IIT/Final Year Project/Legal Document Summarization/english_data_clean_split")

# Function to load a split (train/test) and return a Hugging Face Dataset
def load_split(split_name):
    records = []
    split_path = data_dir / split_name

    for doc_type in ["acts", "bills"]:
        type_path = split_path / doc_type

        for file_path in type_path.glob("*.txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

            records.append({
                "index": file_path.name,
                "type": doc_type[:-1], 
                "text": text
            })

    return Dataset.from_pandas(pd.DataFrame(records))

# Build dataset
dataset = DatasetDict({
    "train": load_split("train"),
    "test": load_split("test")
})

print(dataset)

dataset.save_to_disk("final_legal_dataset_hf")

