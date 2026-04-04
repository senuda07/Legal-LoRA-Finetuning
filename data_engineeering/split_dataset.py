import random
import shutil
from pathlib import Path


input = Path("/Users/senudaliyanage/Downloads/IIT/Final Year Project/Legal Document Summarization/english_data_clean")
output = Path("/Users/senudaliyanage/Downloads/IIT/Final Year Project/Legal Document Summarization/english_data_clean_split")

split_ratio = 0.8
random_seed = 42


random.seed(random_seed)

categories = ["acts", "bills"]

# To store counts
train_counts = {}
test_counts = {}

# Create output directories
for category in categories:
    source_path = input / category

    files = list(source_path.rglob("*.txt"))
    random.shuffle(files)

    split_index = int(len(files) * split_ratio)

    train_files = files[:split_index]
    test_files = files[split_index:]

    train_dest = output / "train" / category
    test_dest = output / "test" / category

    train_dest.mkdir(parents=True, exist_ok=True)
    test_dest.mkdir(parents=True, exist_ok=True)

    for file in train_files:
        shutil.copy(file, train_dest / file.name)

    for file in test_files:
        shutil.copy(file, test_dest / file.name)

    train_counts[category] = len(train_files)
    test_counts[category] = len(test_files)


# File Count
print("\nDataset Split Summary\n")

print("Train;")
print(f"Acts:  {train_counts.get('acts', 0)}")
print(f"Bills: {train_counts.get('bills', 0)}\n")

print("Test;")
print(f"Acts:  {test_counts.get('acts', 0)}")
print(f"Bills: {test_counts.get('bills', 0)}\n")

total_files = sum(train_counts.values()) + sum(test_counts.values())
print(f"Total Files: {total_files}")
