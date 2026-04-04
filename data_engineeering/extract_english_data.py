import os
import shutil

# Only consider these decades
decades = {"2000s", "2010s", "2020s"}

output = "/Users/senudaliyanage/Downloads/IIT/Final Year Project/Legal Document Summarization/english_data_clean"

# Skip empty files.
def has_text(file_path):
    """Return True if file contains non-whitespace text"""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return len(f.read().strip()) > 0
    except Exception:
        return False

# Extract English, non-empty legal documents from 2000s onwards
def extract_english_docs(source_root, output_subfolder):
    """
    Extract English, non-empty legal documents from 2000s onwards
    """
    dest_root = os.path.join(output, output_subfolder)
    os.makedirs(dest_root, exist_ok=True)

    copied_count = 0
    skipped_empty = 0

    for decade in decades:
        decade_path = os.path.join(source_root, decade)
        if not os.path.isdir(decade_path):
            continue

        for year in os.listdir(decade_path):
            year_path = os.path.join(decade_path, year)
            if not os.path.isdir(year_path):
                continue

            dest_year_path = os.path.join(dest_root, year)
            os.makedirs(dest_year_path, exist_ok=True)

            for folder in os.listdir(year_path):
                if not folder.endswith("-en"):
                    continue

                folder_path = os.path.join(year_path, folder)
                doc_path = os.path.join(folder_path, "doc.txt")

                if not os.path.isfile(doc_path):
                    continue

                if not has_text(doc_path):
                    skipped_empty += 1
                    continue

                parts = folder.split("-")
                clean_name = f"{parts[0]}-{parts[1]}-{parts[2]}-{parts[6]}.txt"
                dest_file_path = os.path.join(dest_year_path, clean_name)

                shutil.copy2(doc_path, dest_file_path)
                copied_count += 1

    print(f"{output_subfolder.upper()} extracted: {copied_count} files")
    print(f"{output_subfolder.upper()} skipped empty files: {skipped_empty}\n")


acts_source = "/Users/senudaliyanage/Downloads/IIT/Final Year Project/Legal Document Summarization/legal_dataset/lk_legal_docs-data_lk_acts/data/lk_acts"
bills_source = "/Users/senudaliyanage/Downloads/IIT/Final Year Project/Legal Document Summarization/legal_dataset/lk_legal_docs-data_lk_bills/data/lk_bills"

extract_english_docs(acts_source, "acts")
extract_english_docs(bills_source, "bills")

print("Date extractions completed successfully.")
