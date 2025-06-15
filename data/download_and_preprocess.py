# data/download_and_preprocess.py
import datasets
import pandas as pd
import os
from src.utils import save_jsonl
from config import DATASET_NAME, TRAIN_FILE, VALID_FILE, TEST_FILE, CORPUS_FILE

def main():
    print("Downloading dataset...")
    # Load the default configuration which includes unique_pars.csv and edge_lists
    dataset = datasets.load_dataset(DATASET_NAME)

    print("Processing unique paragraphs...")
    # The 'datasets' library loads each CSV as a split. 'unique_pars' is the key for unique_pars.csv
    # 1. Create a dictionary mapping paragraph_id to its text
    par_dataset = datasets.load_dataset(DATASET_NAME, data_files="unique_pars.csv.gz")
    paragraphs_text_map = {row['PAR_ID']: row['TEXT'] for row in par_dataset['train']}

    # This dictionary will store all unique paragraphs encountered to create the final corpus file
    all_paragraphs = {}

    # 2. Process each split (train, validation, test)
    for split_name in ['train', 'validation', 'test']:
        print(f"Processing '{split_name}' split...")
        current_split_data = dataset[split_name]
        breakpoint()

        # This list will hold structured data with texts included
        processed_data_with_text = []

        # 3. For each edge, construct IDs and look up text
        for edge in current_split_data:
            # Construct the full paragraph IDs from the celex and paragraph number
            citing_id = edge['PAR_ID_FROM']
            cited_id = edge['PAR_ID_TO']

            # Look up the text from the map we created
            citing_text = paragraphs_text_map.get(citing_id)
            cited_text = paragraphs_text_map.get(cited_id)
            breakpoint()
            
            # Ensure both texts were found before adding
            if citing_text is not None and cited_text is not None:
                processed_data_with_text.append({
                    'citing_paragraph_id': citing_id,
                    'cited_paragraph_id': cited_id,
                    'citing_text': citing_text,
                    'cited_text': cited_text,
                })

                # Add the paragraphs to our master dictionary for the corpus
                if citing_id not in all_paragraphs:
                    all_paragraphs[citing_id] = citing_text
                if cited_id not in all_paragraphs:
                    all_paragraphs[cited_id] = cited_text

        # 4. Convert to a DataFrame and group by citing paragraph (query)
        df = pd.DataFrame(processed_data_with_text)

        if not df.empty:
            # Group by citing paragraph to collect all cited_paragraph_ids for a given query
            grouped = df.groupby('citing_paragraph_id').agg(
                citing_text=('citing_text', 'first'),
                cited_paragraph_ids=('cited_paragraph_id', list)
            ).reset_index()

            queries_for_split = []
            for _, row in grouped.iterrows():
                queries_for_split.append({
                    "query_id": row['citing_paragraph_id'],
                    "query_text": row['citing_text'],
                    "relevant_docs": row['cited_paragraph_ids']
                })

            # 5. Save the processed file
            if split_name == 'train':
                file_path = TRAIN_FILE
            elif split_name == 'validation':
                file_path = VALID_FILE
            else: # test
                file_path = TEST_FILE

            save_jsonl(queries_for_split, file_path)
            print(f"Saved {len(queries_for_split)} {split_name} queries to {file_path}")
        else:
            print(f"Warning: No data processed for split '{split_name}'.")

    # 6. Create the final corpus file from all unique paragraphs found
    corpus_data = [{"doc_id": pid, "text": text} for pid, text in all_paragraphs.items()]
    save_jsonl(corpus_data, CORPUS_FILE)
    print(f"Saved {len(corpus_data)} unique paragraphs to {CORPUS_FILE}")

if __name__ == "__main__":
    main()