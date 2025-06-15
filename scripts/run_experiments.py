# scripts/run_experiments.py
import pandas as pd
import os
import torch
import numpy as np
from tqdm import tqdm

# Add project root to path to allow imports from src
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.data_loader import load_and_split_data
from src.models.lexical_retrievers import BM25Retriever, TFIDFRetriever
from src.models.dense_retrievers import SentenceTransformerRetriever
from src.evaluate import evaluate_retrieval
from config import EVAL_K_VALUES, NDCG_K, RESULTS_DIR, MODEL_CONFIGS

def main():
    # --- 1. Load and Prepare Data ---
    # The loader now handles reading CSVs and splitting the data.
    # It returns data structures ready for retrieval.
    test_queries, candidate_corpus, test_ground_truth_by_par_id, candidate_ids_map = load_and_split_data(data_path='./preprocessed')

    if test_queries is None:
        return # Exit if data loading failed

    # The evaluation functions need query indices from 0 to N-1, not original PAR_IDs.
    # We create a map from the query's original PAR_ID to its index in the test_queries list.
    query_par_id_to_idx = {query['query_id']: i for i, query in enumerate(test_queries)}

    # We convert the ground truth to use these new 0-based query and corpus indices.
    test_ground_truth_by_idx = {}
    for par_id, relevant_docs in test_ground_truth_by_par_id.items():
        query_idx = query_par_id_to_idx.get(par_id)
        if query_idx is not None:
            test_ground_truth_by_idx[query_idx] = relevant_docs

    # --- 2. Define Models ---
    max_k_retrieval = max(EVAL_K_VALUES)
    all_results_summary = []

    models_to_run = {
        "BM25": BM25Retriever(),
        "TF-IDF-1gr": TFIDFRetriever(ngram_range=(1,1)),
        "TF-IDF-2gr": TFIDFRetriever(ngram_range=(1,2)),
        "SBERT_ZS": SentenceTransformerRetriever(MODEL_CONFIGS["SBERT"]["id"]),
        # Add other models here as needed, e.g., fine-tuned models
    }

    # --- 3. Run Experiments ---
    for model_name, model_instance in models_to_run.items():
        print(f"\n--- Running {model_name} ---")

        # Index the candidate corpus
        model_instance.index_corpus(candidate_corpus)

        # The retrieve method for all models should return scores for ALL documents in the corpus.
        # This is a change from the previous design. Let's assume this or adapt.
        # For this experiment, we'll build the full score matrix.

        num_queries = len(test_queries)
        num_candidates = len(candidate_corpus)
        scores_matrix = np.zeros((num_queries, num_candidates))

        print(f"Retrieving scores for {num_queries} queries...")
        for i, query_obj in enumerate(tqdm(test_queries, desc=f"Querying with {model_name}")):
            # The retrieve method in the base class should be modified to return a full score vector
            # For now, let's adapt here.
            if hasattr(model_instance, 'get_scores_for_all'):
                 scores = model_instance.get_scores_for_all(query_obj['query_text'])
            else:
                 # Fallback for models that only return top_k
                 retrieved_scores = model_instance.retrieve(query_obj['query_text'], top_k=num_candidates)
                 # Create a full score vector
                 scores = np.zeros(num_candidates)
                 for doc_id, score in retrieved_scores:
                     scores[doc_id] = score
            scores_matrix[i] = scores


        # Evaluate the results
        print("Evaluating...")
        metrics = evaluate_retrieval(
            scores_matrix,
            test_ground_truth_by_idx,
            EVAL_K_VALUES,
            NDCG_K
        )

        print(f"Results for {model_name}: {metrics}")
        result_row = {"Method": model_name}
        result_row.update(metrics)
        all_results_summary.append(result_row)

    # --- 4. Save Summary ---
    summary_df = pd.DataFrame(all_results_summary)
    cols_order = ["Method"] + [f"Recall@{k}" for k in EVAL_K_VALUES] + [f"nDCG@{NDCG_K}", "MRR"]
    cols_present = [col for col in cols_order if col in summary_df.columns]
    summary_df = summary_df[cols_present]

    output_csv_path = os.path.join(RESULTS_DIR, "performance_summary.csv")
    summary_df.to_csv(output_csv_path, index=False, float_format='%.4f')
    print(f"\nFull performance summary saved to: {output_csv_path}")
    print(summary_df)


if __name__ == "__main__":
    # Add a check for the preprocessed directory
    if not os.path.isdir('./preprocessed'):
        print("Error: The './preprocessed' directory was not found.")
        print("Please ensure your 'unique_pars.csv' and 'citations_cleaned.csv' files are inside a 'preprocessed' folder in the project root.")
    else:
        main()