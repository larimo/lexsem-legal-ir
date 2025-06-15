import pandas as pd
import os
import numpy as np
from tqdm import tqdm

import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.data_loader import load_and_split_data
from src.models.dense_retrievers import SentenceTransformerRetriever, OpenAIRetriever
from src.evaluate import evaluate_retrieval
from config import (EVAL_K_VALUES, NDCG_K, RESULTS_DIR, MODEL_CONFIGS, OPENAI_API_KEY)

def main():
    print("--- Running Zero-Shot Dense Models ---")

    # --- 1. Load Data ---
    _, test_queries, candidate_corpus, test_ground_truth_by_par_id, _ = load_and_split_data(data_path='./preprocessed')
    if test_queries is None:
        return

    query_par_id_to_idx = {query['query_id']: i for i, query in enumerate(test_queries)}
    test_ground_truth_by_idx = {query_par_id_to_idx.get(par_id): docs for par_id, docs in test_ground_truth_by_par_id.items() if par_id in query_par_id_to_idx}

    # --- 2. Define Models ---
    models_to_run = {
        "SBERT": SentenceTransformerRetriever(MODEL_CONFIGS["SBERT"]["id"]),
        "SimCSE": SentenceTransformerRetriever(MODEL_CONFIGS["SimCSE"]["id"]),
        "Nomic": SentenceTransformerRetriever(MODEL_CONFIGS["Nomic"]["id"]),
        "Ada-v2": OpenAIRetriever(MODEL_CONFIGS["Ada-v2"]["id"]),
        "Emb-3-large": OpenAIRetriever(MODEL_CONFIGS["Emb-3-large"]["id"]),
    }

    # Check for OpenAI API key if those models are included
    if "Ada-v2" in models_to_run or "Emb-3-large" in models_to_run:
        if not OPENAI_API_KEY:
            print("Warning: OPENAI_API_KEY not found. Skipping OpenAI models.")
            del models_to_run["Ada-v2"]
            del models_to_run["Emb-3-large"]

    all_results_summary = []

    # --- 3. Run Experiments ---
    for model_name, model_instance in models_to_run.items():
        print(f"\n--- Evaluating {model_name} ---")
        model_instance.index_corpus(candidate_corpus)

        num_queries = len(test_queries)
        num_candidates = len(candidate_corpus)
        scores_matrix = np.zeros((num_queries, num_candidates))

        print(f"Retrieving scores for {num_queries} queries...")
        for i, query_obj in enumerate(tqdm(test_queries, desc=f"Querying with {model_name}")):
            retrieved_scores = model_instance.retrieve(query_obj['query_text'], top_k=num_candidates)
            scores = np.zeros(num_candidates)
            for doc_idx, score in retrieved_scores:
                if doc_idx < num_candidates:
                    scores[doc_idx] = score
            scores_matrix[i] = scores

        print("Evaluating...")
        metrics = evaluate_retrieval(scores_matrix, test_ground_truth_by_idx, EVAL_K_VALUES, NDCG_K)

        print(f"Results for {model_name}: {metrics}")
        result_row = {"Method": model_name}
        result_row.update(metrics)
        all_results_summary.append(result_row)

    # --- 4. Save Summary ---
    summary_df = pd.DataFrame(all_results_summary)
    output_csv_path = os.path.join(RESULTS_DIR, "dense_zero_shot_performance.csv")
    summary_df.to_csv(output_csv_path, index=False, float_format='%.4f')
    print(f"\nZero-shot dense models performance summary saved to: {output_csv_path}")
    print(summary_df)

if __name__ == "__main__":
    main()