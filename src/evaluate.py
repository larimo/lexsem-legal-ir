# src/evaluate.py
import numpy as np
import torch
from sklearn.metrics import ndcg_score

def compute_recall_at_k(scores_matrix, ground_truth, k):
    """
    Computes Recall@k.

    Args:
        scores_matrix (torch.Tensor or np.ndarray): Shape (num_queries, num_docs).
        ground_truth (dict): {query_idx: [relevant_doc_indices]}.
        k (int): The "k" in Recall@k.

    Returns:
        float: The mean Recall@k across all queries.
    """
    if isinstance(scores_matrix, np.ndarray):
        scores_matrix = torch.from_numpy(scores_matrix)

    # Get top k indices for each query
    top_k_indices = torch.topk(scores_matrix, k=k, dim=1).indices

    recall_scores = []
    # Loop over queries by index
    for i in range(scores_matrix.shape[0]):
        # Get the set of retrieved document indices for this query
        retrieved_docs = set(top_k_indices[i].tolist())
        # Get the set of ground truth document indices for this query
        true_relevant_docs = set(ground_truth.get(i, []))

        if not true_relevant_docs:
            continue

        # Calculate the number of relevant documents that were retrieved
        num_found = len(retrieved_docs.intersection(true_relevant_docs))
        # Calculate recall for this query
        recall = num_found / len(true_relevant_docs)
        recall_scores.append(recall)

    return np.mean(recall_scores) if recall_scores else 0.0


def compute_mrr(scores_matrix, ground_truth):
    """
    Computes Mean Reciprocal Rank (MRR).

    Args:
        scores_matrix (torch.Tensor or np.ndarray): Shape (num_queries, num_docs).
        ground_truth (dict): {query_idx: [relevant_doc_indices]}.

    Returns:
        float: The MRR score.
    """
    if isinstance(scores_matrix, np.ndarray):
        scores_matrix = torch.from_numpy(scores_matrix)

    # Get the rankings of all documents for each query
    sorted_indices = torch.argsort(scores_matrix, dim=1, descending=True)

    reciprocal_ranks = []
    for i in range(scores_matrix.shape[0]):
        true_relevant_docs = set(ground_truth.get(i, []))
        if not true_relevant_docs:
            continue

        # Find the rank of the first relevant document
        rank = 0
        for r, doc_idx in enumerate(sorted_indices[i].tolist()):
            if doc_idx in true_relevant_docs:
                rank = r + 1 # Ranks are 1-based
                break

        reciprocal_ranks.append(1 / rank if rank > 0 else 0)

    return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0


def compute_ndcg_at_k(scores_matrix, ground_truth, k):
    """
    Computes nDCG@k.

    Args:
        scores_matrix (np.ndarray): Shape (num_queries, num_docs).
        ground_truth (dict): {query_idx: [relevant_doc_indices]}.
        k (int): The "k" in nDCG@k.

    Returns:
        float: The mean nDCG@k score.
    """
    if isinstance(scores_matrix, torch.Tensor):
        scores_matrix = scores_matrix.numpy()

    true_relevance = np.zeros_like(scores_matrix)
    for query_idx, relevant_docs in ground_truth.items():
        for doc_idx in relevant_docs:
            if doc_idx < true_relevance.shape[1]:
                 true_relevance[query_idx, doc_idx] = 1.0

    return ndcg_score(true_relevance, scores_matrix, k=k)

def evaluate_retrieval(scores_matrix, ground_truth, k_values_recall, k_ndcg):
    """
    Runs all standard evaluation metrics.

    Args:
        scores_matrix (np.ndarray): Scores for each query-document pair.
        ground_truth (dict): Ground truth mapping query index to list of relevant doc indices.
        k_values_recall (list): A list of k values for recall.
        k_ndcg (int): The k value for nDCG.

    Returns:
        dict: A dictionary of metric names and their scores.
    """
    metrics = {}

    # Recall@k
    for k in k_values_recall:
        metrics[f"Recall@{k}"] = compute_recall_at_k(scores_matrix, ground_truth, k)

    # nDCG@k
    metrics[f"nDCG@{k_ndcg}"] = compute_ndcg_at_k(scores_matrix, ground_truth, k_ndcg)

    # MRR
    metrics["MRR"] = compute_mrr(scores_matrix, ground_truth)

    return metrics