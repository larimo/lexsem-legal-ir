# scripts/fine_tune_sbert.py
import argparse
import os
from src.data_loader import load_data_splits
from src.models.dense_retrievers import fine_tune_sentence_transformer
from config import FINETUNED_MODEL_DIR, MODEL_CONFIGS

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a SentenceTransformer model.")
    parser.add_argument("--base_model_key", type=str, required=True, 
                        choices=["SBERT-ft", "LegalSBERT-ft"], 
                        help="Key from MODEL_CONFIGS for the base model (e.g., SBERT-ft).")
    parser.add_argument("--output_model_name", type=str, required=True, 
                        help="Name for the output fine-tuned model directory.")
    # Add other fine-tuning hyperparams if needed (epochs, lr, batch_size from config.py)

    args = parser.parse_args()

    base_model_hf_name = MODEL_CONFIGS[args.base_model_key]['id']
    output_path = os.path.join(FINETUNED_MODEL_DIR, args.output_model_name)

    print(f"Loading data for fine-tuning {base_model_hf_name}...")
    train_queries, valid_queries, _, _, corpus_dict = load_data_splits()
    # Note: The paper uses training and validation sets for fine-tuning (or just training set).
    # "We employ the Multiple Negatives Ranking Loss ... to fine-tune both models ... over the training dataset."
    # So, we'll use train_queries. valid_queries could be used for an evaluator.

    print(f"Starting fine-tuning for {args.base_model_key} (HF: {base_model_hf_name}). Output to: {output_path}")
    fine_tune_sentence_transformer(
        base_model_name=base_model_hf_name,
        train_queries=train_queries, # Using only training queries as per paper's description of loss
        valid_queries=valid_queries, # Could be used for dev set in evaluator
        corpus_dict=corpus_dict,
        output_path=output_path
    )
    print(f"Fine-tuning complete. Model saved to {output_path}")

if __name__ == "__main__":
    main()