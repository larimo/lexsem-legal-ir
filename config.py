# config.py
import os
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env file for API keys

# --- Data ---
DATASET_NAME = "larimo/cjeu-paragraph-retrieval"
PROCESSED_DATA_DIR = "data/processed/" # Where preprocessed splits will be stored
TRAIN_FILE = os.path.join(PROCESSED_DATA_DIR, "train.jsonl")
VALID_FILE = os.path.join(PROCESSED_DATA_DIR, "valid.jsonl")
TEST_FILE = os.path.join(PROCESSED_DATA_DIR, "test.jsonl")
CORPUS_FILE = os.path.join(PROCESSED_DATA_DIR, "corpus.jsonl") # All unique paragraphs for indexing

# --- Models ---
# Lexical
BM25_K1 = 1.2
BM25_B = 0.75
TFIDF_TOP_K_NGRAMS = 5000
TFIDF_NGRAM_RANGE = (1, 2) # For TF-IDF-2gr in paper (1,1 for 1gr)

# Dense (Zero-shot identifiers from Hugging Face or API)
SBERT_ZS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SIMCSE_ZS_MODEL = "princeton-nlp/sup-simcse-roberta-base" # Check exact HF name
NOMIC_MODEL = "nomic-ai/nomic-embed-text-v1.5" # Check exact HF name or API details
OPENAI_ADA_V2_MODEL = "text-embedding-ada-002"
OPENAI_EMB_3_LARGE_MODEL = "text-embedding-3-large"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NOMIC_API_KEY = os.getenv("NOMIC_API_KEY") # If Nomic needs one

# Dense (Fine-tuning)
SBERT_FT_BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LEGAL_SBERT_FT_BASE_MODEL = "nlpaueb/legal-bert-base-uncased"
FINETUNED_MODEL_DIR = "models/fine_tuned/"
FINETUNE_EPOCHS = 3 # Example, paper uses default from Sentence Transformers library
FINETUNE_BATCH_SIZE = 16 # Example
FINETUNE_LR = 2e-5 # Example

# --- Evaluation ---
EVAL_K_VALUES = [1, 5, 10, 20]
NDCG_K = 10

# --- Output ---
RESULTS_DIR = "results/"
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

# Ensure directories exist
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(FINETUNED_MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Add model parameter counts and embedding dimensions as per Table 3
MODEL_CONFIGS = {
    "SBERT": {"params": "33M", "dim": 384, "id": SBERT_ZS_MODEL, "type": "hf_sentence_transformer"},
    "SBERT-ft": {"params": "33M", "dim": 384, "id": SBERT_FT_BASE_MODEL, "type": "finetuned_sentence_transformer"},
    "LegalSBERT-ft": {"params": "110M", "dim": 768, "id": LEGAL_SBERT_FT_BASE_MODEL, "type": "finetuned_sentence_transformer"},
    "SimCSE": {"params": "125M", "dim": 768, "id": SIMCSE_ZS_MODEL, "type": "hf_sentence_transformer"},
    "Nomic": {"params": "137M", "dim": 768, "id": NOMIC_MODEL, "type": "nomic_api"}, # or hf if available directly
    "Ada-v2": {"params": "N/A", "dim": 1536, "id": OPENAI_ADA_V2_MODEL, "type": "openai_api"},
    "Emb-3-large": {"params": "N/A", "dim": 3072, "id": OPENAI_EMB_3_LARGE_MODEL, "type": "openai_api"},
}