# Assessing the Performance Gap Between Lexical and Semantic Models for Information Retrieval With Formulaic Legal Language

This repository contains the code for the research paper investigating the performance of lexical and dense retrieval models for legal passage retrieval on Court of Justice of the European Union (CJEU) decisions. The paper analyzes when the formulaic citation structure of CJEU favors simpler statistical methods like BM25 over more complex dense models.

## Appendix
The paper mentions an appendix available at `https://github.com/larimo/lexsem-legal-ir`. The Arxiv version with the paper appendix is available at [To be added].

## Features

* Dataset loading and preprocessing for the CJEU paragraph retrieval task.
* Implementations/wrappers for:
    * Lexical models: BM25, TF-IDF.
    * Zero-shot dense models: SBERT, SimCSE, Nomic, OpenAI Ada-v2, OpenAI Emb-3-large.
    * Fine-tuned dense models: SBERT-ft, LegalSBERT-ft.
* Evaluation using Recall@k, nDCG@10, MAP, MRR.
* Qualitative and quantitative analysis of performance gaps.
* Ablation study on fine-tuning data size.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/lexsem-legal-ir.git](https://github.com/yourusername/lexsem-legal-ir.git)
    cd lexsem-legal-ir
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3.  **Download and preprocess the dataset:**
    The dataset is based on `larimo/cjeu-paragraph-retrieval` from Hugging Face.
    ```bash
    python data/download_dataset.py
    ```
    You might need to set your Hugging Face token for direct download, or manually download and place it.

4.  **Configure API Keys (Optional):**
    For OpenAI models, set your API key as an environment variable:
    ```bash
    export OPENAI_API_KEY="your_openai_api_key"
    ```
    Or modify `src/config.py`. The Nomic model might also require similar setup.

## Running Experiments

The `scripts/` directory contains scripts to reproduce the experiments:

* **Lexical Baselines:** `python scripts/run_lexical_baselines.py`
* **Zero-shot Dense Models:** `python scripts/run_dense_zero_shot.py`
* **Fine-tuning:** `python scripts/run_finetuning_experiments.py` 

Results (tables and figures) will be saved in the `results/` directory.

## Citation

If you use this code or refer to the findings, please cite the original paper:

[Paper Title and Citation Details - To be added once officially published]

A preliminary version was presented at the 6th Natural Legal Language Processing (NLLP) Workshop co-located with EMNLP 2024.
