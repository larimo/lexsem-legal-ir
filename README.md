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
    git clone [https://github.com/yourusername/cjeu-passage-retrieval.git](https://github.com/yourusername/cjeu-passage-retrieval.git)
    cd cjeu-passage-retrieval
    ```

2.  **Create a Project Directory for Data:**
    Create a folder named `preprocessed` in the root of the project.
    ```bash
    mkdir preprocessed
    ```

3.  **Add Data Files:**
    Place your data files (`unique_pars.csv`, `unique_cases.csv`, and `citations_cleaned.csv`) inside the `preprocessed` directory. The project structure should look like this:

    ```
    cjeu-passage-retrieval/
    ├── preprocessed/
    │   ├── unique_pars.csv
    │   ├── unique_cases.csv
    │   └── citations_cleaned.csv
    ├── src/
    ├── scripts/
    └── ... (other files)
    ```

4.  **Create a Python virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

5.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Running Experiments

To run all models (lexical and dense) and evaluate them on the test set:
```bash
python -m scripts.run_experiments

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
