# src/data_loader.py
import pandas as pd
import ast
import os

def load_and_split_data(data_path='preprocessed'):
    """
    Loads and splits the dataset based on the logic from the notebooks.
    """
    print("Loading data from CSV files...")
    # --- 1. Load Raw Data ---
    try:
        unique_pars_path = os.path.join(data_path, 'unique_pars.csv')
        citations_path = os.path.join(data_path, 'citations_cleaned.csv')

        unique_pars = pd.read_csv(unique_pars_path)
        # Load citations as a raw edge list (PAR_ID_TO is an integer column)
        df_citations = pd.read_csv(citations_path)

    except FileNotFoundError as e:
        print(f"Error: Data file not found. Make sure '{e.filename}' exists.")
        print(f"Please place your `preprocessed` directory with CSV files in the project's root folder.")
        return None, None, None, None, None
    
    print("Splitting data into train, validation, and test sets...")
    # --- 2. Perform Temporal Splits ---
    years_in_val = [2017, 2018]
    years_in_test = [2019, 2020, 2021]

    # Add a 'YEAR' column to paragraphs for splitting
    unique_pars['YEAR'] = pd.to_datetime(unique_pars['DATE']).dt.year

    # Split paragraphs by year
    train_pars_df = unique_pars[unique_pars['YEAR'] < years_in_val[0]]
    val_pars_df = unique_pars[unique_pars['YEAR'].isin(years_in_val)]
    test_pars_df = unique_pars[unique_pars['YEAR'].isin(years_in_test)]
    
    # --- 3. Aggregate Citations into Lists ---
    # This is the key step: for each source paragraph, create a list of all target paragraphs.
    citations_agg = df_citations.groupby('PAR_ID_FROM')['PAR_ID_TO'].apply(list).reset_index()

    # Add the year of the citing paragraph to the aggregated citations for splitting
    citations_agg = pd.merge(citations_agg, unique_pars[['PAR_ID', 'YEAR']], left_on='PAR_ID_FROM', right_on='PAR_ID', how='left')

    # Split aggregated citations based on the year of the citing paragraph
    train_citations = citations_agg[citations_agg['YEAR'] < years_in_val[0]]
    test_citations = citations_agg[citations_agg['YEAR'].isin(years_in_test)]
    
    # Filter test citations to remove those citing paragraphs from the test period
    test_pars_ids = set(test_pars_df['PAR_ID'])
    # This line will now work correctly because 'PAR_ID_TO' is a list
    within_citation_mask = test_citations['PAR_ID_TO'].apply(lambda targets: any(t in test_pars_ids for t in targets))
    test_citations_filtered = test_citations[~within_citation_mask]
    print(f"Original test queries: {len(test_citations)}. Filtered to {len(test_citations_filtered)} to remove citations to other test-set paragraphs.")

    # --- 4. Prepare Data Structures for Retrieval ---
    # Candidate corpus: All paragraphs from train and validation sets
    candidate_pars_df = pd.concat([train_pars_df, val_pars_df]).reset_index(drop=True)
    candidate_pars_df['corpus_idx'] = candidate_pars_df.index
    candidate_ids_map = pd.Series(candidate_pars_df.corpus_idx.values, index=candidate_pars_df.PAR_ID).to_dict()

    candidate_corpus = candidate_pars_df.apply(
        lambda row: {'doc_id': row['corpus_idx'], 'text': str(row['TEXT'])},
        axis=1
    ).tolist()
    
    # Full corpus dictionary for fine-tuning text lookup
    all_corpus_dict = pd.Series(unique_pars.TEXT.values, index=unique_pars.PAR_ID).to_dict()

    # Queries: The source paragraphs from the filtered test set
    test_query_pars_df = unique_pars[unique_pars['PAR_ID'].isin(test_citations_filtered['PAR_ID_FROM'])]
    test_queries = test_query_pars_df.apply(
        lambda row: {'query_id': row['PAR_ID'], 'query_text': str(row['TEXT'])},
        axis=1
    ).tolist()

    # Ground Truth: Map query IDs to their relevant document IDs (using the new corpus indices)
    test_ground_truth = {}
    for _, row in test_citations_filtered.iterrows():
        query_id = row['PAR_ID_FROM']
        relevant_docs = [candidate_ids_map[target_id] for target_id in row['PAR_ID_TO'] if target_id in candidate_ids_map]
        if relevant_docs:
             test_ground_truth[query_id] = relevant_docs

    valid_query_ids = set(test_ground_truth.keys())
    test_queries = [q for q in test_queries if q['query_id'] in valid_query_ids]

    # Training Queries for fine-tuning
    train_ground_truth_full = {row['PAR_ID_FROM']: row['PAR_ID_TO'] for _, row in train_citations.iterrows()}
    train_query_pars_df = unique_pars[unique_pars['PAR_ID'].isin(train_ground_truth_full.keys())]
    train_queries_full = train_query_pars_df.apply(
        lambda row: {'query_id': row['PAR_ID'], 'query_text': str(row['TEXT'])},
        axis=1
    ).tolist()
        
    # Add relevant docs to train_queries
    for q in train_queries_full:
        q['relevant_docs'] = train_ground_truth_full.get(q['query_id'], [])

    print(f"Data loading complete. Found {len(train_queries_full)} training queries, {len(test_queries)} test queries, and a corpus of {len(candidate_corpus)} documents.")

    return train_queries_full, test_queries, candidate_corpus, test_ground_truth, all_corpus_dict