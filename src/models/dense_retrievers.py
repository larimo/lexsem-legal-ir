# src/models/dense_retrievers.py
from .base_retriever import BaseRetriever
from sentence_transformers import SentenceTransformer, InputExample, losses, util
from torch.utils.data import DataLoader
import torch
import numpy as np
import os
from openai import OpenAI
# from nomic import NomicClient # Placeholder for Nomic
from config import (OPENAI_API_KEY, NOMIC_API_KEY, MODEL_CONFIGS, 
                    FINETUNED_MODEL_DIR, FINETUNE_EPOCHS, FINETUNE_BATCH_SIZE)

class SentenceTransformerRetriever(BaseRetriever):
    def __init__(self, model_id_or_path):
        model_name_key = model_id_or_path.split('/')[-1] if '/' in model_id_or_path else model_id_or_path
        super().__init__(f"ST_{model_name_key}")
        self.model = SentenceTransformer(model_id_or_path)
        self.doc_ids = []
        self.corpus_embeddings = None

    def index_corpus(self, corpus_data):
        self.doc_ids = [item['doc_id'] for item in corpus_data]
        corpus_texts = [item['text'] for item in corpus_data]
        print(f"Encoding {len(corpus_texts)} documents for {self.model_name}...")
        self.corpus_embeddings = self.model.encode(corpus_texts, convert_to_tensor=True, show_progress_bar=True)
        print(f"{self.model_name} index built.")

    def retrieve(self, query_text, top_k):
        if self.corpus_embeddings is None:
            raise ValueError(f"{self.model_name} not indexed. Call index_corpus first.")
        
        query_embedding = self.model.encode(query_text, convert_to_tensor=True)
        # Cosine similarity
        cos_scores = util.cos_sim(query_embedding, self.corpus_embeddings)[0]
        cos_scores = cos_scores.cpu() # Move to CPU if on GPU

        # Get top_k scores
        top_results = torch.topk(cos_scores, k=min(top_k, len(self.doc_ids)))
        
        results = []
        for score, idx in zip(top_results[0], top_results[1]):
            results.append((self.doc_ids[idx], score.item()))
        return results

class OpenAIRetriever(BaseRetriever):
    def __init__(self, model_api_name):
        super().__init__(f"OpenAI_{model_api_name}")
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model_api_name = model_api_name # e.g., "text-embedding-ada-002"
        self.doc_ids = []
        self.corpus_embeddings = None # Will be a list of numpy arrays

    def _get_embedding(self, text):
        response = self.client.embeddings.create(input=[text], model=self.model_api_name)
        return np.array(response.data[0].embedding)

    def index_corpus(self, corpus_data):
        self.doc_ids = [item['doc_id'] for item in corpus_data]
        corpus_texts = [item['text'] for item in corpus_data]
        print(f"Encoding {len(corpus_texts)} documents for {self.model_name} via API...")
        # Batching would be more efficient here if API supports it well
        self.corpus_embeddings = []
        from tqdm import tqdm
        for text in tqdm(corpus_texts, desc=f"Embedding corpus for {self.model_name}"):
            self.corpus_embeddings.append(self._get_embedding(text))
        self.corpus_embeddings = np.array(self.corpus_embeddings) # Shape: (num_docs, embed_dim)
        print(f"{self.model_name} index built.")

    def retrieve(self, query_text, top_k):
        if self.corpus_embeddings is None:
            raise ValueError(f"{self.model_name} not indexed. Call index_corpus first.")
        
        query_embedding = self._get_embedding(query_text) # Shape: (embed_dim,)
        query_embedding = query_embedding.reshape(1, -1) # Shape: (1, embed_dim)
        
        # Cosine similarity (manual)
        # scores = np.dot(self.corpus_embeddings, query_embedding.T).flatten() / \
        #          (np.linalg.norm(self.corpus_embeddings, axis=1) * np.linalg.norm(query_embedding))
        # Using sklearn for robustness
        from sklearn.metrics.pairwise import cosine_similarity
        scores = cosine_similarity(query_embedding, self.corpus_embeddings).flatten()

        top_n_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_n_indices:
            results.append((self.doc_ids[idx], scores[idx]))
        return results

# Placeholder for Nomic Retriever - similar structure to OpenAIRetriever if API based
# class NomicRetriever(BaseRetriever): ...

def fine_tune_sentence_transformer(base_model_name, train_queries, valid_queries, corpus_dict, output_path):
    """
    Fine-tunes a SentenceTransformer model using MultipleNegativesRankingLoss.
    train_queries: list of {"query_id": ..., "query_text": ..., "relevant_docs": [...]}
    corpus_dict: {doc_id: text}
    """
    model = SentenceTransformer(base_model_name)
    
    train_examples = []
    for query_obj in train_queries:
        query_text = query_obj['query_text']
        # MultipleNegativesRankingLoss expects positive pairs.
        # The paper implies direct (query_text, cited_text) pairs.
        for relevant_doc_id in query_obj['relevant_docs']:
            if relevant_doc_id in corpus_dict:
                positive_text = corpus_dict[relevant_doc_id]
                train_examples.append(InputExample(texts=[query_text, positive_text], label=1)) # Label 1 for similarity
            # else:
                # print(f"Warning: Relevant doc_id {relevant_doc_id} not found in corpus_dict for training.")

    if not train_examples:
        print("No training examples generated. Check data.")
        return None

    print(f"Created {len(train_examples)} training examples for fine-tuning.")
    
    # The paper uses MultipleNegativesRankingLoss.
    # This loss expects (anchor, positive) pairs for each batch, and treats others in batch as negatives.
    # For this, we need to ensure our DataLoader provides batches where each example can be an anchor.
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=FINETUNE_BATCH_SIZE)
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    # TODO: Set up evaluator for validation during training (optional but good practice)
    # from sentence_transformers.evaluation import InformationRetrievalEvaluator
    # dev_examples = ... create (query, positive, negative) tuples or similar for IR eval
    # evaluator = InformationRetrievalEvaluator(queries_dev, corpus_dev, relevant_docs_dev)

    print(f"Starting fine-tuning of {base_model_name} for {FINETUNE_EPOCHS} epochs.")
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              epochs=FINETUNE_EPOCHS,
              warmup_steps=100, # Example
              output_path=output_path,
              show_progress_bar=True,
              # evaluator=evaluator, evaluation_steps=500 # If evaluator is set up
              )
    print(f"Fine-tuned model saved to {output_path}")
    return output_path