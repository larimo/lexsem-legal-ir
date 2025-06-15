# src/models/lexical_retrievers.py
from .base_retriever import BaseRetriever
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from src.utils import tokenize_text # Assuming a basic tokenizer
from config import BM25_K1, BM25_B, TFIDF_TOP_K_NGRAMS, TFIDF_NGRAM_RANGE

class BM25Retriever(BaseRetriever):
    def __init__(self, k1=BM25_K1, b=BM25_B):
        super().__init__("BM25")
        self.k1 = k1
        self.b = b
        self.bm25 = None
        self.doc_ids = []
        self.corpus_texts = []

    def index_corpus(self, corpus_data):
        self.doc_ids = [item['doc_id'] for item in corpus_data]
        self.corpus_texts = [item['text'] for item in corpus_data]
        tokenized_corpus = [tokenize_text(doc) for doc in self.corpus_texts]
        self.bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)
        print(f"BM25 index built with {len(self.doc_ids)} documents.")

    def retrieve(self, query_text, top_k):
        if not self.bm25:
            raise ValueError("BM25 model not indexed. Call index_corpus first.")
        tokenized_query = tokenize_text(query_text)
        doc_scores = self.bm25.get_scores(tokenized_query)
        
        # Get top_k scores and their indices
        top_n_indices = np.argsort(doc_scores)[::-1][:top_k]
        
        results = []
        for idx in top_n_indices:
            results.append((self.doc_ids[idx], doc_scores[idx]))
        return results
    
    def get_scores_for_all(self, query_text):
        """Returns scores for the query against all documents in the corpus."""
        if not self.bm25:
            raise ValueError("BM25 model not indexed. Call index_corpus first.")
        tokenized_query = tokenize_text(query_text)
        return self.bm25.get_scores(tokenized_query)

class TFIDFRetriever(BaseRetriever):
    def __init__(self, ngram_range=(1,1), top_k_features=TFIDF_TOP_K_NGRAMS, model_suffix=""):
        super().__init__(f"TF-IDF-{ngram_range[1]}gr{model_suffix}")
        self.ngram_range = ngram_range
        self.top_k_features = top_k_features # Paper: "top-K frequent N-grams... K=5,000"
        self.vectorizer = None
        self.corpus_matrix = None
        self.doc_ids = []

    def index_corpus(self, corpus_data):
        self.doc_ids = [item['doc_id'] for item in corpus_data]
        corpus_texts = [item['text'] for item in corpus_data]
        
        # The paper mentions "top-K frequent N-grams of the training set".
        # This implies the vocabulary should be built on the training texts.
        # For simplicity here, we build it on the provided corpus_data.
        # A more faithful implementation would pass training texts here.
        self.vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            tokenizer=tokenize_text, # Use consistent tokenizer
            max_features=self.top_k_features, # To mimic "top-K frequent N-grams"
            stop_words='english' # Optional, but common
        )
        self.corpus_matrix = self.vectorizer.fit_transform(corpus_texts)
        print(f"TF-IDF index built with {len(self.doc_ids)} documents and {self.corpus_matrix.shape[1]} features.")

    def retrieve(self, query_text, top_k):
        if not self.vectorizer:
            raise ValueError("TF-IDF model not indexed. Call index_corpus first.")
        
        query_vector = self.vectorizer.transform([query_text])
        scores = cosine_similarity(query_vector, self.corpus_matrix).flatten()
        
        top_n_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_n_indices:
            results.append((self.doc_ids[idx], scores[idx]))
        return results