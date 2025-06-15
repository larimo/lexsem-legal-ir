# src/models/base_retriever.py
from abc import ABC, abstractmethod

class BaseRetriever(ABC):
    """
    Abstract Base Class for all retrieval models.
    It defines the required structure for indexing and retrieving documents.
    """
    def __init__(self, model_name, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs # For model-specific params

    @abstractmethod
    def index_corpus(self, corpus_data):
        """
        Builds an index from the corpus.
        This method must be implemented by any subclass.
        
        Args:
            corpus_data (list): A list of dictionaries, e.g., [{"doc_id": "id1", "text": "..."}]
        """
        pass

    @abstractmethod
    def retrieve(self, query_text, top_k):
        """
        Retrieves top_k relevant document IDs for a given query_text.
        This method must be implemented by any subclass.

        Args:
            query_text (str): The query text.
            top_k (int): The number of top documents to retrieve.

        Returns:
            list: A list of (doc_id, score) tuples.
        """
        pass

    def batch_retrieve(self, queries, top_k):
        """
        Retrieves top_k relevant document IDs for a batch of queries.
        
        Args:
            queries (list): A list of dictionaries, e.g., [{"query_id": "q1", "query_text": "..."}]
            top_k (int): The number of top documents to retrieve for each query.

        Returns:
            dict: A dictionary mapping query_id to a list of (doc_id, score) tuples.
        """
        results = {}
        # Using tqdm for a progress bar, assuming it's installed
        try:
            from tqdm import tqdm
            iterator = tqdm(queries, desc=f"Batch retrieving with {self.model_name}")
        except ImportError:
            iterator = queries

        for query_obj in iterator:
            query_id = query_obj['query_id']
            query_text = query_obj['query_text']
            results[query_id] = self.retrieve(query_text, top_k)
        return results