import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class FaissIndexer:
    def __init__(self, index_path: str, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.indexer = faiss.IndexFlatL2(self.model.get_sentence_embedding_dimension())
        self.docs = []
        self.embeddings = []
        self.index_path = index_path

    def index(self, docs: list[str]):
        pass