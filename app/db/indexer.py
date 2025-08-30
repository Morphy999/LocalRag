import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

class FaissIndexer:
    def __init__(self, index_path: str, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.indexer = faiss.IndexFlatL2(self.model.get_sentence_embedding_dimension())
        self.texts = []
        self.embeddings = None
        self.index_path = index_path

    def build_index(self, chunks: list[str]):
        
        self.texts = chunks.copy()

        self.embeddings = self.model.encode(self.texts, convert_to_numpy = True)
        
        self.indexer.add(np.array(self.embeddings))
        
        faiss.write_index(self.indexer, self.index_path)

        print(f"Index built and saved to {self.index_path}")


    def load_index(self)->bool:
        if Path(self.index_path).exists(): 
            self.indexer = faiss.read_index(self.index_path)
            print(f"Index loaded from {self.index_path}")
            return True
        else:
            print(f"Index not found at {self.index_path}")
            return False

    def retrieve(self, query: str, k: int = 5)->list[str]:
        query_embedding = self.model.encode(query, convert_to_numpy = True)
        _, indices = self.indexer.search(np.array([query_embedding]), k)
        return [self.texts[i] for i in indices[0]]