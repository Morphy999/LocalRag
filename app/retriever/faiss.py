from pathlib import Path
from ..db.indexer import FaissIndexer
from ..db.chunking import Chunker
from ..utils import resolve_path

class FaissRetriever:
    def __init__(self, index_path: str, text_path: str = 'data/teste.txt', chunk_size: int = 100, chunk_overlap: int = 20):
        self.index_path = index_path
        self.indexer = FaissIndexer(index_path=index_path)
        self.chunker = Chunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.text_path = resolve_path(text_path)

    def retrieve(self, prompt: str, k: int = 5) -> list[str]:
        print("carregando index")
        if not self.indexer.load_index():
            get_texts = self.get_texts(self.text_path)
            
            print("gerando chunks")
            chunks = self.chunker.hybrid_chunk(get_texts)
            print(f"Quantidade de chunks: {len(chunks)}")
            
            print("construindo index")
            self.indexer.build_index(chunks)

        print("retornando resultados")
        return self.indexer.retrieve(prompt, k)

    def get_texts(self, text_path: Path) -> str:
        with open(text_path, "r", encoding="utf-8") as file:
            return file.read()
