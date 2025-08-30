from ..db.indexer import FaissIndexer
from ..db.chunking import Chunker


class FaissRetriever:
    def __init__(self, index_path: str, text_path: str = 'data/teste.txt', chunk_size: int = 100, chunk_overlap: int = 10):
        self.index_path = index_path
        self.indexer = FaissIndexer(index_path=index_path)
        self.chunker = Chunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.text_path = text_path

    def retrieve(self, prompt: str, k: int = 5)->list[str]:

        if not self.indexer.load_index():

            get_texts = self.get_texts(self.text_path)
            
            chunks = self.chunker.hybrid_chunk(get_texts)
            
            self.indexer.build_index(chunks)

        return self.indexer.retrieve(prompt, k)

    def get_texts(self, text_path: str)->str:
        with open(text_path, "r") as file:
            return file.read()