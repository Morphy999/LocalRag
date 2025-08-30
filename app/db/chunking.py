from transformers import AutoTokenizer

class Chunker:
    def __init__(self, chunk_size: int = 100, chunk_overlap: int = 10, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def count_tokens(self, text: str)->int:
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        return len(tokens)

    def hybrid_chunk(self, text: str)->list[str]:
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
        chunks = []
        current_chunk = []
        current_len = 0

        for p in paragraphs:
            p_len = self.count_tokens(p)
            if p_len > self.chunk_size:
                words = p.split()
                start = 0
                while start < len(words):
                    sub = " ".join(words[start:start+self.chunk_size])
                    chunks.append(sub)
                    start += self.chunk_size - self.chunk_overlap
                continue

            if current_len + p_len <= self.chunk_size:
                current_chunk.append(p)
                current_len += p_len
            else:
                chunk_text = " ".join(current_chunk)
                chunks.append(chunk_text)

                overlap_text = " ".join(chunk_text.split()[-self.chunk_overlap:])
                current_chunk = [overlap_text, p]
                current_len = self.count_tokens(overlap_text + " " + p)

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks