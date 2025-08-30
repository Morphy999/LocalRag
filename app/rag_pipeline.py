from retriever.faiss import FaissRetriever
from dto import PromptRequest
import ollama

class RAGPipeline:
    def __init__(self, index_path: str, with_ranker: bool = True):
        self.with_ranker = with_ranker
        self.retriever = FaissRetriever(index_path=index_path)

    def run(self, prompt: PromptRequest)->PromptRequest:
        retriever_result = self.retriever.retrieve(prompt.prompt)

        if self.with_ranker:
            ranked_results  = self.ranker(retriever_result)
        else:
            ranked_results  = retriever_result

        generator_prompt = self.generator(prompt.prompt, ranked_results)

        return generator_prompt

    def ranker(self, prompt: PromptRequest)->list[str]:
        pass

    def generator(self, query: str, docs: list[str]) -> PromptRequest:
        context = "\n".join(docs)
        prompt = f"""
        
        Você é um assistente especializado. Use o contexto abaixo para responder:

        Contexto:
        {context}

        Pergunta: {query}
        """
        return PromptRequest(prompt=prompt)