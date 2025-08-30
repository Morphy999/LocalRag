from .retriever.faiss import FaissRetriever
from .dto import PromptRequest

class RAGPipeline:
    def __init__(self, index_path: str = None, text_path: str = None, with_ranker: bool = True):
        self.with_ranker = with_ranker
        self.retriever = FaissRetriever(index_path=index_path, text_path=text_path)
        self.text_path = text_path

    def run(self, prompt: PromptRequest)->PromptRequest:
    
        retriever_result:list[str] = self.retriever.retrieve(prompt.prompt)

        if self.with_ranker:
            ranked_results:list[str] = self.ranker(retriever_result)
        else:
            ranked_results:list[str] = retriever_result

        generator_prompt:PromptRequest = self.generator(prompt.prompt, ranked_results)

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