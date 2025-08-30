from fastapi import FastAPI
import ollama

from .dto import PromptRequest
from .rag_pipeline import RAGPipeline

api = FastAPI()

@api.post("/ask_ollama3")
async def ask_ollama3(request: PromptRequest):
    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": request.prompt}]
    )
    return {"answer": response['message']['content']}


@api.post("/ask_with_rag")
async def ask_with_rag(request: PromptRequest):

    # TODO: Implement RAG

    rag_pipeline = RAGPipeline(index_path="data/faiss_index.bin", text_path="data/teste.txt", with_ranker=False)

    response_rag = rag_pipeline.run(request)

    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": response_rag.prompt}]
    )

    return {"answer": response['message']['content']}