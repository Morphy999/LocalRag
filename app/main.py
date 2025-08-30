from fastapi import FastAPI
import ollama

from .dto import PromptRequest
from .rag_pipeline import RAGPipeline

api = FastAPI()

@api.post("/ask_ollama3_endpoint")
async def ask_ollama3_endpoint(request: PromptRequest):
    return ask_ollama3(request)


@api.post("/ask_ollama3_with_rag_endpoint")
async def ask_ollama3_with_rag_endpoint(request: PromptRequest):
    return ask_ollama3_with_rag(request)

    
def ask_ollama3(request: PromptRequest):

    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": request.prompt}]
    )

    return {"answer": response['message']['content']}


def ask_ollama3_with_rag(request: PromptRequest):

    rag_pipeline = RAGPipeline(index_path="data/faiss_index.bin", text_path="data/teste.txt", with_ranker=False)

    response_rag:PromptRequest = rag_pipeline.run(request)

    return ask_ollama3(response_rag)
