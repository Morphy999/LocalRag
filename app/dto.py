from pydantic import BaseModel

class PromptRequest(BaseModel):
    prompt: str
    num_docs: int = 1