from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from pathlib import Path
import json
from app.rag.vectorstore import load_vectorstore
from app.rag.retrieve import retrieve_documents
from app.rag.answer import answer_question, detect_intent

app = FastAPI(
    title="Git RAG API",
    description="An API for retrieving and answering questions about Git using RAG.",
    version="0.1.0"
)

class ChatRequest(BaseModel):
    question: str
    top_k: int = 4
    
class Source(BaseModel):
    file: str
    snippet: str
    
class ChatResponse(BaseModel):
    answer: str
    intent: str
    sources: List[Source]
    
@app.get("/")
def root():
    return {"message": "Welcome to Git RAG API", "version": "0.1.0", "docs": "/docs"}

@app.on_event("startup")
def startup_event():
    # Vectorstore is loaded inside retrieve_documents, so no need to preload here
    pass
    
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    intent = detect_intent(req.question)

    docs = retrieve_documents(
        query=req.question,
        k=req.top_k
    )

    answer = answer_question(req.question, docs)

    sources = []
    for doc in docs:
        sources.append({
            "file": doc.metadata.get("source", "unknown"),
            "snippet": doc.page_content[:300]
        })

    return {
        "answer": answer.strip(),
        "intent": intent,
        "sources": sources
    }