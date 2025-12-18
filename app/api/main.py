from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

from app.rag.vectorstore import load_vectorstore
from app.rag.retrieve import retrieve_documents
from app.rag.answer import answer_question, detect_intent

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(
    title="Git RAG API",
    description="API for answering Git questions using Retrieval-Augmented Generation.",
    version="0.1.0",
)

# -----------------------------
# Global resources (HF-safe)
# -----------------------------
VECTORSTORE = None


# -----------------------------
# Schemas
# -----------------------------
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


# -----------------------------
# Startup
# -----------------------------
@app.on_event("startup")
def startup_event():
    """
    Load heavy resources once.
    Hugging Face Spaces REALLY needs this.
    """
    global VECTORSTORE
    try:
        VECTORSTORE = load_vectorstore()
        print("✅ Vectorstore loaded successfully.")
    except Exception as e:
        print("❌ Failed to load vectorstore:", e)
        VECTORSTORE = None


# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def root():
    return {
        "message": "Welcome to Git RAG API",
        "version": "0.1.0",
        "docs": "/docs",
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if VECTORSTORE is None:
        raise HTTPException(
            status_code=500,
            detail="Vectorstore not available.",
        )

    try:
        intent = detect_intent(req.question)

        docs = retrieve_documents(
            query=req.question,
            k=req.top_k,
            vectorstore=VECTORSTORE,  # <-- important
        )

        answer = answer_question(req.question, docs)

        sources = [
            {
                "file": doc.metadata.get("source", "unknown"),
                "snippet": doc.page_content[:300],
            }
            for doc in docs
        ]

        return {
            "answer": answer.strip(),
            "intent": intent,
            "sources": sources,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process request: {str(e)}",
        )
