from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from app.rag.vectorstore import load_vectorstore
from app.rag.retrieve import retrieve_documents
from app.rag.answer import answer_question, detect_intent, get_llm

app = FastAPI(
    title="Git RAG API",
    description="An API for retrieving and answering questions about Git using RAG.",
    version="0.1.0"
)

# --------- Models ---------

class ChatRequest(BaseModel):
    question: str
    top_k: int | None = None


class Source(BaseModel):
    file: str
    snippet: str


class ChatResponse(BaseModel):
    answer: str
    intent: str
    sources: List[Source]


# --------- Global state ---------

vectorstore = None
llm = None


# --------- Lifecycle ---------

@app.on_event("startup")
def startup_event():
    global vectorstore, llm

    print("Loading vectorstore...")
    vectorstore = load_vectorstore()

    print("Loading LLM...")
    llm = get_llm()

    print("Startup completed.")


# --------- Routes ---------

@app.get("/")
def root():
    return {
        "message": "Welcome to Git RAG API",
        "version": "0.1.0",
        "docs": "/docs"
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    intent = detect_intent(req.question)

    # Adaptive top-k based on intent
    if req.top_k:
        k = req.top_k
    else:
        k = 6 if intent == "comparison" else 4

    docs = retrieve_documents(
        query=req.question,
        k=k,
        vectorstore=vectorstore
    )

    answer = answer_question(
        question=req.question,
        documents=docs,
        llm=llm
    )

    sources = []
    for doc in docs:
        snippet = "\n".join(doc.page_content.splitlines()[:6])

        sources.append({
            "file": doc.metadata.get("source", "unknown"),
            "snippet": snippet
        })

    return {
        "answer": answer.strip(),
        "intent": intent,
        "sources": sources
    }
