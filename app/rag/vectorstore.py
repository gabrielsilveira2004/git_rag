# Vectorstore builder and loader for Git RAG
# Focus: clean semantic signal for simple Q&A

from pathlib import Path
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from app.rag.ingest import ingest_git_documentation
from app.rag.chunk import chunk_documents

# -----------------------------
# Paths & config
# -----------------------------
BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "data"
VECTORSTORE_DIR = DATA_DIR / "vectorstore"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Sections that are useful for answering human questions
ALLOWED_SECTIONS = {
    "NAME",
    "DESCRIPTION",
    "INTRO",
    "OPTIONS",
    "EXAMPLES",
}

# Commands to ignore (internal / non-user-facing)
BLACKLIST_COMMAND_KEYWORDS = {
    "hash-object",
    "update-index",
    "fsck",
    "cat-file",
    "mktree",
    "read-tree",
    "write-tree",
}

# -----------------------------
# Embedding model
# -----------------------------
_EMBEDDINGS = None
_VECTORSTORE = None
_VECTORSTORE_LOCK = None

def get_embedding_model():
    global _EMBEDDINGS
    if _EMBEDDINGS is None:
        _EMBEDDINGS = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return _EMBEDDINGS

# -----------------------------
# Corpus cleanup
# -----------------------------
def is_useful_document(doc: Document) -> bool:
    command = doc.metadata.get("command", "").lower()
    section = doc.metadata.get("section", "").upper()

    # Filter internal commands
    for bad in BLACKLIST_COMMAND_KEYWORDS:
        if bad in command:
            return False

    # Keep only meaningful sections
    if section not in ALLOWED_SECTIONS:
        return False

    # Avoid extremely long or extremely short chunks
    size = len(doc.page_content)
    if size < 150 or size > 2000:
        return False

    return True


def enrich_document(doc: Document) -> Document:
    """
    Add explicit semantic context to improve embedding quality.
    """
    command = doc.metadata.get("command", "unknown")
    section = doc.metadata.get("section", "CONTENT")

    header = (
        f"This text is from the Git documentation.\n"
        f"It describes the Git command: {command}.\n"
        f"Section: {section}.\n\n"
    )

    doc.page_content = header + doc.page_content.strip()
    return doc

# -----------------------------
# Build / Save / Load
# -----------------------------
def build_vectorstore(chunks: List[Document]) -> FAISS:
    if not chunks:
        raise ValueError("No chunks provided.")

    embeddings = get_embedding_model()
    return FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
        normalize_L2=True,
    )


def save_vectorstore(vectorstore: FAISS):
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(VECTORSTORE_DIR))

    print(f"Vectorstore saved to: {VECTORSTORE_DIR}")
    print(f"Total vectors indexed: {vectorstore.index.ntotal}")


import threading
def load_vectorstore() -> FAISS:
    """
    Load vectorstore once and cache in memory (singleton pattern).
    Thread-safe for API usage (FastAPI, HF Spaces, etc).
    """
    global _VECTORSTORE, _VECTORSTORE_LOCK
    if _VECTORSTORE_LOCK is None:
        _VECTORSTORE_LOCK = threading.Lock()
    if _VECTORSTORE is not None:
        return _VECTORSTORE
    with _VECTORSTORE_LOCK:
        if _VECTORSTORE is None:
            embeddings = get_embedding_model()
            _VECTORSTORE = FAISS.load_local(
                str(VECTORSTORE_DIR),
                embeddings,
                allow_dangerous_deserialization=True,
            )
    return _VECTORSTORE

# -----------------------------
# Manual test runner
# -----------------------------
def main():
    print("\nIngesting documentation...")
    documents = ingest_git_documentation()

    print("\nChunking documents...")
    chunks = chunk_documents(documents)

    print(f"Raw chunks: {len(chunks)}")

    print("\nFiltering & enriching chunks...")
    clean_chunks: List[Document] = []

    for doc in chunks:
        if is_useful_document(doc):
            clean_chunks.append(enrich_document(doc))

    print(f"Clean chunks: {len(clean_chunks)}")

    print("\nBuilding vectorstore...")
    vectorstore = build_vectorstore(clean_chunks)

    print("\nSaving vectorstore...")
    save_vectorstore(vectorstore)

    print("\nTesting similarity search...")
    test_queries = [
        "What is Git?",
        "How do I create a branch?",
        "How to resolve merge conflicts?",
        "What does git rebase do?",
    ]

    for query in test_queries:
        print("\n" + "=" * 80)
        print(f"Query: {query}")

        results = vectorstore.similarity_search(query, k=5)

        for i, doc in enumerate(results, start=1):
            print(f"\nResult {i}")
            print(f"Command: {doc.metadata.get('command')}")
            print(f"Section: {doc.metadata.get('section')}")
            print("-" * 40)
            print(doc.page_content[:300], "...")

if __name__ == "__main__":
    # Só execute testes locais, nunca em ambiente de API
    main()
