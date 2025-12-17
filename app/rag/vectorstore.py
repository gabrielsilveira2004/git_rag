# Vectorstore builder and loader for Git RAG
from pathlib import Path
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from .ingest import ingest_user_documentation
from .chunk import chunk_documents

# -----------------------------
# Paths & config
# -----------------------------
BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "data"
VECTORSTORE_DIR = DATA_DIR / "vectorstore"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# -----------------------------
# Embedding model
# -----------------------------
def get_embedding_model():
    """
    Returns a sentence-transformer embedding model.
    """
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


# -----------------------------
# Build / Save / Load
# -----------------------------
def build_vectorstore(chunks: List[Document]) -> FAISS:
    """
    Builds a FAISS vectorstore from chunked documents.
    """
    if not chunks:
        raise ValueError("No chunks provided to build the vectorstore.")

    embedding_model = get_embedding_model()

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embedding_model,
    )

    return vectorstore


def save_vectorstore(vectorstore: FAISS):
    """
    Saves the vectorstore to disk.
    """
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(VECTORSTORE_DIR))

    print(f"✅ Vectorstore saved to: {VECTORSTORE_DIR}")
    print(f"📦 Total vectors indexed: {vectorstore.index.ntotal}")


def load_vectorstore() -> FAISS:
    """
    Loads the vectorstore from disk.
    """
    if not VECTORSTORE_DIR.exists():
        raise FileNotFoundError(
            f"Vectorstore directory not found: {VECTORSTORE_DIR}"
        )

    embedding_model = get_embedding_model()

    vectorstore = FAISS.load_local(
        str(VECTORSTORE_DIR),
        embedding_model,
        allow_dangerous_deserialization=True,
    )

    return vectorstore


# -----------------------------
# Manual test runner
# -----------------------------
if __name__ == "__main__":
    print("\n🔹 Ingesting documentation...")
    documents = ingest_user_documentation()

    print("\n🔹 Chunking documents...")
    chunks = chunk_documents(documents)

    print(f"Documents: {len(documents)}")
    print(f"Chunks: {len(chunks)}")

    print("\n🔹 Building vectorstore...")
    vectorstore = build_vectorstore(chunks)

    print("\n🔹 Saving vectorstore...")
    save_vectorstore(vectorstore)

    print("\n🔹 Reloading vectorstore...")
    reloaded_vs = load_vectorstore()
    print(f"Reloaded vectors: {reloaded_vs.index.ntotal}")

    # -----------------------------
    # Semantic sanity checks
    # -----------------------------
    test_queries = [
        "How does git checkout-index work?",
        "What is the difference between git merge and git rebase?",
        "What does git blame --reverse do?",
    ]

    for query in test_queries:
        print("\n" + "=" * 80)
        print(f"Query: {query}")

        results = reloaded_vs.similarity_search(query, k=3)

        for i, doc in enumerate(results, start=1):
            print(f"\nResult {i}")
            print(f"Source: {doc.metadata.get('source')}")
            print(f"Topic: {doc.metadata.get('topic')}")
            print(f"Section: {doc.metadata.get('section_title')}")
            print("-" * 40)
            print(doc.page_content[:400], "...")