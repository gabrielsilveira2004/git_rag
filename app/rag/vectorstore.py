from pathlib import Path
from typing import List, Tuple

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from .ingest import ingest_user_documentation
from .chunk import chunk_documents

BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "data"
VECTORSTORE_DIR = DATA_DIR / "vectorstore"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# Load the sentence-transformer embedding model
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


# Build a FAISS vectorstore from document chunks
def build_vectorstore(chunks: List[Document]) -> FAISS:
    if not chunks:
        raise ValueError("No chunks provided")

    embeddings = get_embedding_model()
    return FAISS.from_documents(chunks, embeddings)


# Save the vectorstore locally
def save_vectorstore(vectorstore: FAISS):
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(VECTORSTORE_DIR))

    print(f"Vectorstore saved at: {VECTORSTORE_DIR}")
    print(f"Total vectors: {vectorstore.index.ntotal}")


# Load the vectorstore from disk
def load_vectorstore() -> FAISS:
    embeddings = get_embedding_model()
    return FAISS.load_local(
        str(VECTORSTORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )


# Run basic semantic tests against the vectorstore
def run_semantic_tests(vectorstore: FAISS, queries: List[str]):
    for query in queries:
        print("\n" + "=" * 80)
        print(f"Query: {query}")

        results: List[Tuple[Document, float]] = (
            vectorstore.similarity_search_with_score(query, k=6)
        )

        for doc, score in results:
            if score > 0.45:
                continue

            print("\nScore:", round(score, 4))
            print("Source:", doc.metadata.get("source"))
            print("Topic:", doc.metadata.get("topic"))
            print("Section:", doc.metadata.get("section_title"))
            print("-" * 40)
            print(doc.page_content[:400], "...")


if __name__ == "__main__":
    print("\nIngesting documentation...")
    documents = ingest_user_documentation()

    print("Chunking documents...")
    chunks = chunk_documents(documents)

    print(f"Documents: {len(documents)}")
    print(f"Chunks: {len(chunks)}")

    print("Building vectorstore...")
    vectorstore = build_vectorstore(chunks)

    print("Saving vectorstore...")
    save_vectorstore(vectorstore)

    print("Reloading vectorstore...")
    reloaded_vs = load_vectorstore()

    test_queries = [
        "How does git checkout-index work?",
        "What is the difference between git merge and git rebase?",
        "What does git blame --reverse do?",
    ]

    run_semantic_tests(reloaded_vs, test_queries)
