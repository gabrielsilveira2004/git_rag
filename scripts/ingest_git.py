# Complete pipeline for ingestion, chunking, indexing and testing of Git RAG
from pathlib import Path
import sys

# Add the root directory to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.rag.ingest import ingest_user_documentation
from app.rag.chunk import chunk_documents
from app.rag.vectorstore import build_vectorstore, save_vectorstore
from app.rag.retrieve import retrieve_documents
from app.rag.answer import answer_question

def main():
    print("Starting complete Git RAG pipeline...")

    # 1. Ingestion: Clone repo and load documents
    print("\nStep 1: Document ingestion...")
    documents = ingest_user_documentation()
    print(f"Loaded {len(documents)} documents.")

    # 2. Chunking: Split documents into chunks
    print("\nStep 2: Document chunking...")
    chunks = chunk_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    # 3. Vector Store: Build and save vector store
    print("\nStep 3: Building vector store...")
    vectorstore = build_vectorstore(chunks)
    save_vectorstore(vectorstore)
    print("Vector store built and saved.")

    # 4. Test Retrieve and Answer
    print("\nStep 4: Testing retrieve and answer...")

    test_questions = [
        "How does git commit work?",
        "What is the difference between git merge and git rebase?"
    ]

    for question in test_questions:
        print(f"\nQuestion: {question}")

        # Retrieve
        retrieved_docs = retrieve_documents(question, k=4)
        print(f"Retrieved documents: {len(retrieved_docs)}")

        # Answer
        answer = answer_question(question, retrieved_docs)
        print(f"Answer: {answer[:200]}...")

    print("\nPipeline complete! The system is ready and tested.")
    print("Run 'uvicorn app.api.main:app --reload' to start the API.")

if __name__ == "__main__":
    main()
