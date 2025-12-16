from pathlib import Path
import sys

# Add the parent directory to sys.path to import modules from app
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.rag.ingest import load_documents
from app.rag.chunk import chunk_documents
from app.rag.vectorstore import (
    build_vectorstore,
    save_vectorstore,
    load_vectorstore,
    VECTORSTORE_DIR
)

def test_build_vectorstore():
    try:
        # Load and chunk documents
        docs = load_documents()
        print(f"Documents loaded: {len(docs)}")

        chunks = chunk_documents(docs)
        print(f"Chunks created: {len(chunks)}")

        # Build vectorstore
        vectorstore = build_vectorstore(chunks)
        print("Vectorstore built successfully.")

        # Basic sanity check
        print(f"Total vectors indexed: {vectorstore.index.ntotal}")

        # Example semantic search
        query = "How does git checkout-index work?"
        results = vectorstore.similarity_search(query, k=3)

        print(f"\nQuery: {query}")
        print(f"Results returned: {len(results)}")

        for i, doc in enumerate(results, start=1):
            print(f"\nResult {i}:")
            print(f"Source: {doc.metadata.get('source')}")
            print(doc.page_content[:300], "...")

        print("\nVectorstore build test passed!")

    except Exception as e:
        print(f"Vectorstore build test failed: {e}")

def test_save_and_load_vectorstore():
    try:
        # Load and chunk a small subset (speed)
        docs = load_documents()[:10]
        chunks = chunk_documents(docs)

        vectorstore = build_vectorstore(chunks)

        # Save vectorstore
        save_vectorstore(vectorstore)
        print(f"Vectorstore saved at: {VECTORSTORE_DIR}")

        # Load vectorstore
        loaded_vectorstore = load_vectorstore()
        print("Vectorstore loaded successfully.")

        print(f"Total vectors after load: {loaded_vectorstore.index.ntotal}")

        # Test query on loaded vectorstore
        query = "What is Git?"
        results = loaded_vectorstore.similarity_search(query, k=2)

        print(f"\nQuery after load: {query}")
        for i, doc in enumerate(results, start=1):
            print(f"\nResult {i}:")
            print(f"Source: {doc.metadata.get('source')}")
            print(doc.page_content[:300], "...")

        print("\nSave/load vectorstore test passed!")

    except Exception as e:
        print(f"Save/load vectorstore test failed: {e}")

if __name__ == "__main__":
    test_build_vectorstore()
    test_save_and_load_vectorstore()
