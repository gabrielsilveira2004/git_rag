from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.rag.ingest import load_documents
from app.rag.chunk import chunk_documents
from app.rag.vectorstore import build_vectorstore, get_embedding_model
from langchain_community.vectorstores import FAISS

TEST_VECTORSTORE_DIR = Path("data/vectorstore_test")


def test_vectorstore_build_and_search():
    # Cleanup
    if TEST_VECTORSTORE_DIR.exists():
        shutil.rmtree(TEST_VECTORSTORE_DIR)

    # Load small subset
    docs = load_documents()[:10]
    chunks = chunk_documents(docs)

    vectorstore = build_vectorstore(chunks)

    # Save test index
    TEST_VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(TEST_VECTORSTORE_DIR))

    # Reload test index
    embedding_model = get_embedding_model()
    loaded_vs = FAISS.load_local(
        str(TEST_VECTORSTORE_DIR),
        embedding_model,
        allow_dangerous_deserialization=True
    )

    assert loaded_vs.index.ntotal > 0

    queries = [
        "What is Git?",
        "How does git checkout-index work?",
        "What is the difference between git merge and git rebase?",
    ]

    for query in queries:
        results = loaded_vs.similarity_search(query, k=3)

        print("\n" + "=" * 60)
        print(f"Query: {query}")
        print(f"Results: {len(results)}")

        sources = set()

        for i, doc in enumerate(results, 1):
            source = doc.metadata.get("source")
            sources.add(source)

            print(f"\nResult {i}")
            print(f"Source: {source}")
            print(doc.page_content[:200], "...")

        print(f"\nUnique sources returned: {len(sources)}")

    print("\n✅ Vectorstore TEST passed")

if __name__ == "__main__":
    test_vectorstore_build_and_search()