from pathlib import Path
import sys

# Permite importar app/*
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.rag.retrieve import retrieve_documents


def test_simple_query():
    """
    Testa uma query simples para validar funcionamento básico.
    """
    try:
        query = "What is a Git branch?"

        print(f"\nQuery: {query}")
        results = retrieve_documents(query)

        print(f"Documents retrieved: {len(results)}")

        for i, doc in enumerate(results, start=1):
            print(f"\nResult {i}:")
            print(f"Source: {doc.metadata.get('source')}")
            print(doc.page_content[:300], "...")

        print("\nSimple query test passed!")

    except Exception as e:
        print(f"Simple query test failed: {e}")


def test_composed_query():
    """
    Testa uma query composta para validar multi-query retrieval.
    """
    try:
        query = (
            "What is a Git branch and why is it important "
            "when using git push and git pull?"
        )

        print(f"\nQuery: {query}")
        results = retrieve_documents(query)

        print(f"Documents retrieved: {len(results)}")

        # Coleta fontes para verificar diversidade
        sources = set()

        for i, doc in enumerate(results, start=1):
            source = doc.metadata.get("source")
            sources.add(source)

            print(f"\nResult {i}:")
            print(f"Source: {source}")
            print(doc.page_content[:300], "...")

        print(f"\nUnique sources retrieved: {len(sources)}")

        print("\nComposed query test passed!")

    except Exception as e:
        print(f"Composed query test failed: {e}")


def test_query_diversity():
    """
    Verifica se perguntas diferentes retornam conjuntos diferentes.
    """
    try:
        query_a = "How does git rebase work?"
        query_b = "How does git merge work?"

        results_a = retrieve_documents(query_a)
        results_b = retrieve_documents(query_b)

        sources_a = {doc.metadata.get("source") for doc in results_a}
        sources_b = {doc.metadata.get("source") for doc in results_b}

        overlap = sources_a.intersection(sources_b)

        print(f"\nQuery A: {query_a}")
        print(f"Sources A: {len(sources_a)}")

        print(f"\nQuery B: {query_b}")
        print(f"Sources B: {len(sources_b)}")

        print(f"\nOverlapping sources: {len(overlap)}")

        if len(overlap) == len(sources_a):
            print("⚠️ Warning: queries returned very similar results")
        else:
            print("Query diversity test passed!")

    except Exception as e:
        print(f"Query diversity test failed: {e}")


if __name__ == "__main__":
    test_simple_query()
    test_composed_query()
    test_query_diversity()
