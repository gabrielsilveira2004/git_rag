# Testing ingest.py
from pathlib import Path
import sys

# Adicionar o diretório pai ao sys.path para importar módulos de app
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.rag.ingest import ingest_user_documentation, load_documents, clone_repo


def test_clone_repo():
    # Teste se a função clone_repo não lança erro
    try:
        clone_repo()
        print("Clone repo test passed")
    except Exception as e:
        print(f"Clone repo test failed: {e}")

def test_load_documents():
    # Teste se load_documents retorna uma lista
    try:
        docs = load_documents()
        assert isinstance(docs, list)
        print(f"Load documents test passed. Total documents: {len(docs)}")
        
        # Mostrar até 3 exemplos
        for i in range(min(3, len(docs))):
            doc = docs[i]
            print(f"\nExemple {i+1}:")
            print(f"  Source: {doc.metadata.get('source', 'N/A')}")
            print(f"  Content (first 200 chars): {doc.page_content[:200]}...")
    except Exception as e:
        print(f"Load documents test failed: {e}")

if __name__ == "__main__":
    test_clone_repo()
    test_load_documents()

 
