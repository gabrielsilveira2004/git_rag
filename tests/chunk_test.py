from pathlib import Path
import sys

# Add the parent directory to sys.path to import modules from app
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.rag.ingest import load_documents
from app.rag.chunk import chunk_documents, chunk_documents_by_section

MAX_EXPECTED_CHUNK_SIZE = 900  # margem de segurança

def test_chunk_documents():
    try:
        docs = load_documents()
        print(f"Documents loaded: {len(docs)}")

        chunks = chunk_documents(docs)
        print(f"Chunks created: {len(chunks)}")

        # Basic sanity checks
        assert len(chunks) > len(docs), "Chunking did not increase document count"

        # Inspect first few chunks
        for i, chunk in enumerate(chunks[:3], start=1):
            content = chunk.page_content

            print(f"\nChunk Example {i}:")
            print(f"  Source: {chunk.metadata.get('source')}")
            print(f"  Section title: {chunk.metadata.get('section_title', 'N/A')}")
            print(f"  Section index: {chunk.metadata.get('section_index', 'N/A')}")
            print(f"  Sub-chunk index: {chunk.metadata.get('sub_chunk_index', 'N/A')}")
            print(f"  Chunk size: {len(content)} characters")

            # Validate contextual prefix
            assert "Git documentation" in content, "Missing contextual prefix"
            assert "Section:" in content, "Missing section context"

            # Size control
            assert len(content) <= MAX_EXPECTED_CHUNK_SIZE + 200, \
                "Chunk size too large, semantic dilution risk"

            print("  Content preview:")
            print(content[:300], "...")

        print("\nChunking test passed!")

    except Exception as e:
        print(f"Chunking test failed: {e}")

def test_chunk_single_document():
    try:
        docs = load_documents()
        if not docs:
            print("No documents to test individual chunking.")
            return

        single_doc = docs[0]
        chunks = chunk_documents_by_section(single_doc)

        print(f"Single document chunked into {len(chunks)} parts.")

        # Check first chunk deeply
        first_chunk = chunks[0]
        print("\nFirst chunk inspection:")
        print(f"  Source: {first_chunk.metadata.get('source')}")
        print(f"  Section title: {first_chunk.metadata.get('section_title')}")
        print(f"  Sub-chunk index: {first_chunk.metadata.get('sub_chunk_index', 'N/A')}")
        print(f"  Size: {len(first_chunk.page_content)}")

        assert "Git documentation" in first_chunk.page_content
        assert "Section:" in first_chunk.page_content

        print("  Content preview:")
        print(first_chunk.page_content[:300], "...")

        print("\nIndividual document chunking test passed!")

    except Exception as e:
        print(f"Individual chunking test failed: {e}")

if __name__ == "__main__":
    test_chunk_documents()
    test_chunk_single_document()
