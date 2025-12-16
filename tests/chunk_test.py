from pathlib import Path
import sys

# Add the parent directory to sys.path to import modules from app
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.rag.ingest import load_documents
from app.rag.chunk import chunk_documents, chunk_documents_by_section

def test_chunk_documents():
    try:
        # Load documents
        docs = load_documents()
        print(f"Documents loaded: {len(docs)}")
        
        # Perform chunking
        chunks = chunk_documents(docs)
        print(f"Chunks created: {len(chunks)}")
        
        # Show up to 3 examples of chunks
        for i in range(min(3, len(chunks))):
            chunk = chunks[i]
            print(f"\nChunk Example {i+1}:")
            print(f"  Section title: {chunk.metadata.get('section_title', 'N/A')}")
            print(f"  Section index: {chunk.metadata.get('section_index', 'N/A')}")
            print(f"  Content (first 200 characters): {chunk.page_content[:200]}...")
        
        print("\nChunking test passed!")
    except Exception as e:
        print(f"Chunking test failed: {e}")

def test_chunk_single_document():
    try:
        docs = load_documents()
        if not docs:
            print("No documents to test individual chunking.")
            return
        
        # Test chunking on a single document
        single_doc = docs[0]
        chunks = chunk_documents_by_section(single_doc)
        print(f"Single document chunked into {len(chunks)} parts.")
        
        if chunks:
            print(f"First chunk: {chunks[0].metadata.get('section_title', 'N/A')}")
        
        print("Individual chunking test passed!")
    except Exception as e:
        print(f"Individual chunking test failed: {e}")

if __name__ == "__main__":
    test_chunk_documents()
    test_chunk_single_document()