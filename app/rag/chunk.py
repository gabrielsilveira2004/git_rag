# Semantic chunking for Git documentation
# Responsibility: size control + clean content for embeddings

from typing import List
from langchain_core.documents import Document

MAX_CHUNK_SIZE = 800
MIN_CHUNK_SIZE = 120


def split_by_paragraph(text: str, max_size: int) -> List[str]:
    """
    Split text into chunks using paragraph boundaries.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks: List[str] = []
    current = ""

    for p in paragraphs:
        if len(current) + len(p) <= max_size:
            current = f"{current}\n\n{p}" if current else p
        else:
            if current:
                chunks.append(current.strip())
            current = p

    if current:
        chunks.append(current.strip())

    return chunks


def chunk_document(doc: Document) -> List[Document]:
    """
    Chunk a single semantic document into embedding-friendly chunks.
    """
    text = doc.page_content.strip()
    if not text:
        return []

    parts = split_by_paragraph(text, MAX_CHUNK_SIZE)
    results: List[Document] = []

    for idx, part in enumerate(parts, start=1):
        if len(part) < MIN_CHUNK_SIZE:
            continue

        results.append(
            Document(
                page_content=part,
                metadata={
                    **doc.metadata,
                    "chunk_index": idx,
                },
            )
        )

    return results


def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Chunk all documents produced by ingest step.
    """
    all_chunks: List[Document] = []

    for doc in documents:
        all_chunks.extend(chunk_document(doc))

    return all_chunks


# -----------------------------
# Manual test
# -----------------------------
if __name__ == "__main__":
    from ingest import ingest_git_documentation

    docs = ingest_git_documentation()
    chunks = chunk_documents(docs)

    print(f"Total chunks: {len(chunks)}\n")

    for c in chunks[:5]:
        print("=" * 80)
        print(c.page_content[:400])
        print("Metadata:", c.metadata)
