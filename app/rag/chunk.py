# Semantic-aware chunking for Git documentation (refactored)
import re
from typing import List
from langchain_core.documents import Document
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from .ingest import load_documents

# -----------------------------
# Configuration
# -----------------------------
MAX_CHUNK_SIZE = 1000
MIN_CHUNK_SIZE = 200

# Matches:
# - == SECTION
# - NAME\n----
SECTION_PATTERN = re.compile(
    r"^(==+\s+.*|\w[\w\s-]+\n[-~]{3,})",
    re.MULTILINE
)

# -----------------------------
# Helpers
# -----------------------------
def detect_doc_topic(source: str) -> str:
    """
    Infer high-level topic from file path.
    """
    name = source.lower()

    if name.startswith("documentation/git-"):
        return "command"
    if "technical" in name:
        return "technical"
    return "general"


def extract_section_title(raw_header: str) -> str:
    """
    Normalize section title from AsciiDoc headers.
    """
    raw_header = raw_header.strip()

    if raw_header.startswith("="):
        return raw_header.lstrip("=").strip()

    # Handle:
    # NAME
    # ----
    return raw_header.splitlines()[0].strip()


def split_by_paragraph(text: str, max_size: int) -> List[str]:
    """
    Split text by paragraphs while preserving semantic boundaries.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks: List[str] = []
    current = ""

    for p in paragraphs:
        if len(current) + len(p) <= max_size:
            current = f"{current}\n\n{p}" if current else p
        else:
            chunks.append(current)
            current = p

    if current:
        chunks.append(current)

    return chunks


# -----------------------------
# Main chunking logic
# -----------------------------
def chunk_document(doc: Document) -> List[Document]:
    text = doc.page_content
    matches = list(SECTION_PATTERN.finditer(text))

    source = doc.metadata.get("source", "unknown")
    topic = detect_doc_topic(source)
    subject = doc.metadata.get("file_name", "").replace(".adoc", "")

    # No sections found → treat entire document as one section
    if not matches:
        return build_chunks(
            section_title="Document",
            section_text=text,
            doc=doc,
            topic=topic,
            subject=subject,
            section_index=1,
        )

    chunks: List[Document] = []

    for i, match in enumerate(matches):
        header_text = match.group(0)
        section_title = extract_section_title(header_text)

        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()

        section_chunks = build_chunks(
            section_title=section_title,
            section_text=section_text,
            doc=doc,
            topic=topic,
            subject=subject,
            section_index=i + 1,
        )

        chunks.extend(section_chunks)

    return chunks


def build_chunks(
    section_title: str,
    section_text: str,
    doc: Document,
    topic: str,
    subject: str,
    section_index: int,
) -> List[Document]:

    sub_texts = (
        [section_text]
        if len(section_text) <= MAX_CHUNK_SIZE
        else split_by_paragraph(section_text, MAX_CHUNK_SIZE)
    )

    results: List[Document] = []

    for idx, text_part in enumerate(sub_texts, start=1):
        if len(text_part.strip()) < MIN_CHUNK_SIZE:
            continue

        header = (
            f"Git documentation\n"
            f"Subject: {subject}\n"
            f"Topic: {topic}\n"
            f"Section: {section_title}\n\n"
        )

        results.append(
            Document(
                page_content=header + text_part.strip(),
                metadata={
                    **doc.metadata,
                    "topic": topic,
                    "subject": subject,
                    "section_title": section_title,
                    "section_index": section_index,
                    "sub_chunk_index": idx,
                },
            )
        )

    return results


def chunk_documents(documents: List[Document]) -> List[Document]:
    all_chunks: List[Document] = []

    for doc in documents:
        all_chunks.extend(chunk_document(doc))

    return all_chunks


# -----------------------------
# Standalone execution
# -----------------------------
if __name__ == "__main__":
    print("Loading documents...")
    docs = load_documents()
    print(f"Documents loaded: {len(docs)}")

    print("\nChunking documents...")
    chunks = chunk_documents(docs)
    print(f"Total chunks created: {len(chunks)}")

    print("\nChunk preview:")
    for i, chunk in enumerate(chunks[:5], start=1):
        print("=" * 80)
        print(f"Chunk {i}")
        print(f"Source: {chunk.metadata.get('source')}")
        print(f"Subject: {chunk.metadata.get('subject')}")
        print(f"Topic: {chunk.metadata.get('topic')}")
        print(f"Section: {chunk.metadata.get('section_title')}")
        print(f"Size: {len(chunk.page_content)} characters\n")
        print(chunk.page_content[:500], "...")
