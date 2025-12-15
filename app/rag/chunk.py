# Split the documents into smaller chunks for processing
import re
from langchain_core.documents import Document

SECTION_PATTERN = re.compile(r"^==+\s+(.*)", re.MULTILINE) # Pattern to identify section headers

# Split an AsciiDoc document into sections based on headers (==, ===, etc.)
def chunk_documents_by_section(doc: Document) -> list[Document]:
    text = doc.page_content
    matches = list(SECTION_PATTERN.finditer(text))
    
    if not matches:
        return [doc]  # No sections found, return the original document
    
    chunks = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        
        section_title = match.group(1).strip()
        section_content = text[start:end].strip()
        section_text = section_content[len(match.group(0)):].strip()  # Exclude the header line
        
        chunks.append(
            Document(
                page_content=section_content,
                metadata={
                    **doc.metadata,
                    "section_title": section_title,
                    "section_index": i + 1,
                    "section_text": section_text,
                },
            )
        )
            
    return chunks

def chunk_documents(documents: list[Document]) -> list[Document]:
    all_chunks = []
    for doc in documents:
        chunks = chunk_documents_by_section(doc)
        all_chunks.extend(chunks)
        
    return all_chunks