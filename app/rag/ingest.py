# Load and normalize Git documentation for RAG ingestion
from pathlib import Path
from git import Repo
from langchain_core.documents import Document

# -----------------------------
# Paths and constants
# -----------------------------
BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "data"
REPO_DIR = DATA_DIR / "git_repo"
DOCS_DIR = REPO_DIR / "Documentation"

GIT_REPO_URL = "https://github.com/git/git.git"
DOC_EXTENSIONS = {".md", ".txt", ".adoc"}


# -----------------------------
# Repository helpers
# -----------------------------
def clone_repo() -> None:
    if REPO_DIR.exists() and (REPO_DIR / ".git").exists():
        print("Git repository already present.")
        return

    print("Cloning Git repository...")
    Repo.clone_from(GIT_REPO_URL, REPO_DIR)
    print("Repository cloned successfully.")


def get_repo_commit_hash() -> str:
    repo = Repo(REPO_DIR)
    return repo.head.commit.hexsha


# -----------------------------
# Document loading
# -----------------------------
def load_documents() -> list[Document]:
    if not DOCS_DIR.exists():
        raise RuntimeError(f"Documentation directory not found: {DOCS_DIR}")

    files = [
        f for f in DOCS_DIR.rglob("*")
        if f.suffix.lower() in DOC_EXTENSIONS and f.is_file()
    ]

    print(f"Found {len(files)} documentation files.")

    commit_hash = get_repo_commit_hash()
    documents: list[Document] = []

    for path in files:
        try:
            raw_text = path.read_text(encoding="utf-8", errors="ignore")
            normalized_text = normalize_text(raw_text)

            relative_path = path.relative_to(REPO_DIR)

            documents.append(
                Document(
                    page_content=normalized_text,
                    metadata={
                        "source": str(relative_path),
                        "file_name": path.name,
                        "doc_dir": str(relative_path.parent),
                        "git_commit": commit_hash,
                    },
                )
            )

        except Exception as e:
            print(f"Failed to read {path}: {e}")

    return documents


# -----------------------------
# Text normalization (NOT chunking)
# -----------------------------
def normalize_text(text: str) -> str:
    """
    Basic normalization:
    - normalize line endings
    - trim excessive whitespace
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.splitlines()]
    return "\n".join(lines).strip()


# -----------------------------
# Ingest orchestration
# -----------------------------
def ingest_user_documentation() -> list[Document]:
    clone_repo()
    documents = load_documents()
    print(f"Ingested {len(documents)} documents.")
    return documents


# -----------------------------
# Preview utility (debug only)
# -----------------------------
def preview_documents(documents: list[Document], n: int = 3, chars: int = 300):
    print(f"\nPreviewing {min(n, len(documents))} documents:\n")

    for i, doc in enumerate(documents[:n], start=1):
        content = doc.page_content

        print(f"[{i}] Source: {doc.metadata['source']}")
        print(f"    Directory: {doc.metadata['doc_dir']}")
        print(f"    Size: {len(content)} chars")
        print(f"    Commit: {doc.metadata['git_commit']}")
        print("    Preview:")

        snippet = content[:chars].replace("\n", " ")
        print(f"    {snippet}...")
        print("-" * 80)
      
# COMMENT OUT THE FOLLOWING LINES IF NOT RUNNING DIRECTLY  
if __name__ == "__main__":
    docs = ingest_user_documentation()
    preview_documents(docs)