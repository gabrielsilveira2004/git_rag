# Improved ingest for Git documentation (Asciidoc)
# Focus: semantic structure, not chunking yet

from pathlib import Path
from git import Repo
from langchain_core.documents import Document
import re

# -----------------------------
# Paths and constants
# -----------------------------
BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "data"
REPO_DIR = DATA_DIR / "git_repo"
DOCS_DIR = REPO_DIR / "Documentation"

GIT_REPO_URL = "https://github.com/git/git.git"

# Only ingest manpage-like docs
DOC_EXTENSIONS = {".txt", ".adoc"}


# -----------------------------
# Repository helpers
# -----------------------------
def clone_repo() -> None:
    """Clone Git repository if not present."""
    if REPO_DIR.exists() and (REPO_DIR / ".git").exists():
        print("Git repository already present.")
        return

    print("Cloning Git repository...")
    Repo.clone_from(GIT_REPO_URL, REPO_DIR)
    print("Repository cloned successfully.")


def get_repo_commit_hash() -> str:
    """Return current commit hash."""
    repo = Repo(REPO_DIR)
    return repo.head.commit.hexsha


# -----------------------------
# Asciidoc parsing helpers
# -----------------------------
def extract_command_name(text: str, file_name: str) -> str:
    """
    Try to extract command name from asciidoc title.
    Fallback to file name.
    """
    # Example: "= git-pull(1)"
    match = re.search(r"^=\s+(git-[\w-]+)", text, re.MULTILINE)
    if match:
        return match.group(1).replace("-", " ")

    # Fallback: git-pull.txt -> git pull
    return file_name.replace(".txt", "").replace(".adoc", "").replace("-", " ")


def split_into_sections(text: str) -> list[tuple[str, str]]:
    """
    Split asciidoc text into sections.
    Prioritizes main sections (NAME, DESCRIPTION, EXAMPLES, OPTIONS).

    Returns:
        List of (section_title, section_text)
    """
    sections = []
    current_title = "INTRO"
    buffer = []

    for line in text.splitlines():
        if line.startswith("== "):
            # Save previous section
            if buffer:
                section_text = "\n".join(buffer).strip()
                if section_text:
                    sections.append((current_title, section_text))
                buffer = []

            current_title = line.replace("==", "").strip()
        else:
            buffer.append(line)

    # Last section
    if buffer:
        section_text = "\n".join(buffer).strip()
        if section_text:
            sections.append((current_title, section_text))

    return sections


def normalize_text(text: str) -> str:
    """Basic text normalization."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.splitlines()]
    return "\n".join(lines).strip()


# -----------------------------
# Document loading (semantic ingest)
# Priority sections for semantic quality
PRIORITY_SECTIONS = {"DESCRIPTION", "NAME", "OPTIONS", "EXAMPLES", "INTRO"}
MIN_SECTION_SIZE = 100  # Increased from 50 to avoid noise


def load_documents() -> list[Document]:
    if not DOCS_DIR.exists():
        raise RuntimeError(f"Documentation directory not found: {DOCS_DIR}")

    files = [
        f for f in DOCS_DIR.rglob("*")
        if f.suffix.lower() in DOC_EXTENSIONS
        and f.is_file()
        and f.name.startswith("git-")  # avoid auxiliary docs
    ]

    print(f"Found {len(files)} Git documentation files.")

    commit_hash = get_repo_commit_hash()
    documents: list[Document] = []

    for path in files:
        try:
            raw_text = path.read_text(encoding="utf-8", errors="ignore")
            raw_text = normalize_text(raw_text)

            command = extract_command_name(raw_text, path.name)
            sections = split_into_sections(raw_text)

            for section_title, section_text in sections:
                # Filter: minimum size and meaningful content
                if len(section_text) < MIN_SECTION_SIZE:
                    continue

                # Deprioritize very short sections
                if len(section_text) < 200 and section_title not in PRIORITY_SECTIONS:
                    continue

                documents.append(
                    Document(
                        page_content=section_text,
                        metadata={
                            "command": command,
                            "section": section_title,
                            "source": str(path.relative_to(REPO_DIR)),
                            "file_name": path.name,
                            "git_commit": commit_hash,
                            "doc_type": "manpage",
                        },
                    )
                )

        except Exception as e:
            print(f"Failed to read {path}: {e}")

    return documents


# -----------------------------
# Ingest orchestration
# -----------------------------
def ingest_git_documentation() -> list[Document]:
    clone_repo()
    documents = load_documents()
    print(f"Ingested {len(documents)} semantic documents.")
    return documents


# -----------------------------
# Preview utility (test/debug)
# -----------------------------
def preview_documents(
    documents: list[Document],
    n: int = 5,
    chars: int = 300,
):
    print(f"\nPreviewing {min(n, len(documents))} documents:\n")

    for i, doc in enumerate(documents[:n], start=1):
        print(f"[{i}] Command: {doc.metadata['command']}")
        print(f"    Section: {doc.metadata['section']}")
        print(f"    Source: {doc.metadata['source']}")
        print(f"    Size: {len(doc.page_content)} chars")
        print("    Preview:")
        snippet = doc.page_content[:chars].replace("\n", " ")
        print(f"    {snippet}...")
        print("-" * 80)


# -----------------------------
# Manual test
# -----------------------------
if __name__ == "__main__":
    docs = ingest_git_documentation()
    preview_documents(docs, 20, 500)