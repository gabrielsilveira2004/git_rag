# Transform Git user docummentation into a RAG-ingestible and processable format
from pathlib import Path
from git import Repo
from langchain_core.documents import Document

BASE_DIR = Path.cwd() # Current working directory
DATA_DIR = BASE_DIR / "data" # Data directory
REPO_DIR = DATA_DIR / "git_repo" # Git repository directory
DOCS_DIR = REPO_DIR / "Documentation" # Documentation directory within the repo
GIT_REPO_URL = "https://github.com/git/git.git" # Git repository URL
DOC_EXTENSIONS = {".md", ".txt", ".adoc"} # Supported documentation file extensions

def get_git_commit_hash():
    repo = Repo(REPO_DIR)
    return repo.head.object.hexsha

# MAIN FUNCTIONS
def clone_repo(): # Clone the Git repository if not already cloned
    if REPO_DIR.exists() and (REPO_DIR / '.git').exists():
        print("Repository already exists.")
    else:
        print("Cloning Git repository...")
        Repo.clone_from(GIT_REPO_URL, REPO_DIR)
        print("Repository cloned.")
    
def load_documents(): # Load and transform user documentation files into Document objects
    if not DOCS_DIR.exists():
        raise RuntimeError(f"Documentation directory {DOCS_DIR} does not exist.")
    
    files = [
        f for f in DOCS_DIR.rglob("*")
        if f.suffix.lower() in DOC_EXTENSIONS
    ]
    
    print(f"Found {len(files)} documentation files.")
    
    documents = []
    
    for file_path in files:
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": str(file_path.relative_to(REPO_DIR)),
                        "file_name": file_path.name,
                        "file_path": str(file_path),
                        "type": "user_documentation",
                        "git_commit": get_git_commit_hash(),
                    },
                )
            )
        
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
        
    return documents

def ingest_user_documentation(): # Main function to clone repo and load documents
    clone_repo()
    documents = load_documents()
    print(f"Loaded {len(documents)} documentation files.")
    return documents

# PREVIEW FUNCTION
def preview_documents(documents, n=3, preview_chars=300):
    print(f"\nPreviewing {min(n, len(documents))} documents:\n")

    for i, doc in enumerate(documents[:n], start=1):
        content = doc.page_content.strip()

        print(f"[{i}] Source: {doc.metadata.get('source')}")
        print(f"    File: {doc.metadata.get('file_name')}")
        print(f"    Size: {len(content)} characters")
        print(f"    Git commit: {doc.metadata.get('git_commit')}")
        print("    Content preview:\n")

        snippet = content[:preview_chars]
        snippet = snippet.replace("\n", " ").strip()

        print(f"    {snippet}...")
        print("-" * 80)

# if __name__ == "__main__":
   # ingest_user_documentation()
    #documents = ingest_user_documentation()
    #preview_documents(documents=documents, n=5)

    #documents = ingest_user_documentation()
    #chunked_docs = chunk_documents(documents)
    #print(f"Total chunked documents: {len(chunked_docs)}")
    #print(f"Chunk preview:")
    #for c in chunked_docs[:3]:
        #print(f"Source: {c.metadata.get('source')}, Section: {c.metadata.get('section_title')}, Size: {len(c.page_content)} characters")