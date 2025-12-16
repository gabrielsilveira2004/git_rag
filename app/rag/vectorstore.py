# Indexing chunks into a vector store for retrieval
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Data directory and vector store path
BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "data"
VECTORSTORE_DIR = DATA_DIR / "vectorstore"

# Embedding model name
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Gets the embedding model
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

# Builds the vector store from document chunks
def build_vectorstore(chunks: list[Document]) -> FAISS:
    embedding_model = get_embedding_model()
    
    vectorstore = FAISS.from_documents(documents=chunks,
                                       embedding=embedding_model)
    return vectorstore

# Saves the vector store to disk
def save_vectorstore(vectorstore: FAISS):
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(VECTORSTORE_DIR))
    print(f"Vector store saved to {VECTORSTORE_DIR}")
    
# Loads the vector store from disk
def load_vectorstore() -> FAISS:
    if not VECTORSTORE_DIR.exists():
        raise FileNotFoundError(f"Vector store directory {VECTORSTORE_DIR} does not exist.")
    
    embedding_model = get_embedding_model()
    vectorstore = FAISS.load_local(str(VECTORSTORE_DIR), embedding_model, allow_dangerous_deserialization=True)

    return vectorstore


