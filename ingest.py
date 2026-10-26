"""
ingest.py — RetainIQ document ingestion pipeline
Loads the ChurnSense documents, splits them into chunks, embeds them with
all-MiniLM-L6-v2 and stores them in a local FAISS index.
Run this ONCE before starting the app: python ingest.py
"""

from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ── Config ────────────────────────────────────────────────────────────────────
DOCS_DIR = Path(".")  # the three knowledge-base files sit at the top level of the repo
DOC_FILES = [
    "churnsense_eda_findings.txt",
    "churnsense_shap_model_report.txt",
    "retainiq_retention_playbook.txt",
]
VECTORSTORE_DIR = Path("vectorstore")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# ── Load documents ────────────────────────────────────────────────────────────
def load_documents():
    docs = []
    doc_files = [DOCS_DIR / name for name in DOC_FILES if (DOCS_DIR / name).exists()]
    
    if not doc_files:
        raise FileNotFoundError(f"Knowledge-base files not found: {DOC_FILES}")
    
    for filepath in doc_files:
        loader = TextLoader(str(filepath), encoding="utf-8")
        loaded = loader.load()
        # Tag each doc with its source filename
        for doc in loaded:
            doc.metadata["source"] = filepath.name
        docs.extend(loaded)
        print(f"  Loaded: {filepath.name} ({len(loaded[0].page_content)} chars)")
    
    return docs

# ── Chunk documents ───────────────────────────────────────────────────────────
def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "---", ". ", " "],
    )
    chunks = splitter.split_documents(docs)
    print(f"  Created {len(chunks)} chunks from {len(docs)} documents")
    return chunks

# ── Embed and store ───────────────────────────────────────────────────────────
def build_vectorstore(chunks):
    print(f"  Loading embedding model: {EMBED_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    
    print(f"  Building FAISS index...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    VECTORSTORE_DIR.mkdir(exist_ok=True)
    vectorstore.save_local(str(VECTORSTORE_DIR))
    print(f"  Saved vectorstore to {VECTORSTORE_DIR}/")
    
    return vectorstore

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n=== RetainIQ: document ingestion ===\n")
    
    print("Step 1: Loading documents...")
    docs = load_documents()
    
    print("\nStep 2: Chunking documents...")
    chunks = chunk_documents(docs)
    
    print("\nStep 3: Embedding and storing in FAISS...")
    build_vectorstore(chunks)
    
    print("\n✅ Ingestion complete! Run the app with: streamlit run app.py")
