import os
import chromadb
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from app.config import settings
from app.services.chunking import chunk_pdf
from app.services.retrieval import AdaptiveCutoff, CrossEncoderReranker, HybridRetriever

COLLECTION_NAME = "risk_auditor_docs"

# Using local HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)

vector_store_instance = None
retriever_instance = None

def get_chroma_client():
    # Retrieve Chroma configuration from environment with fallbacks for local execution
    chroma_host = os.environ.get("CHROMA_HOST", "localhost")
    chroma_port = int(os.environ.get("CHROMA_PORT", "8001"))
    # Connect directly to the isolated Docker microservice via HTTP
    return chromadb.HttpClient(host=chroma_host, port=chroma_port)

def process_pdf(file_path: str):
    global vector_store_instance
    # Section-aware, token-sized chunks with 1-based page numbers and deterministic ids (see chunking.py);
    # documents without recognisable headings fall back to page-by-page chunks
    splits, ids = chunk_pdf(file_path, settings.EMBEDDING_MODEL)

    # Send the chunked documents into the standalone server via HttpClient
    client = get_chroma_client()
    store = Chroma(client=client, collection_name=COLLECTION_NAME, embedding_function=embeddings)
    # Drop this file's previous chunks first, so a re-upload (or a new version under the same name)
    # replaces them instead of leaving duplicates or stale text behind
    store.delete(where={"source": file_path})
    store.add_documents(splits, ids=ids)
    vector_store_instance = store
    return vector_store_instance

def get_vector_store():
    global vector_store_instance
    if vector_store_instance is None:
        try:
            client = get_chroma_client()
            db = Chroma(client=client, collection_name=COLLECTION_NAME, embedding_function=embeddings)
            
            # Use HTTP Client to fetch remote collection data to verify if it's empty
            collection = client.get_or_create_collection(COLLECTION_NAME)
            count = collection.count()
            if count > 0:
                vector_store_instance = db
                print(f"📦 [Microservice Log]: Successfully bridged to the standalone Chroma container via HTTP, mounting {count} memory chunks!")
            else:
                vector_store_instance = None
        except Exception as e:
            # Fail silently here, meaning the service contains no data or is offline
            vector_store_instance = None
                
    return vector_store_instance


def get_retriever():
    """Hybrid retriever over the current vector store, or None when no document has been ingested."""
    global retriever_instance
    store = get_vector_store()
    if store is None:
        return None
    # A new upload replaces the store object, which also discards the old in-memory BM25 index
    if retriever_instance is None or retriever_instance.store is not store:
        reranker = CrossEncoderReranker(settings.RERANK_MODEL) if settings.RERANK_MODEL else None
        cutoff = AdaptiveCutoff(
            min_score=settings.RERANK_MIN_SCORE,
            margin=settings.RERANK_MARGIN,
            min_k=settings.RETRIEVAL_MIN_K,
        ) if settings.ADAPTIVE_TOP_K else None
        retriever_instance = HybridRetriever(
            store,
            reranker=reranker,
            candidates=settings.RETRIEVAL_CANDIDATES,
            use_bm25=settings.USE_BM25,
            cutoff=cutoff,
        )
    return retriever_instance
