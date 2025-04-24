from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# === Ensemble Retriever Creation ===

# Factory function to create an EnsembleRetriever.
# This retriever combines results from a dense vector retriever (FAISS)
# and a sparse keyword retriever (BM25) for potentially better recall.
def create_ensemble_retriever(documents, embeddings, faiss_weight=0.7, bm25_weight=0.3, k=5):
    """Creates an EnsembleRetriever combining FAISS and BM25."""
    if not documents:
        raise ValueError("Cannot create retriever with empty documents list.")

    # Initialize FAISS vector store and retriever
    vectorstore = FAISS.from_documents(documents, embeddings)
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    # Initialize BM25 retriever
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = k

    # Initialize Ensemble Retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever],
        weights=[faiss_weight, bm25_weight]
    )
    print("Ensemble (Hybrid) retriever created.")
    return ensemble_retriever

# === Document Filtering (Simple Example) ===

# (DEPRECATED/Example) - A very basic filter function based on keyword matching.
# This is likely not used in the final LangGraph flow and is provided as a simple illustration.
# More sophisticated filtering would typically use semantic similarity scores from the retriever.
def filter_relevant_documents(query, docs, threshold=0.7):
    """
    A simple relevancy filter that checks if the query keyword exists in the document text.
    In practice, replace this with a more sophisticated scoring (e.g., semantic similarity score).
    """
    filtered_docs = []
    query_lower = query.lower()
    for doc in docs:
        # Simple keyword check (can be improved with embedding similarity)
        score = 1.0 if query_lower in doc.page_content.lower() else 0.0 # Using 0.0 for non-matches
        if score >= threshold:
            filtered_docs.append(doc)
    return filtered_docs 