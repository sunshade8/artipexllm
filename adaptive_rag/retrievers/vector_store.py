from ..models import GraphState
from ..config import VECTOR_STORE_PATH, SAMPLE_DATA_DIR, DATA_DIR, PDF_DIR
import os

from ..models import GraphState

# 문서 검색 노드
def retrieve_documents(state: GraphState):
    print("==== [문서 검색 중] ====")
    question = state["question"]
    conversation_history = state.get("conversation_history", [])
    
    # 문서 검색 수행
    documents = ensemble_retriever.invoke(question)
    return {"documents": documents, "conversation_history": conversation_history}

