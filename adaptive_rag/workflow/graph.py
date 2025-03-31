from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from ..models import GraphState
from ..handlers.router import route_question
from ..handlers.memory_handler import memory_query_handler
from ..retrievers.web_search import web_search
from ..retrievers.vector_store import retrieve_documents
from ..evaluators.document_evaluator import grade_documents, decide_to_generate
from ..evaluators.hallucination_checker import check_hallucination
from ..generators.answer_generator import generate_answer, rewrite_question

def create_workflow():
    """Adaptive RAG 워크플로우를 생성하고 컴파일합니다."""
    from langgraph.graph import END, StateGraph, START
    from langgraph.checkpoint.memory import MemorySaver
    
    
    # 그래프 초기화
    workflow = StateGraph(GraphState)
    
    # 노드 추가
    workflow.add_node("memory_handler", memory_query_handler)  # 메모리 노드 추가
    workflow.add_node("web_search", web_search)
    workflow.add_node("retrieve_documents", retrieve_documents)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate_answer", generate_answer)
    workflow.add_node("rewrite_question", rewrite_question)
    
    # 엣지 설정
    workflow.add_conditional_edges(
        START,
        route_question,
        {
            "memory": "memory_handler",  # 메모리 관련 질문
            "web_search": "web_search",
            "vectorstore": "retrieve_documents"
        }
    )
    
    # 메모리 핸들러 결과 처리
    workflow.add_edge("memory_handler", END)
    
    workflow.add_edge("web_search", "generate_answer")
    workflow.add_edge("retrieve_documents", "grade_documents")
    
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "rewrite": "rewrite_question",
            "generate": "generate_answer"
        }
    )
    
    workflow.add_edge("rewrite_question", "retrieve_documents")
    
    workflow.add_conditional_edges(
        "generate_answer",
        check_hallucination,
        {
            "hallucination": "generate_answer",
            "relevant": END,
            "not_relevant": "rewrite_question"
        }
    )
    
    # 그래프 컴파일
    adaptive_rag = workflow.compile(checkpointer=MemorySaver())

    return adaptive_rag
