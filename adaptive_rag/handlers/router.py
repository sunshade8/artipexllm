from ..models import GraphState
from .memory_handler import memory_query_handler

# 질문 라우팅 노드 (개선됨)
def route_question(state: GraphState):
    print("==== [질문 라우팅 중] ====")
    question = state["question"]
    conversation_history = state.get("conversation_history", [])
    
    # 메모리 관련 질문 체크 (더 강력한 우선순위)
    memory_result = memory_query_handler(state)
    if memory_result is not None:
        print("메모리 관련 질문 감지됨, 이전 대화 기록 반환")
        return "memory"
    
    # 이전 대화 내용을 참조하는 경우 (예: "더 설명해줘", "그건 어떻게?") 
    # 벡터 저장소 검색이 더 적합할 수 있음
    follow_up_phrases = ["더", "그럼", "그건", "어떻게", "왜", "언제", "누가", "무엇을", "어디서"]
    if conversation_history and any(phrase in question for phrase in follow_up_phrases):
        print("후속 질문 감지됨, 벡터 저장소로 라우팅")
        return "vectorstore"
    
    # 간단한 키워드 기반 라우팅
    # 최신 정보, 뉴스, 통계, 날짜 관련 키워드는 웹 검색으로
    web_search_keywords = ["최신", "뉴스", "통계", "언제", "최근", "오늘", "어제", "날짜", 
                           "2023", "2024", "현재", "요즘", "트렌드", "동향"]
    
    if any(keyword in question for keyword in web_search_keywords):
        print("웹 검색으로 라우팅됨")
        return "web_search"
    else:
        print("벡터 저장소로 라우팅됨")
        return "vectorstore"

