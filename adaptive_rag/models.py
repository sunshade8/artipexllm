from typing import List, Dict, TypedDict
from langchain_core.documents import Document

# 그래프 상태 정의
class GraphState(TypedDict):
    '''그래프의 상태를 나타내는 데이터 모델'''
    
    question: str  # 사용자 질문
    documents: List[Document]  # 검색된 문서 목록
    generation: str  # 생성된 답변
    conversation_history: List[Dict]  # 이전 대화 기록

