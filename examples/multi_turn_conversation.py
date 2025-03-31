from adaptive_rag import create_adaptive_rag
from langchain_core.runnables import RunnableConfig
import uuid
import os
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# Adaptive RAG 시스템 생성
adaptive_rag = create_adaptive_rag()

# 고유 세션 ID 생성
session_id = str(uuid.uuid4())[:8]
print(f"Session ID: {session_id}")

# 설정 생성
config = RunnableConfig(
    configurable={"thread_id": session_id}
)

# 대화 기록 초기화
conversation_history = []

# 첫 번째 질문
question1 = "색상이 감정에 미치는 영향에 대해 알려줘"
print(f"User: {question1}")

# 실행
result1 = adaptive_rag.invoke(
    {
        "question": question1,
        "conversation_history": conversation_history
    }, 
    config=config
)

print(f"Assistant: {result1['generation']}")
conversation_history = result1["conversation_history"]

# 두 번째 질문 (후속 질문)
question2 = "그럼 파란색에 대해 더 구체적으로 설명해줄래?"
print(f"\nUser: {question2}")

# 실행
result2 = adaptive_rag.invoke(
    {
        "question": question2,
        "conversation_history": conversation_history
    }, 
    config=config
)

print(f"Assistant: {result2['generation']}")
conversation_history = result2["conversation_history"]

# 세 번째 질문 (후속 질문)
question3 = "이런 색상을 활용한 간단한 예술 활동을 추천해줄래?"
print(f"\nUser: {question3}")

# 실행
result3 = adaptive_rag.invoke(
    {
        "question": question3,
        "conversation_history": conversation_history
    }, 
    config=config
)

print(f"Assistant: {result3['generation']}")
