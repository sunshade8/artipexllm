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

# 질문 준비
question = "색상이 감정에 미치는 영향에 대해 설명해주세요"
conversation_history = []

# 실행
result = adaptive_rag.invoke(
    {
        "question": question,
        "conversation_history": conversation_history
    }, 
    config=config
)

# 결과 출력
print(f"질문: {question}")
print(f"답변: {result['generation']}")
