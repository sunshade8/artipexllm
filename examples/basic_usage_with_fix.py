from adaptive_rag import create_adaptive_rag
from langchain_core.runnables import RunnableConfig
import uuid
import os
import sys
from dotenv import load_dotenv
import traceback

# 환경 변수 로드
load_dotenv()

# API 키 수정 스크립트 가져오기
sys.path.append('/Users/ijunhyeong/Desktop/artipex-clone')
try:
    from fix_api_key import fix_openai_api_key
    # API 키 설정 적용
    api_key = fix_openai_api_key()
    print(f"OpenAI API 키가 설정되었습니다: {api_key[:10]}...")
except ImportError:
    print("fix_api_key.py 파일을 불러올 수 없습니다. 경로를 확인해주세요.")
    sys.exit(1)
except Exception as e:
    print(f"API 키 설정 중 오류 발생: {str(e)}")
    traceback.print_exc()
    sys.exit(1)

print("\n=== Adaptive RAG 테스트 실행 중 ===")
print("참고: 벡터 저장소가 설정되지 않은 경우 기본 더미 응답을 받게 됩니다.")
print("벡터 저장소 생성을 위해 scripts/create_vector_store.py 실행을 고려하세요.\n")

# Adaptive RAG 시스템 생성
try:
    adaptive_rag = create_adaptive_rag()
except Exception as e:
    print(f"Adaptive RAG 시스템 생성 중 오류 발생: {str(e)}")
    traceback.print_exc()
    sys.exit(1)

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
try:
    result = adaptive_rag.invoke(
        {
            "question": question,
            "conversation_history": conversation_history
        }, 
        config=config
    )

    # 결과 출력
    print(f"\n질문: {question}")
    print(f"\n답변: {result['generation']}\n")
    
    # 대화 기록 확인
    print("대화 기록이 저장되었습니다.")
    print(f"대화 항목 수: {len(result['conversation_history'])}")
    
except Exception as e:
    print(f"실행 중 오류 발생: {str(e)}")
    traceback.print_exc()
    sys.exit(1)

print("\n테스트 완료!") 