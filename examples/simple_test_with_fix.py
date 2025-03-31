"""
간단한 Artipex LLM 테스트 스크립트
API 키 수정 스크립트를 사용하여 direct 방식으로 답변 생성기를 테스트합니다.
"""

import uuid
import os
import sys
from dotenv import load_dotenv
from langchain_core.documents import Document
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

# 테스트 모드 메시지
print("\n=== 간단한 답변 생성기 테스트 ===")
print("이 스크립트는 전체 워크플로우 대신 답변 생성 부분만 테스트합니다.\n")

# 테스트용 더미 문서 생성
dummy_documents = [
    Document(
        page_content="색상은 인간의 감정에 큰 영향을 미칩니다. 빨간색은 열정과 활력을 주며, 파란색은 차분함과 안정감을 줍니다.",
        metadata={"source": "색채심리학개론", "page": 12}
    ),
    Document(
        page_content="예술 활동은 스트레스 감소에 효과적입니다. 색칠하기, 그림 그리기 등의 활동이 마음을 진정시키는 데 도움이 됩니다.",
        metadata={"source": "예술치료의 이해", "page": 45}
    ),
    Document(
        page_content="만다라 색칠하기는 집중력을 높이고 명상 효과를 줍니다. 이는 불안을 줄이는 데 도움이 됩니다.",
        metadata={"source": "명상과 예술", "page": 28}
    )
]

# 고유 세션 ID 생성
session_id = str(uuid.uuid4())[:8]
print(f"Session ID: {session_id}")

# 질문 준비
question = "색상이 감정에 미치는 영향에 대해 설명해주세요"
conversation_history = []

# 답변 생성 함수만 직접 호출
from adaptive_rag.generators.answer_generator import generate_answer, format_docs
from adaptive_rag.models import GraphState

# 상태 객체 생성
state = GraphState(
    question=question,
    documents=dummy_documents,
    conversation_history=conversation_history
)

# 답변 생성 시도
try:
    print("답변 생성 중...")
    result = generate_answer(state)
    
    # 결과 출력
    print(f"\n질문: {question}")
    print(f"\n답변: {result['generation']}\n")
    
    # 대화 기록 확인
    print("대화 기록이 저장되었습니다.")
    print(f"대화 항목 수: {len(result['conversation_history'])}")
    
except Exception as e:
    print(f"오류 발생: {str(e)}")
    traceback.print_exc()
    sys.exit(1)

print("\n테스트 완료!") 