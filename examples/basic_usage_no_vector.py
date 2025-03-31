from adaptive_rag import create_adaptive_rag
from langchain_core.runnables import RunnableConfig
import uuid
import os
from dotenv import load_dotenv
from langchain_core.documents import Document
import sys

# 환경 변수 로드
load_dotenv()

# 수동 테스트 모드 메시지
print("\n=== 벡터 저장소 없이 테스트 모드로 실행 중 ===")
print("참고: 이 스크립트는 실제 문서를 검색하지 않고 더미 데이터를 사용합니다.")
print("전체 기능을 사용하려면 벡터 저장소를 생성한 후 basic_usage.py를 실행하세요.\n")

# 벡터 저장소 없이 테스트하기 위한 더미 문서 생성
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

# 이 스크립트의 목적은 전체 시스템을 실행하는 것이 아니라
# 답변 생성 부분만 테스트하는 것입니다.
from adaptive_rag.generators.answer_generator import generate_answer, format_docs
from adaptive_rag.models import GraphState

# 더미 상태 객체 생성
state = GraphState(
    question=question,
    documents=dummy_documents,
    conversation_history=conversation_history
)

# 답변 생성 함수 직접 호출
try:
    result = generate_answer(state)
    
    # 결과 출력
    print(f"\n질문: {question}")
    print(f"\n답변: {result['generation']}\n")
    
    # 대화 기록 확인
    print("대화 기록이 저장되었습니다.")
    print(f"대화 항목 수: {len(result['conversation_history'])}")
    
except Exception as e:
    print(f"오류 발생: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n테스트 완료!") 