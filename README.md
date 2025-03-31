# Adaptive RAG Ko

한국어 지원 Adaptive RAG 시스템으로, 멀티턴 대화 기능을 포함합니다.

## 특징

- 질문 유형에 따른 지능적 라우팅 (벡터 저장소/웹 검색)
- 멀티턴 대화 지원 (대화 기록 및 문맥 유지)
- 검색된 문서의 관련성 평가
- 할루시네이션(환각) 감지 및 처리
- 한국어 질문 처리 최적화

## 설치

### 사전 요구사항

- Python 3.8 이상
- OpenAI API 키
- Tavily API 키 (웹 검색용)

### pip으로 설치

```bash
pip install adaptive-rag-ko
```

### 소스에서 직접 설치

```bash
git clone https://github.com/yourusername/adaptive-rag-ko.git
cd adaptive-rag-ko
pip install -e .
```

### 환경 변수 설정

`.env` 파일을 생성하여 필요한 API 키를 설정하거나, 환경 변수로 직접 설정할 수 있습니다:

```bash
# .env 파일 예시
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key

# 선택적 설정
ADAPTIVE_RAG_DATA_DIR=/path/to/your/data
ADAPTIVE_RAG_MODEL_DIR=/path/to/your/models
ADAPTIVE_RAG_DEFAULT_MODEL=gpt-4o
```

## 데이터 준비

### 샘플 데이터 사용

이 패키지는 기본적인 샘플 데이터를 포함하고 있어 별도의 설정 없이 테스트해볼 수 있습니다. 색상 심리학과 예술 치료에 관한 PDF 문서들도 포함되어 있어 실제 RAG 시스템을 체험할 수 있습니다.

### 추가 데이터 다운로드

더 많은 예제 데이터가 필요하다면 제공된 스크립트를 사용할 수 있습니다:

```bash
python scripts/download_example_data.py
```

### 나만의 데이터 사용

자신의 데이터를 사용하려면 `data` 디렉토리에 데이터 파일을 추가하고, 환경 변수로 경로를 지정하세요:

```bash
export ADAPTIVE_RAG_DATA_DIR=/path/to/your/data
```

## 사용 예시

### 기본 사용법

```python
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
```

### 멀티턴 대화

`examples/multi_turn_conversation.py` 파일을 참고하세요.

## 프로젝트 구조

```
adaptive-rag-ko/
├── adaptive_rag/       # 메인 패키지
│   ├── config.py       # 설정 관리
│   ├── models.py       # 데이터 모델
│   ├── retrievers/     # 검색 모듈
│   ├── handlers/       # 질문 처리
│   ├── evaluators/     # 문서 평가
│   ├── generators/     # 답변 생성
│   └── workflow/       # 그래프 워크플로우
├── data/               # 데이터 디렉토리
│   └── sample/         # 샘플 데이터
├── models/             # 모델 및 인덱스 저장
├── examples/           # 사용 예제
├── notebooks/          # 주피터 노트북
└── scripts/            # 유틸리티 스크립트
```

## 문제 해결

- **API 키 오류**: 환경 변수가 올바르게 설정되었는지 확인하세요.
- **데이터 경로 오류**: 적절한 데이터 경로가 설정되었는지 확인하세요.
- **메모리 오류**: 대용량 파일 처리 시 청크 크기를 조정하세요.

## 라이센스

MIT
