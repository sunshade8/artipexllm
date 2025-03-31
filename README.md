# Adaptive RAG Ko

한국어 지원 Adaptive RAG 시스템으로, 멀티턴 대화 기능을 포함합니다.

## 특징

- 질문 유형에 따른 지능적 라우팅 (벡터 저장소/웹 검색)
- 멀티턴 대화 지원 (대화 기록 및 문맥 유지)
- 검색된 문서의 관련성 평가
- 할루시네이션(환각) 감지 및 처리
- 한국어 질문 처리 최적화

## 🚀 팀원을 위한 빠른 시작 가이드

이 섹션에서는 저장소를 클론하고 시스템을 설정하는 방법을 단계별로 설명합니다.

### 1. 저장소 클론하기

```bash
git clone https://github.com/yourusername/adaptive-rag-ko.git
cd adaptive-rag-ko
```

### 2. 패키지 설치하기

```bash
pip install -e .
```

이 명령은 모든 필요한 의존성을 설치하고 개발 모드로 패키지를 설정합니다.

### 3. API 키 설정하기

API 키를 설정하는 방법은 두 가지가 있습니다:

#### 3.1 .env 파일 사용 (기본 방법)

`.env.example` 파일을 `.env`로 복사하고 API 키를 설정합니다:

```bash
cp .env.example .env
```

그런 다음 텍스트 에디터로 `.env` 파일을 열고 자신의 API 키를 입력합니다:

```
# API 키 설정
OPENAI_API_KEY=your-actual-openai-api-key
TAVILY_API_KEY=your-actual-tavily-api-key

# 그 외 설정은 기본값 사용
```

#### 3.2 fix_api_key.py 사용 (프로젝트 특정 API 키 문제 해결)

프로젝트 특정 OpenAI API 키(sk-proj-로 시작하는 키)를 사용하거나 API 키 설정에 문제가 있는 경우, 다음 방법을 사용하세요:

1. `fix_api_key.py` 파일을 확인하고 API 키를 업데이트합니다:

```python
def fix_openai_api_key():
    api_key = "여기에-당신의-API-키-입력"
    os.environ["OPENAI_API_KEY"] = api_key
    return api_key
```

2. 이 파일은 다른 스크립트에서 자동으로 가져와 API 키 문제를 해결합니다.

> **중요**: `.env` 파일과 `fix_api_key.py`는 `.gitignore`에 포함되어 있어 GitHub에 업로드되지 않습니다. 각자 자신의 API 키를 안전하게 사용할 수 있습니다.

### 4. 벡터 스토어 생성하기

API 키 설정에 따라 두 가지 방법으로 벡터 스토어를 생성할 수 있습니다:

#### 4.1 기본 스크립트 사용
```bash
python scripts/create_vector_store.py
```

#### 4.2 API 키 문제 해결 스크립트 사용
```bash
python scripts/create_vector_store_with_fix.py
```

이 과정은 약간의 시간이 소요될 수 있으며, OpenAI API 사용량에 따라 비용이 발생할 수 있습니다.

### 5. 시스템 실행하기

여러 방법으로 시스템을 실행할 수 있습니다:

#### 5.1 기본 사용 예제:
```bash
python examples/basic_usage.py  # 기본 예제
python examples/basic_usage_with_fix.py  # API 키 문제 해결 버전
```

#### 5.2 대화형 질문 스크립트:
```bash
python examples/ask_question.py  # 대화형 인터페이스로 질문 입력
```

#### 5.3 단순 데모 실행:
```bash
python examples/simplified_demo.py  # 단일 파일로 구현된 간소화 버전
```

#### 5.4 멀티턴 대화 예제:
```bash
python examples/multi_turn_conversation.py
```

### 6. 문제 해결

- **API 키 오류**: 
  - `.env` 파일에 API 키가 올바르게 설정되었는지 확인하세요.
  - `fix_api_key.py` 파일을 사용하여 API 키 문제를 해결해보세요.
  - sk-proj- 형식의 API 키를 사용하는 경우 `create_vector_store_with_fix.py`를 사용하세요.

- **ImportError**: 패키지가 올바르게 설치되었는지 확인하세요 (`pip install -e .`).
- **벡터 스토어 오류**: 벡터 스토어 생성 스크립트를 실행했는지 확인하세요.

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

# API 키 문제가 있는 경우
# from fix_api_key import fix_openai_api_key
# api_key = fix_openai_api_key()

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

### 대화형 인터페이스 사용

```python
# ask_question.py 스크립트를 실행하여 대화형 인터페이스 사용
python examples/ask_question.py
```

### 멀티턴 대화

`examples/multi_turn_conversation.py` 파일을 참고하세요.

## 새로 추가된 스크립트 설명

이 프로젝트에는 다음과 같은 추가 스크립트가 포함되어 있습니다:

- **ask_question.py**: 대화형 콘솔 인터페이스로 질문을 입력하고 답변을 받을 수 있는 스크립트
- **simplified_demo.py**: 전체 시스템의 핵심 기능을 단일 파일로 구현한 간소화 버전
- **basic_usage_with_fix.py**: API 키 문제를 해결한 기본 사용 예제
- **create_vector_store_with_fix.py**: API 키 문제를 해결한 벡터 스토어 생성 스크립트
- **fix_api_key.py**: OpenAI API 키 관련 문제를 해결하는 유틸리티 스크립트

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
│   ├── sample/         # 샘플 데이터
│   └── pdf/            # PDF 문서
├── models/             # 모델 및 인덱스 저장
│   └── vector_store/   # 벡터 스토어
├── examples/           # 사용 예제
│   ├── basic_usage.py
│   ├── basic_usage_with_fix.py
│   ├── ask_question.py
│   ├── simplified_demo.py
│   └── multi_turn_conversation.py
├── notebooks/          # 주피터 노트북
└── scripts/            # 유틸리티 스크립트
    ├── create_vector_store.py
    └── create_vector_store_with_fix.py
```

## 문제 해결

- **API 키 오류**: 
  - `fix_api_key.py`를 사용하여 API 키 문제를 해결해보세요.
  - OpenAI API 키가 프로젝트 특정 형식(sk-proj-)인 경우, 관련 수정 스크립트를 사용하세요.

- **데이터 경로 오류**: 적절한 데이터 경로가 설정되었는지 확인하세요.
- **메모리 오류**: 대용량 파일 처리 시 청크 크기를 조정하세요.

## 라이센스

MIT
