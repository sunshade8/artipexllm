# Adaptive RAG Ko 시작하기

이 문서는 팀원들이 Adaptive RAG Ko 시스템을 설정하고 실행하는 방법을 상세히 안내합니다.

## 목차

1. [사전 요구사항](#사전-요구사항)
2. [설치 과정](#설치-과정)
3. [API 키 설정](#api-키-설정)
4. [벡터 스토어 생성](#벡터-스토어-생성)
5. [시스템 실행](#시스템-실행)
6. [문제 해결](#문제-해결)

## 사전 요구사항

시작하기 전에 다음 항목이 준비되어 있는지 확인하세요:

- Python 3.8 이상 설치
- 유효한 OpenAI API 키 (https://platform.openai.com에서 발급)
- 유효한 Tavily API 키 (웹 검색용, https://tavily.com에서 발급)
- Git 설치
- 터미널/명령 프롬프트 사용 가능

## 설치 과정

### 1. 저장소 클론하기

먼저 GitHub에서 저장소를 클론합니다:

```bash
git clone https://github.com/yourusername/adaptive-rag-ko.git
cd adaptive-rag-ko
```

### 2. 가상 환경 생성 (선택 사항)

가상 환경을 사용하면 다른 프로젝트와의 의존성 충돌을 방지할 수 있습니다:

```bash
# venv 사용
python -m venv venv

# 가상 환경 활성화
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3. 패키지 설치

다음 명령으로 패키지와 모든 의존성을 설치합니다:

```bash
pip install -e .
```

## API 키 설정

### 1. 환경 파일 생성

예제 환경 파일을 복사하여 자신의 설정 파일을 만듭니다:

```bash
cp .env.example .env
```

### 2. API 키 입력

텍스트 에디터로 `.env` 파일을 열고 자신의 API 키를 입력합니다:

```
# API 키 설정
OPENAI_API_KEY=sk-your-actual-openai-api-key-here
TAVILY_API_KEY=your-actual-tavily-api-key-here

# 그 외 설정은 기본값 사용 가능
```

> **보안 주의사항**: `.env` 파일은 `.gitignore`에 포함되어 있어 GitHub에 업로드되지 않습니다. 민감한 키가 공개되지 않도록 주의하세요.

## 벡터 스토어 생성

저장소에는 PDF 파일이 이미 포함되어 있지만, 벡터 스토어는 각 환경에서 생성해야 합니다. API 키를 설정한 후 다음 명령으로 벡터 스토어를 생성하세요:

```bash
python scripts/create_vector_store.py
```

이 과정은 PDF 문서를 처리하여 검색 가능한 벡터 인덱스를 만듭니다. 처리 시간은 문서 크기와 수에 따라 달라질 수 있으며, OpenAI API 사용량에 따라 비용이 발생할 수 있습니다.

성공적으로 완료되면 `models/sample_indices/vector_store` 디렉토리에 벡터 스토어가 생성됩니다.

## 시스템 실행

### 기본 사용 예제

단일 질문에 대한 답변을 얻는 기본 예제를 실행합니다:

```bash
python examples/basic_usage.py
```

### 멀티턴 대화 예제

여러 질문을 연속해서 묻는 대화형 예제를 실행합니다:

```bash
python examples/multi_turn_conversation.py
```

### 자신만의 코드 작성

패키지를 활용하여 자신만의 코드를 작성할 수도 있습니다:

```python
from adaptive_rag import create_adaptive_rag
from langchain_core.runnables import RunnableConfig
import uuid
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# Adaptive RAG 시스템 생성
adaptive_rag = create_adaptive_rag()

# 세션 ID 및 설정
session_id = str(uuid.uuid4())[:8]
config = RunnableConfig(
    configurable={"thread_id": session_id}
)

# 질문 및 실행
question = "자신만의 질문을 여기에 입력하세요"
result = adaptive_rag.invoke(
    {
        "question": question,
        "conversation_history": []
    }, 
    config=config
)

# 결과 출력
print(f"질문: {question}")
print(f"답변: {result['generation']}")
```

## 문제 해결

### 일반적인 오류

#### API 키 오류
```
openai.error.AuthenticationError: No API key provided.
```
**해결책**: `.env` 파일에 유효한 OpenAI API 키가 설정되어 있는지 확인하세요.

#### 모듈 찾기 오류
```
ModuleNotFoundError: No module named 'adaptive_rag'
```
**해결책**: 패키지가 올바르게 설치되었는지 확인하세요 (`pip install -e .`).

#### 벡터 스토어 오류
```
FileNotFoundError: [Errno 2] No such file or directory: '.../models/sample_indices/vector_store'
```
**해결책**: `python scripts/create_vector_store.py`를 실행하여 벡터 스토어를 생성했는지 확인하세요.

### 그 외 문제

- **메모리 부족**: 대용량 파일 처리 시 메모리 부족 오류가 발생할 수 있습니다. `scripts/create_vector_store.py` 파일에서 `chunk_size` 값을 낮추어 조정해보세요.
- **속도 문제**: 처리 속도가 느리면 더 작은 데이터셋으로 시작하거나, `data/pdf/` 디렉토리에서 일부 PDF 파일만 남겨두고 벡터 스토어를 다시 생성해보세요.
- **호환성 문제**: 패키지 버전 충돌이 발생하면 `pip install -e .` 명령을 다시 실행하여 모든 의존성이 올바르게 설치되었는지 확인하세요.

## 추가 도움말

추가 질문이나 문제가 있으면 GitHub 저장소의 Issues 섹션에 문의하거나 팀 채팅에서 문의하세요. 