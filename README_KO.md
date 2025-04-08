# EmpathyArtRAG

EmpathyArtRAG는 대화형 인공지능과 감정 인식 기반 예술작품 추천 기능을 결합한 고급 AI 시스템으로, 사용자에게 독특하고 공감적인 경험을 제공합니다.

## 🌟 주요 기능

- **감정 인식 기술**: 사용자 메시지와 대화 맥락에서 자동으로 감정을 감지
- **스마트 검색**: 쿼리 유형에 따라 적응적으로 정보 검색 여부를 결정
- **예술작품 추천**: 사용자의 감정 상태와 공명하는, 관련 예술작품을 제안
- **유연한 아키텍처**: 쉽게 커스터마이징할 수 있는 모듈식 설계와 구성 가능한 워크플로우 그래프

## 🧠 작동 방식

시스템은 다음과 같은 정교한 워크플로우를 통해 사용자 상호작용을 처리합니다:

1. **초기 처리**: 사용자의 질문을 분석하고 주요 정보를 추출
2. **감정 감지**: 키워드 분석을 사용하여 메시지에 표현된 감정을 식별
3. **쿼리 라우팅**: 질문 유형과 감정적 내용에 기반하여 최적의 경로 결정
4. **맥락 검색**: 사실적 응답이 필요할 때 관련 정보 가져오기
5. **예술 추천**: 감지된 감정을 보완하는 예술작품 선택
6. **응답 생성**: 정보 요구와 감정적 맥락을 모두 다루는 사려 깊은 답변 생성

## 📁 프로젝트 구조

```
empathy_art_rag/
├── art/                    # 예술작품 추천 컴포넌트
│   └── artwork_recommender.py
├── generators/             # 응답 생성 모듈
│   ├── answer_generator.py
│   └── art_generator.py
├── handlers/               # 쿼리 처리 시스템
│   ├── emotion_handler.py
│   ├── question_handler.py
│   ├── retrieval_handler.py
│   └── routing_handler.py
├── models.py               # 데이터 모델 및 구조
├── workflow.py             # 워크플로우 정의 및 제어
├── config.py               # 시스템 구성
└── main.py                 # 메인 어플리케이션 인터페이스
```

## 🚀 시작하기

### 필수 요구사항

- Python 3.8 또는 그 이상
- 예술작품 데이터셋 접근 권한 (CSV 파일 및 이미지 폴더)
- 필요한 Python 패키지 (requirements.txt 참조)

### 설치 방법

1. **저장소 복제**:
   ```bash
   git clone https://github.com/username/EmpathyArtRAG.git
   cd EmpathyArtRAG
   ```

2. **의존성 설치**:
   ```bash
   pip install -r requirements.txt
   ```

3. **환경 구성**:
   `.env.example`을 `.env`로 복사하고 필요한 설정으로 업데이트:
   ```bash
   cp .env.example .env
   # 원하는 편집기로 .env 수정
   ```

   구성해야 할 주요 설정:
   - `ART_CSV_PATH`: 예술작품 메타데이터 CSV 경로
   - `ART_IMAGE_FOLDER`: 예술작품 이미지가 포함된 폴더 경로
   - `MODEL_NAME`: 사용할 언어 모델 이름

## 💻 사용 예시

### 기본 사용법

```python
from empathy_art_rag import process_query

# 간단한 쿼리 처리
response = process_query(
    query="오늘은 업무로 인해 정말 압도된 기분이에요",
    conversation_id=None  # 새 대화
)

# 응답 표시
print(response.answer)

# 예술작품이 추천되었는지 확인
if response.artwork:
    print("\n추천 예술작품:")
    print(f"제목: {response.artwork['title']}")
    print(f"작가: {response.artwork['artist']}")
    print(f"설명: {response.artwork['description']}")
```

### 데모 실행

프로젝트에는 쉽게 테스트할 수 있는 데모 스크립트가 포함되어 있습니다:

```bash
# 예제 대화 실행
python demo.py

# 대화형 데모 실행
python demo.py --interactive
```

### 웹 인터페이스

시각적 상호작용을 위한 웹 애플리케이션도 제공됩니다:

```bash
# 웹 애플리케이션 시작
python web_app.py
```

브라우저에서 `http://localhost:8000`을 방문하세요.

## 🔧 커스터마이징

이 시스템은 쉽게 확장할 수 있도록 설계되었습니다:

- **감정 추가**: `emotion_handler.py`에 새 키워드를 추가하여 감정 감지 확장
- **예술작품 업데이트**: 새 예술작품을 데이터셋에 추가하여 추천 범위 확장
- **워크플로우 수정**: `workflow.py`의 워크플로우 그래프를 조정하여 처리 로직 변경
- **LLM 변경**: `config.py`를 업데이트하여 다른 언어 모델과 통합

## 📊 성능 참고사항

- 예술작품 임베딩을 캐싱하는 동안 첫 번째 쿼리는 느릴 수 있음
- 유사한 감정 콘텐츠를 가진 후속 상호작용은 더 빠름
- 시스템은 대량 처리보다는 대화 흐름에 최적화되어 있음

## 🔍 문제 해결

일반적인 문제와 해결책:

- **예술작품 누락**: `.env`의 경로가 올바른지, CSV 파일에 유효한 항목이 있는지 확인
- **느린 응답**: CLIP 모델이 올바르게 로드되고 임베딩이 캐싱되고 있는지 확인
- **감지되지 않는 감정**: 더 명시적인 감정 언어를 사용하거나 새 감정 키워드 추가

## 📝 라이선스

[MIT 라이선스](LICENSE)

## 🙏 감사의 말

이 프로젝트는 EmotionArtSystem 및 adaptive-rag-ko-improved 시스템의 구성 요소를 통합하고, 맞춤형 감정 감지 및 예술작품 추천 알고리즘으로 향상되었습니다. 