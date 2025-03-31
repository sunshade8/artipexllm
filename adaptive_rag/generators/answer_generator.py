from ..models import GraphState
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from ..config import OPENAI_API_KEY, DEFAULT_MODEL, DEFAULT_TEMPERATURE

# LLM 인스턴스 생성
if OPENAI_API_KEY.startswith("sk-proj-"):
    # Project-specific API 키 처리
    llm = ChatOpenAI(
        model_name=DEFAULT_MODEL,
        openai_api_key=OPENAI_API_KEY,
        openai_api_base="https://api.openai.com/v1",
        temperature=DEFAULT_TEMPERATURE
    )
else:
    llm = ChatOpenAI(temperature=DEFAULT_TEMPERATURE)

# 프롬프트 템플릿 정의
prompt_template = PromptTemplate.from_template(
    """
    당신은 예술 치료사로서, 사용자의 감정과 필요에 공감하고 예술을 통한 정서적 지원을 제공합니다.
    
    주어진 질문에 답변할 때, 다음 제시된 문서 정보를 참고하세요:
    
    {context}
    
    {conversation_history}
    
    사용자 질문: {question}
    
    답변 요구사항:
    1. 사용자의 감정에 공감하고 존중하는 어조를 유지하세요.
    2. 예술에 관한 지혜와 통찰력을 공유하세요.
    3. 가능하면 관련 예술 작품이나 예술가를 추천하세요.
    4. 사용자가 시도해볼 수 있는 간단한 창의적 활동을 제안하세요.
    5. 참조한 문서 출처를 [출처: 문서명, 페이지] 형식으로 인용하세요.
    
    답변:
    """
)

# 문서 포맷팅 함수 정의
def format_docs(documents):
    """문서 목록을 포맷팅하여 하나의 문자열로 반환합니다.
    
    이 함수는 문서의 내용과 메타데이터(출처, 페이지 번호)를 포함하여 형식화합니다.
    참조 구현의 형식을 따릅니다.
    """
    if not documents:
        return "관련 문서를 찾을 수 없습니다."
    
    return "\n\n".join(
        [
            f'<document><content>{doc.page_content}</content><source>{doc.metadata.get("source", "Unknown")}</source><page>{doc.metadata.get("page", 0)+1}</page></document>'
            for doc in documents
        ]
    )

# 질문 재작성 함수
def rewrite_question(state: GraphState) -> GraphState:
    """검색 결과가 불충분할 때 질문을 재작성합니다."""
    question = state["question"]
    
    # 간단한 구현: 질문에 "예술 치료" 키워드 추가
    if "예술" not in question and "치료" not in question:
        rewritten_question = f"{question} (예술 치료 관점에서)"
    else:
        rewritten_question = question
    
    return {"question": rewritten_question}

# 답변 생성 노드 (대화 기록 활용 개선)
def generate_answer(state: GraphState) -> GraphState:
    print("==== [답변 생성 중] ====")
    question = state["question"]
    documents = state.get("documents", [])
    
    # 대화 기록 가져오기 (없으면 빈 리스트 생성)
    conversation_history = state.get("conversation_history", [])
    
    # 대화 기록 분석 - 마지막 대화에서 중요 키워드 추출
    keywords = []
    if conversation_history:
        last_interactions = conversation_history[-2:] if len(conversation_history) >= 2 else conversation_history
        for interaction in last_interactions:
            prev_q = interaction.get("question", "").lower()
            # 중요 키워드 추출 (예: 색상, 감정, 예술 활동 등)
            for key in ["색상", "색깔", "파란색", "빨간색", "노란색", "감정", "기분", "활동", "그림", "작품"]:
                if key in prev_q and key not in keywords:
                    keywords.append(key)
    
    # 대화 기록을 프롬프트에 포함시키기 위해 포맷팅
    formatted_history = ""
    if conversation_history:
        formatted_history += "최근 대화 내용:\n"
        for i, interaction in enumerate(conversation_history[-3:]):  # 최근 3개 대화만 포함
            formatted_history += f"User: {interaction['question']}\n"
            formatted_history += f"Assistant: {interaction['answer']}\n\n"
    
    # 문서 포맷팅
    formatted_docs = format_docs(documents)
    
    # 대화 기록을 컨텍스트로 활용하여 답변 생성
    # 이전 대화의 주제를 고려하도록 지시하는 추가 컨텍스트
    context_hint = ""
    if keywords:
        context_hint = f"이전 대화에서 다음 주제가 언급되었습니다: {', '.join(keywords)}. 이를 고려하여 답변해주세요."
    
    # 대화 기록을 컨텍스트로 활용하여 답변 생성
    chain = prompt_template | llm
    generation = chain.invoke({
        "context": formatted_docs, 
        "question": question,
        "conversation_history": formatted_history + "\n" + context_hint if context_hint else formatted_history
    })
    generation = generation.content if hasattr(generation, 'content') else str(generation)
    
    # 현재 대화 추가
    conversation_history.append({
        "question": question,
        "answer": generation
    })
    
    # 결과 상태 반환 (대화 기록 포함)
    return {
        "generation": generation, 
        "documents": documents, 
        "question": question,
        "conversation_history": conversation_history
    }

