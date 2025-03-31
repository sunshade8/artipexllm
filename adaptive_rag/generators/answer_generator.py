from ..models import GraphState

# 문서 포맷팅 함수 정의
def format_docs(documents):
    """문서 목록을 포맷팅하여 하나의 문자열로 반환합니다.
    
    이 함수는 문서의 내용과 메타데이터(출처, 페이지 번호)를 포함하여 형식화합니다.
    참조 구현의 형식을 따릅니다.
    """
    return "\n\n".join(
        [
            f'<document><content>{doc.page_content}</content><source>{doc.metadata.get("source", "Unknown")}</source><page>{doc.metadata.get("page", 0)+1}</page></document>'
            for doc in documents
        ]
    )


# Define a function that transforms the input question into the required dictionary format.
def transform_question(question: str, conversation_history: str = "") -> dict:
    # 문서 검색
    retrieved_docs = retriever.invoke(question)
    # 문서 포맷팅
    formatted_context = format_docs(retrieved_docs)
    return {
        "context": formatted_context, 
        "question": question,
        "conversation_history": conversation_history
    }

# 답변 생성 노드 (대화 기록 활용 개선)
def generate_answer(state: GraphState):
    print("==== [답변 생성 중] ====")
    question = state["question"]
    documents = state["documents"]
    
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
    generation = OG_chain.invoke({
        "context": formatted_docs, 
        "question": question,
        "conversation_history": formatted_history + "\n" + context_hint if context_hint else formatted_history
    })
    
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

