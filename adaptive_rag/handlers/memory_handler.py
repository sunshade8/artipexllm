from ..models import GraphState

# 이전 질문에 대한 답변 생성 함수 (개선됨)
def memory_query_handler(state: GraphState):
    print("==== [이전 대화 기록 확인 중] ====")
    question = state["question"]
    
    # 대화 기록 가져오기 (없으면 빈 리스트 생성)
    conversation_history = state.get("conversation_history", [])
    
    # 현재 질문이 이전 질문/대화 내용에 대한 것인지 확인 (확장된 질문 패턴)
    memory_questions = [
        "내가 한 질문이 뭐야?", "내가 뭐라고 물었지?", "이전 질문", "전에 물어본 내용",
        "what was my question", "previous question", "what did i ask", "전에 질문한 내용",
        "직전에 대화한", "어떤 내용", "무슨 얘기", "이야기했", "말했", "언급했", 
        "전에 어떤", "우리가 얘기한", "방금 전에", "지금까지", "우리 대화",
        "어떤 색", "무슨 색", "색상", "어떤 주제", "전에 말한"
    ]
    
    is_memory_question = any(mem_q.lower() in question.lower() for mem_q in memory_questions)
    
    if is_memory_question and conversation_history:
        print("메모리 관련 질문 감지됨, 이전 대화 기록 반환")
        # 이전 대화 기록 있음
        if any(word in question.lower() for word in ["색상", "색깔", "컬러", "color"]):
            # 색상 관련 질문인 경우, 마지막 대화에서 색상 정보 추출
            last_question = conversation_history[-1]["question"] if conversation_history else ""
            colors = ["빨간색", "파란색", "노란색", "초록색", "보라색", "검정색", "흰색", 
                     "red", "blue", "yellow", "green", "purple", "black", "white"]
            
            found_colors = []
            for color in colors:
                if color in last_question.lower():
                    found_colors.append(color)
            
            if found_colors:
                color_info = ", ".join(found_colors)
                generation = f"""직전 대화에서는 {color_info}에 관해 이야기했습니다. 

구체적으로 아래와 같은 질문을 하셨습니다:
"{last_question}"
"""
                return {"generation": generation, "question": question, "conversation_history": conversation_history}
        
        # 일반적인 대화 기록 반환
        prev_interactions = ""
        for i, interaction in enumerate(conversation_history):
            prev_interactions += f"{i+1}. 질문: {interaction['question']}\n"
        
        generation = f"""이전에 다음과 같은 질문을 하셨습니다:

{prev_interactions}"""
        return {"generation": generation, "question": question, "conversation_history": conversation_history}
    
    # 대화 기록 관련 질문이 아니거나 기록이 없는 경우
    return None

