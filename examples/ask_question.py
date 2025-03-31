#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
사용자가 질문을 입력하여 아트 테라피 모델과 대화할 수 있는 대화형 스크립트입니다.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
import uuid
from typing import List, Dict, Any, Optional

# 환경 변수 로드
load_dotenv()

# fix_api_key.py 파일 경로를 시스템 경로에 추가
sys.path.append(str(Path(__file__).resolve().parent.parent))

# 현재 프로젝트의 fix_api_key 모듈 임포트
from fix_api_key import fix_openai_api_key

# API 키 환경변수 설정 - API 키가 없으면 종료
api_key = fix_openai_api_key()
if not api_key:
    print("API 키가 설정되지 않아 프로그램을 종료합니다.")
    sys.exit(1)

# 경로 설정
PACKAGE_ROOT = Path(__file__).parent.parent.absolute()
DATA_DIR = os.path.join(PACKAGE_ROOT, "data")
PDF_DIR = os.path.join(DATA_DIR, "pdf")
MODEL_DIR = os.path.join(PACKAGE_ROOT, "models")
VECTOR_STORE_PATH = os.path.join(MODEL_DIR, "vector_store")

# 모델 설정
DEFAULT_MODEL = "gpt-4o"  # 또는 gpt-3.5-turbo 등 다른 모델
DEFAULT_TEMPERATURE = 0.0

# LLM 초기화
llm = ChatOpenAI(
    model_name=DEFAULT_MODEL,
    openai_api_key=api_key,
    openai_api_base="https://api.openai.com/v1",
    temperature=DEFAULT_TEMPERATURE
)

# 프롬프트 템플릿
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

def format_docs(documents):
    """문서 목록을 포맷팅하여 하나의 문자열로 반환합니다."""
    if not documents:
        return "관련 문서를 찾을 수 없습니다."
    
    return "\n\n".join(
        [
            f'<document><content>{doc.page_content}</content><source>{doc.metadata.get("source", "Unknown")}</source><page>{doc.metadata.get("page", 0)+1}</page></document>'
            for doc in documents
        ]
    )

def get_documents_from_vectorstore(question):
    """벡터 스토어에서 질문과 관련된 문서를 검색합니다."""
    try:
        # OpenAI 임베딩 초기화
        embedding = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            openai_api_key=api_key,
            openai_api_base="https://api.openai.com/v1"
        )
        
        # 벡터 스토어 초기화
        if os.path.exists(VECTOR_STORE_PATH):
            print("🔍 벡터 스토어에서 관련 문서를 검색합니다...")
            vector_store = Chroma(persist_directory=VECTOR_STORE_PATH, embedding_function=embedding)
            # 질문으로 검색
            docs = vector_store.similarity_search(question, k=5)
            return docs
        else:
            print("⚠️ 벡터 스토어를 찾을 수 없습니다. 더미 문서를 사용합니다.")
            return get_dummy_documents()
    except Exception as e:
        print(f"⚠️ 문서 검색 중 오류: {str(e)}")
        return get_dummy_documents()
        
def get_dummy_documents():
    """테스트용 더미 문서를 반환합니다."""
    return [
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

def generate_answer(question, documents, conversation_history=None):
    """질문과 문서를 바탕으로 답변을 생성합니다."""
    if conversation_history is None:
        conversation_history = []
    
    # 대화 기록 포맷팅
    formatted_history = ""
    if conversation_history:
        formatted_history += "최근 대화 내용:\n"
        for i, interaction in enumerate(conversation_history[-3:]):  # 최근 3개 대화만 포함
            formatted_history += f"User: {interaction['question']}\n"
            formatted_history += f"Assistant: {interaction['answer']}\n\n"

    # 문서 포맷팅
    formatted_docs = format_docs(documents)
    
    # 답변 생성
    print("💭 답변을 생성 중입니다...")
    chain = prompt_template | llm
    generation = chain.invoke({
        "context": formatted_docs, 
        "question": question,
        "conversation_history": formatted_history
    })
    
    if hasattr(generation, 'content'):
        return generation.content
    return str(generation)

def main():
    """메인 실행 함수"""
    print("\n===== Artipex LLM 예술 치료 대화 =====")
    print("종료하려면 'quit', 'exit', 또는 'q'를 입력하세요.")
    
    # 세션 ID 생성
    session_id = str(uuid.uuid4())[:8]
    print(f"세션 ID: {session_id}\n")
    
    # 대화 기록 초기화
    conversation_history = []
    
    # 대화 루프
    while True:
        # 사용자 입력 받기
        user_question = input("\n🔹 질문을 입력하세요: ")
        
        # 종료 명령 확인
        if user_question.lower() in ['quit', 'exit', 'q', '종료']:
            print("\n대화를 종료합니다. 감사합니다!")
            break
        
        if not user_question.strip():
            print("질문을 입력해주세요.")
            continue
        
        # 문서 검색
        documents = get_documents_from_vectorstore(user_question)
        print(f"✅ {len(documents)}개의 관련 문서를 찾았습니다.")
        
        # 답변 생성
        answer = generate_answer(user_question, documents, conversation_history)
        
        # 대화 기록에 추가
        conversation_history.append({
            "question": user_question,
            "answer": answer
        })
        
        # 결과 출력
        print("\n🔸 답변:")
        print(answer)
        print("\n" + "-" * 80)

if __name__ == "__main__":
    main() 