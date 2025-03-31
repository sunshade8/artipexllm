from ..models import GraphState
from ..config import VECTOR_STORE_PATH, SAMPLE_DATA_DIR, DATA_DIR, PDF_DIR, OPENAI_API_KEY
import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# 벡터 저장소 초기화 (없으면 빈 응답 반환)
try:
    # OpenAI 임베딩 모델 초기화
    if OPENAI_API_KEY.startswith("sk-proj-"):
        # Project-specific API 키 처리
        embedding = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            openai_api_key=OPENAI_API_KEY,
            openai_api_base="https://api.openai.com/v1"
        )
    else:
        embedding = OpenAIEmbeddings()
    
    # 벡터 저장소 초기화 시도
    if os.path.exists(VECTOR_STORE_PATH):
        vector_store = Chroma(persist_directory=VECTOR_STORE_PATH, embedding_function=embedding)
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    else:
        print(f"경고: 벡터 저장소를 찾을 수 없습니다. 경로: {VECTOR_STORE_PATH}")
        # 벡터 저장소가 없을 경우 빈 문서 반환하는 간단한 retriever 생성
        def dummy_retrieve(query):
            print(f"벡터 저장소가 없어 더미 문서를 반환합니다. 쿼리: {query}")
            return [Document(page_content="벡터 저장소가 없습니다. 'scripts/create_vector_store.py'를 실행하여 생성해주세요.",
                            metadata={"source": "system"})]
        
        # 호출 가능한 객체로 만들기
        class DummyRetriever:
            def invoke(self, query):
                return dummy_retrieve(query)
        
        retriever = DummyRetriever()
        
except Exception as e:
    print(f"벡터 저장소 초기화 중 오류 발생: {str(e)}")
    # 에러 발생 시 빈 문서 반환
    def error_retrieve(query):
        return [Document(page_content=f"벡터 저장소 초기화 중 오류가 발생했습니다: {str(e)}",
                       metadata={"source": "system"})]
    
    class ErrorRetriever:
        def invoke(self, query):
            return error_retrieve(query)
    
    retriever = ErrorRetriever()

# 벡터 저장소에서 벡터를 검색하는 함수
ensemble_retriever = retriever

# 문서 검색 노드
def retrieve_documents(state: GraphState) -> GraphState:
    print("==== [문서 검색 중] ====")
    question = state["question"]
    conversation_history = state.get("conversation_history", [])
    
    # 문서 검색 수행
    documents = ensemble_retriever.invoke(question)
    return {"documents": documents, "conversation_history": conversation_history}

