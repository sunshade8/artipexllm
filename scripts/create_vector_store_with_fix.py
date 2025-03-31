#!/usr/bin/env python
"""
벡터 스토어 생성 스크립트 (API 키 수정 포함)
이 스크립트는 PDF 디렉토리의 문서를 처리하여 Chroma 벡터 스토어를 생성합니다.
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 환경 변수 로드 (.env 파일 사용)
load_dotenv()

# 스크립트 디렉토리 확인
script_dir = Path(__file__).parent.absolute()
package_dir = script_dir.parent
sys.path.append(str(package_dir))

# API 키 수정 스크립트 가져오기
try:
    from fix_api_key import fix_openai_api_key
    # API 키 설정 적용
    api_key = fix_openai_api_key()
    print(f"OpenAI API 키가 설정되었습니다: {api_key[:10]}...")
except ImportError:
    print("fix_api_key.py 파일을 불러올 수 없습니다.")
    sys.exit(1)
except Exception as e:
    print(f"API 키 설정 중 오류 발생: {str(e)}")
    sys.exit(1)

# 설정 가져오기
from adaptive_rag.config import PDF_DIR, VECTOR_STORE_PATH

def main():
    # OpenAI API 키 확인
    if not os.environ.get("OPENAI_API_KEY"):
        print("오류: OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        print("API 키를 설정하고 다시 시도하세요.")
        sys.exit(1)
    
    # PDF 디렉토리 확인
    if not os.path.exists(PDF_DIR) or not any(f.endswith('.pdf') for f in os.listdir(PDF_DIR)):
        print(f"오류: PDF 파일을 찾을 수 없습니다. 디렉토리: {PDF_DIR}")
        sys.exit(1)
    
    print(f"PDF 디렉토리: {PDF_DIR}")
    print(f"벡터 스토어 경로: {VECTOR_STORE_PATH}")
    
    # 문서 로딩 및 텍스트 분할
    print("문서 로딩 중...")
    documents = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
    )
    
    for filename in os.listdir(PDF_DIR):
        if filename.endswith(".pdf"):
            try:
                pdf_path = os.path.join(PDF_DIR, filename)
                print(f"처리 중: {filename}")
                
                loader = PyPDFLoader(pdf_path)
                pdf_docs = loader.load()
                
                for doc in pdf_docs:
                    if not doc.page_content.strip():  # 빈 페이지 건너뛰기
                        continue
                    # 메타데이터에 파일명 추가
                    doc.metadata['filename'] = filename
                
                chunks = text_splitter.split_documents(pdf_docs)
                documents.extend(chunks)
                print(f"  - {len(chunks)} 청크 생성됨")
                
            except Exception as e:
                print(f"  - 오류 발생: {str(e)}")
    
    if not documents:
        print("처리할 문서가 없습니다.")
        sys.exit(1)
    
    print(f"총 {len(documents)} 청크 처리됨")
    
    # 벡터 스토어 생성
    print("임베딩 생성 및 벡터 스토어 저장 중...")
    
    # 임베딩 모델 초기화 (프로젝트 특정 API 키 처리)
    embedding_model = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        openai_api_key=api_key,
        openai_api_base="https://api.openai.com/v1"
    )
    
    # 벡터 스토어 디렉토리 확인
    os.makedirs(os.path.dirname(VECTOR_STORE_PATH), exist_ok=True)
    
    Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=VECTOR_STORE_PATH
    )
    
    print(f"벡터 스토어가 성공적으로 생성되었습니다: {VECTOR_STORE_PATH}")

if __name__ == "__main__":
    main() 