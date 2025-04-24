import os
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# === PDF Loading and Splitting Utilities ===

# Loads a single PDF file using PyPDFLoader.
def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    print(f"'{os.path.basename(file_path)}'에서 {len(documents)} 페이지를 로드했습니다.")
    return documents

# Determines an appropriate chunk size for text splitting based on the text length.
# Uses larger chunks for longer texts.
# 문서 길이에 따른 적응형 청크 크기 설정 함수
def adaptive_chunk_size(text):
    length = len(text)
    if length < 2000:
        return 500
    elif length < 5000:
        return 1000
    else:
        return 1500

# 모든 PDF 로드 및 적응형 분할 함수
def load_and_split_pdfs(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    pdf_files = glob.glob(f"{data_dir}/*.pdf")
    print(f"Total {len(pdf_files)} pdf files found.")

    raw_documents = []
    for pdf_file in pdf_files:
        try:
            docs = load_pdf(pdf_file)
            raw_documents.extend(docs)
        except Exception as e:
            print(f"파일 '{os.path.basename(pdf_file)}' error during loading the files: {e}")
    print(f"총 {len(raw_documents)} pages loaded.")

    adaptive_documents = []
    for doc in raw_documents:
        chunk_size = adaptive_chunk_size(doc.page_content)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=200,
            length_function=len,
        )
        splits = text_splitter.split_documents([doc])
        adaptive_documents.extend(splits)

    print(f"적응형 분할 방식으로 문서를 {len(adaptive_documents)} 청크로 분할했습니다.")
    return adaptive_documents 