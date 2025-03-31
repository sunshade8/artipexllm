# Config file for adaptive-rag-ko
import os
from pathlib import Path

# 패키지 루트 디렉토리 자동 감지
PACKAGE_ROOT = Path(__file__).parent.parent.absolute()

# 데이터 및 모델 디렉토리 설정 (환경 변수에서 읽거나 기본값 사용)
DATA_DIR = os.environ.get("ADAPTIVE_RAG_DATA_DIR", os.path.join(PACKAGE_ROOT, "data"))
MODEL_DIR = os.environ.get("ADAPTIVE_RAG_MODEL_DIR", os.path.join(PACKAGE_ROOT, "models"))

# 벡터 저장소 경로
VECTOR_STORE_PATH = os.environ.get("ADAPTIVE_RAG_VECTOR_STORE_PATH", 
                                 os.path.join(MODEL_DIR, "vector_store"))

# 샘플 데이터 경로
SAMPLE_DATA_DIR = os.path.join(DATA_DIR, "sample")
# PDF 데이터 경로
PDF_DIR = os.path.join(DATA_DIR, "pdf")

# API 키 설정 (환경 변수에서 읽기)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")

# 기타 설정
DEFAULT_MODEL = os.environ.get("ADAPTIVE_RAG_DEFAULT_MODEL", "gpt-4o")
DEFAULT_TEMPERATURE = float(os.environ.get("ADAPTIVE_RAG_TEMPERATURE", "0.0"))

# 파일이 존재하는지 확인하고 사용자에게 경고
def check_environment():
    """환경 설정을 확인하고 문제가 있으면 사용자에게 경고합니다."""
    warnings = []
    
    if not OPENAI_API_KEY:
        warnings.append("⚠️ OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
    
    if not TAVILY_API_KEY:
        warnings.append("⚠️ TAVILY_API_KEY 환경 변수가 설정되지 않았습니다.")
    
    if not os.path.exists(SAMPLE_DATA_DIR):
        warnings.append(f"⚠️ 샘플 데이터 디렉토리가 존재하지 않습니다: {SAMPLE_DATA_DIR}")
    
    return warnings
