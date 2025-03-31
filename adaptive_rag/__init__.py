"""
Adaptive RAG Ko - 한국어 지원 Adaptive RAG 시스템
"""
from .workflow.graph import create_workflow
from .models import GraphState
from .config import check_environment

__version__ = "0.1.0"

# 환경 체크 실행
env_warnings = check_environment()
if env_warnings:
    print("\n".join(env_warnings))
    print("환경 설정을 완료해주세요. README.md 파일을 참고하세요.")

def create_adaptive_rag():
    """Adaptive RAG 시스템의 기본 인스턴스를 생성합니다."""
    return create_workflow()
