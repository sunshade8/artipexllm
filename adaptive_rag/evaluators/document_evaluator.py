from ..models import GraphState
from typing import Dict, List, Tuple, Optional, Any

def grade_documents(state: GraphState) -> GraphState:
    """검색된 문서의 관련성을 평가합니다."""
    # 간단한 구현: 모든 문서를 관련 있다고 간주
    return {"documents_quality": "good", "documents": state.get("documents", [])}

def decide_to_generate(state: GraphState) -> str:
    """문서 품질에 따라 답변 생성 여부를 결정합니다."""
    # 간단한 구현: 항상 생성
    return "generate"

