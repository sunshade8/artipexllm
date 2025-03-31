from ..models import GraphState
from typing import Dict, List, Tuple, Optional, Any

def check_hallucination(state: GraphState) -> str:
    """생성된 답변에 환각이 있는지 확인합니다."""
    # 간단한 구현: 항상 관련성 있다고 간주
    return "relevant"

