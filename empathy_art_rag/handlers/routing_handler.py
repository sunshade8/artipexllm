"""
Handler for routing decisions in the workflow.

This module is responsible for determining the next steps in the workflow,
such as whether retrieval is needed for a query.
"""

import re
from ..models import GraphState

# Patterns that suggest the question is informational and needs retrieval
INFORMATIONAL_PATTERNS = [
    r"(?i)what is",
    r"(?i)how (to|do|does|can|could)",
    r"(?i)why (is|are|do|does)",
    r"(?i)where (is|are|can)",
    r"(?i)when (is|was|will)",
    r"(?i)explain",
    r"(?i)describe",
    r"(?i)tell me about",
    r"(?i)information on",
    r"(?i)details (of|about)",
    # Korean patterns
    r"무엇",
    r"어떻게",
    r"왜",
    r"어디",
    r"언제",
    r"설명",
    r"알려줘",
    r"정보",
]

# Patterns that suggest the question is conversational/emotional and may not need retrieval
CONVERSATIONAL_PATTERNS = [
    r"(?i)^hi$",
    r"(?i)^hello$",
    r"(?i)how are you",
    r"(?i)nice to meet you",
    r"(?i)good (morning|afternoon|evening|day)",
    r"(?i)thanks",
    r"(?i)thank you",
    r"(?i)feel",
    r"(?i)emotion",
    r"(?i)sad",
    r"(?i)happy",
    r"(?i)angry",
    r"(?i)frustrated",
    r"(?i)excited",
    # Korean patterns
    r"안녕",
    r"반가워",
    r"좋은 아침|점심|저녁|하루",
    r"고마워",
    r"감사",
    r"기분",
    r"감정",
    r"슬프",
    r"행복",
    r"화나",
    r"짜증",
]

def determine_needs_retrieval(state: GraphState) -> GraphState:
    """
    Determine whether the query needs context retrieval.
    
    Args:
        state: Current graph state
        
    Returns:
        Updated state with retrieval decision
    """
    print("==== [Determining If Retrieval Needed] ====")
    
    # Extract processed query from state
    query = state.get("processed_query", "")
    query_lower = state.get("query_lower", query.lower())
    
    # Check for informational patterns
    informational_match = False
    for pattern in INFORMATIONAL_PATTERNS:
        if re.search(pattern, query):
            informational_match = True
            print(f"Informational pattern match: {pattern}")
            break
            
    # Check for conversational patterns
    conversational_match = False
    for pattern in CONVERSATIONAL_PATTERNS:
        if re.search(pattern, query):
            conversational_match = True
            print(f"Conversational pattern match: {pattern}")
            break
    
    # Decision logic
    retrieval_needed = False
    
    # If it's clearly informational and not just conversational
    if informational_match and not conversational_match:
        retrieval_needed = True
    # If mixed signals, lean toward retrieval
    elif informational_match and conversational_match:
        retrieval_needed = True
    # If clearly conversational
    elif conversational_match and not informational_match:
        retrieval_needed = False
    # Default case: if we can't determine, do retrieval for safety
    else:
        retrieval_needed = True
    
    # Check for the detected emotion - if it's primarily emotional, we might skip retrieval
    detected_emotion = state.get("detected_emotion")
    if detected_emotion and not retrieval_needed:
        print(f"Skipping retrieval due to emotion focus: {detected_emotion}")
        retrieval_needed = False
    
    print(f"Retrieval needed: {retrieval_needed}")
    
    # Add routing result to state
    updated_state = {
        **state,
        "retrieval_needed": retrieval_needed,
        "determine_needs_retrieval_routing": 0 if retrieval_needed else 1  # 0=retrieve, 1=skip retrieval
    }
    
    return updated_state 