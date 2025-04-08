"""
Handler for processing user questions.

This module is responsible for initial processing of user questions.
"""

from ..models import GraphState

def process_question(state: GraphState) -> GraphState:
    """
    Process an incoming user question.
    
    Args:
        state: Current graph state
        
    Returns:
        Updated state with processed question
    """
    print("==== [Processing Question] ====")
    
    # Extract question from state
    query = state.get("query", "")
    
    # Basic preprocessing
    # - Trim whitespace
    # - Convert to lowercase for analysis (but keep original)
    processed_query = query.strip()
    query_lower = processed_query.lower()
    
    # Add to state
    updated_state = {
        **state,
        "processed_query": processed_query,
        "query_lower": query_lower,
        "query_length": len(processed_query),
        "query_word_count": len(processed_query.split())
    }
    
    print(f"Processed query: {processed_query}")
    
    return updated_state 