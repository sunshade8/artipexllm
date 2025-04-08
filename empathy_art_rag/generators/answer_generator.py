"""
Answer generator for user queries.

This module is responsible for generating answers to user queries.
"""

from ..models import GraphState
from ..config import DEFAULT_MODEL, LLM_TEMPERATURE

# Placeholder for LLM integration - would be replaced with actual LLM API
def call_llm(prompt: str, model: str = DEFAULT_MODEL, temperature: float = LLM_TEMPERATURE) -> str:
    """
    Call the LLM to generate a response.
    
    Args:
        prompt: The input prompt for the LLM
        model: The LLM model to use
        temperature: Temperature setting for generation
        
    Returns:
        Generated text response
    """
    # TODO: Implement actual LLM API call
    # For now, return mock responses
    
    if "happy" in prompt.lower() or "joy" in prompt.lower():
        return "It's wonderful that you're feeling happy! Happiness is a wonderful emotion that can brighten your day and the days of those around you."
    
    if "sad" in prompt.lower():
        return "I'm sorry to hear you're feeling sad. Remember that it's okay to feel your emotions, and sadness is a natural part of life."
    
    if "angry" in prompt.lower():
        return "I understand feeling angry can be intense. Taking deep breaths and finding a constructive outlet for your feelings might help."
    
    if "fear" in prompt.lower() or "afraid" in prompt.lower():
        return "Fear is a normal response to uncertainty or perceived threats. Remember that many fears are temporary, and you have the strength to face them."
    
    if "art" in prompt.lower() or "artwork" in prompt.lower():
        return "Art is a beautiful way to express emotions and connect with others. Different art styles can evoke different emotional responses."
    
    # Default response
    return "I'm here to help you with your questions and provide support. Would you like to know more about how art can reflect and influence emotions?"

def generate_answer(state: GraphState) -> GraphState:
    """
    Generate the final answer for the user query.
    
    Args:
        state: Current graph state
        
    Returns:
        Updated state with generated answer
    """
    print("==== [Generating Answer] ====")
    
    # Get key state information
    query = state.get("processed_query", "")
    retrieved_context = state.get("retrieved_context", "")
    detected_emotion = state.get("detected_emotion", "")
    art_recommendation = state.get("art_recommendation", None)
    
    # Build the prompt for the LLM
    prompt = f"Query: {query}\n"
    
    if retrieved_context:
        prompt += f"Context: {retrieved_context}\n"
    
    if detected_emotion:
        prompt += f"Detected emotion: {detected_emotion}\n"
    
    # Call the LLM
    base_answer = call_llm(prompt)
    
    # Final answer combines LLM response with any art recommendation
    final_answer = base_answer
    
    # If we have an art recommendation, include it
    if art_recommendation and "artwork" in art_recommendation:
        # The formatted art text is already in the recommendation
        art_text = art_recommendation.get("formatted_text", "")
        
        # Add the art recommendation after the base answer
        if art_text:
            final_answer = f"{base_answer}\n\n--- Artwork Recommendation ---\n{art_text}"
    
    print(f"Generated answer: {final_answer[:100]}...")
    
    # Update state with the answer
    updated_state = {
        **state,
        "answer": final_answer
    }
    
    return updated_state 