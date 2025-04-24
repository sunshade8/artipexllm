"""
Art suggestion node for ArtiTech LangGraph system.

This module provides functions for:
1. Suggesting art recommendations based on detected emotions
2. Handling user responses to these suggestions
3. Initiating the art recommendation process
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from PIL import Image  # Add proper import for PIL Image

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

class ArtSuggestionResponse(BaseModel):
    """Response format for art suggestion."""
    suggestion_accepted: bool = Field(
        description="Whether the user accepted the suggestion for art recommendations"
    )

# --- Define Valid Emotions --- 
# List of emotions the underlying ArtworkRecommender system understands.
# Duplicated here for use within the mapping function.
VALID_ART_SYSTEM_EMOTIONS = [
    "Joy", "Love", "Serenity", "Amusement", "Gratitude", "Hope",
    "Admiration", "Sadness", "Anger", "Fear", "Disgust", "Confusion",
    "Boredom", "Dreamy", "Whimsical", "Mystical", "Spiritual",
    "Nostalgia", "Contemplation", "Wonder", "Awe", "Connectedness",
    "Calmness" # Ensure Calmness is explicitly here if needed
]
VALID_ART_SYSTEM_EMOTIONS_LOWER = [e.lower() for e in VALID_ART_SYSTEM_EMOTIONS]
# Add variations users might say
VALID_ART_SYSTEM_EMOTIONS_LOWER.extend(["calm", "peaceful", "happy", "sad"]) # Add common variations

def suggest_art_node(state: Dict) -> Dict:
    """
    Node for suggesting art recommendations based on detected emotions.
    
    This node:
    1. Checks if user directly asked for art recommendations
    2. Otherwise, if emotion was detected, offers art recommendations
    3. Processes user response to determine if art recommendations should be shown
    
    Args:
        state: Current graph state containing detected emotions and conversation
        
    Returns:
        Updated state with suggestion results and routing information
    """
    print("==== [Art Suggestion Node] ====")
    
    # Extract the necessary information
    question = state.get("question", "")
    generation = state.get("generation", "")
    detected_emotion = state.get("detected_emotion", None)
    conversation_history = state.get("conversation_history", [])
    
    # Initialize variables for decisions
    direct_art_request = False
    offer_art = False
    
    # Check if the user directly asked for art recommendations
    art_request_phrases = [
        "recommend art", "show me art", "suggest art", 
        "art recommendation", "artwork that", "painting that",
        "pieces of art", "art therapy", "therapeutic art",
        "healing art", "art for", "artwork for"
    ]
    
    for phrase in art_request_phrases:
        if phrase.lower() in question.lower():
            direct_art_request = True
            print(f"Direct art request detected: '{phrase}' in '{question}'")
            break
    
    # If direct request, we'll show art recommendations
    if direct_art_request:
        print("User directly requested art recommendations")
        offer_art = True
    # Otherwise, if emotion detected, suggest art recommendations
    elif detected_emotion:
        print(f"Detected emotion: {detected_emotion} - suggesting art recommendations")
        offer_art = True
    else:
        print("No emotion detected and no direct art request - skipping art suggestions")
        offer_art = False
    
    # Prepare a suggestion message if we're offering art
    suggestion_message = ""
    if offer_art:
        if direct_art_request:
            # User directly asked, so we'll provide recommendations right away
            suggestion_message = f"I'll find some artwork that might resonate with you."
        else:
            # User didn't ask directly, so we'll make a suggestion based on emotion
            emotion_category = _get_emotion_category(detected_emotion)
            
            if emotion_category == "Positive Emotions":
                suggestion_message = (
                    f"I notice you're feeling {detected_emotion}. Would you like me to "
                    f"recommend some artwork that could complement this positive emotion?"
                )
            elif emotion_category == "Negative Emotions":
                suggestion_message = (
                    f"I sense you might be feeling {detected_emotion}. Art can sometimes help us "
                    f"process difficult emotions. Would you like me to suggest artwork that might be helpful?"
                )
            else:
                suggestion_message = (
                    f"I notice an emotion of {detected_emotion} in your message. Would you "
                    f"like to see some artwork that relates to this feeling?"
                )
    
    # Update the state with our suggestion information
    updated_state = {
        **state,
        "direct_art_request": direct_art_request,
        "offer_art": offer_art,
        "suggestion_message": suggestion_message,
        "art_suggestion_routing": "show_art" if direct_art_request else "suggest_art"
    }
    
    return updated_state

def art_suggestion_router(state: Dict) -> str:
    """
    Router function to determine next step regarding art suggestions using an LLM.
    
    This router handles both initial interactions and follow-up requests based on 
    the user's message and conversation state.
    
    Returns:
    - "show_art": If the user asks for art, accepts a suggestion, or context implies showing art.
    - "suggest_art": If the context suggests an offer to show art is appropriate.
    - "skip_art": If art is not relevant, user declined, or context suggests skipping.
    """
    print("---- ROUTING ART SUGGESTION ----")
    
    # Extract necessary context from state
    question = state.get("question", "")
    messages = state.get("messages", [])
    offer_art = state.get("offer_art", False) # Was art suggested based on emotion?
    processed_suggestion = state.get("processed_suggestion", False) # Did we already process a yes/no to an offer?
    suggestion_accepted = state.get("suggestion_accepted", False) # Explicitly accepted? (Set by process_art_response)
    
    # Get the last message if available (usually the user's latest)
    last_message_content = ""
    if messages and hasattr(messages[-1], 'content'):
        last_message_content = messages[-1].content
    elif question: # Fallback to using 'question' if messages structure is different
        last_message_content = question

    # Determine the AI's last action (if relevant) - simplified check
    ai_just_offered_art = offer_art and not processed_suggestion and not suggestion_accepted

    # Prepare context for the LLM
    context_summary = f"User's latest message: '{last_message_content}'\\n"
    if ai_just_offered_art:
        context_summary += "Context: The AI just offered to suggest an artwork based on the user's emotion.\\n"
    elif suggestion_accepted:
         context_summary += "Context: The user previously accepted an art suggestion.\\n"
    elif offer_art and processed_suggestion and not suggestion_accepted:
         context_summary += "Context: The user previously declined an art suggestion.\\n"
    else:
        context_summary += "Context: Standard conversation flow, art has not been explicitly offered or discussed in the last turn.\\n"

    # Define the LLM and Prompt
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert router for a conversational AI assistant that can suggest art based on user emotion. 
Your task is to determine the next action based on the user's latest message and the conversational context.
Possible actions are:
1.  'show_art': Use this if the user explicitly asks for art, accepts a previous suggestion, or their message clearly indicates they want to see art now.
2.  'suggest_art': Use this ONLY if the context indicates the AI should proactively OFFER to show art (e.g., after detecting emotion but before asking), AND the user's current message is NOT a direct request for art or a decline.
3.  'skip_art': Use this if the user declines an art suggestion, asks an unrelated question, or the context doesn't warrant suggesting or showing art right now.

Analyze the provided context and user message. Respond ONLY with 'show_art', 'suggest_art', or 'skip_art'."""),
        ("human", f"{context_summary}Based on this, what is the next action?")
    ])

    # Create and invoke the chain
    router_chain = prompt | llm | StrOutputParser()
    
    try:
        decision = router_chain.invoke({}).strip().lower()
        # Add extra stripping for potential quotes returned by LLM
        decision = decision.strip("'\"") 
        print(f"LLM Router Decision (stripped): {decision}") # Log the stripped decision

        # Validate the decision
        if decision in ["show_art", "suggest_art", "skip_art"]:
            # Handle the case where the LLM decides to suggest, but we shouldn't overwrite flags
            # If the decision is 'suggest_art', it implies we need the 'suggest_art' node to run.
            # If the decision is 'show_art', it implies the user wants art *now*.
            # If the decision is 'skip_art', we stop the art flow.
            print(f"ART_SUGGESTION_ROUTER: Returning valid decision: '{decision}' (Type: {type(decision)})")
            return decision
        else:
            print(f"Warning: LLM router returned invalid decision '{decision}'. Defaulting to 'skip_art'.")
            print(f"ART_SUGGESTION_ROUTER: Returning default decision: 'skip_art' (Type: {type('skip_art')})")
            return "skip_art"
            
    except Exception as e:
        print(f"Error during LLM routing: {e}. Defaulting to 'skip_art'.")
        print(f"ART_SUGGESTION_ROUTER: Returning error default decision: 'skip_art' (Type: {type('skip_art')})")
        return "skip_art"

def process_art_suggestion_response(state: Dict) -> Dict:
    """
    Process user's response to art suggestion.
    
    Args:
        state: Current graph state containing suggestion and user response
        
    Returns:
        Updated state with processed response information
    """
    print("==== [Processing Art Suggestion Response] ====")
    
    # Extract the necessary information
    question = state.get("question", "")
    detected_emotion = state.get("detected_emotion", None)
    
    # Initialize LLM for processing
    llm = ChatOpenAI(temperature=0.1, model="gpt-3.5-turbo")
    
    # Define the prompt for determining if user accepted suggestion
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an assistant that determines if a user has accepted an offer for art recommendations.
        Analyze the user's response and determine if they want to see art recommendations.
        Respond with either "yes" if they accepted or "no" if they declined or were ambiguous."""),
        ("human", """Context: The user was asked if they wanted art recommendations related to their emotion.
        User's response: {question}
        Did they accept the offer? (yes/no)""")
    ])
    
    # Create the chain
    chain = prompt | llm | StrOutputParser()
    
    # Determine if the user accepted the suggestion
    try:
        response = chain.invoke({"question": question})
        suggestion_accepted = response.strip().lower() == "yes"
    except Exception as e:
        print(f"Error processing suggestion response: {e}")
        suggestion_accepted = False
    
    print(f"User {'accepted' if suggestion_accepted else 'declined'} art suggestion")
    
    # Update the state with the processed response
    updated_state = {
        **state,
        "suggestion_accepted": suggestion_accepted,
        "processed_suggestion": True,
        "art_suggestion_routing": "show_art" if suggestion_accepted else "skip_art"
    }
    
    return updated_state

# NOTE: Helper functions _map_emotion_to_art_system and _get_emotion_category
# are defined below. If they are used elsewhere, ensure changes are compatible.

# Function to map detected or requested emotions to the specific emotion names
# understood by the ArtworkRecommender system.
# Uses an LLM for fuzzy matching if a direct match isn't found.
def _map_emotion_to_art_system(emotion: str) -> str:
    """Maps a potentially fuzzy emotion string to a valid emotion for the art system."""
    print(f"[_map_emotion_to_art_system] called with: {emotion}")
    
    if not emotion or not isinstance(emotion, str):
        print("[_map_emotion_to_art_system] Invalid input emotion. Returning default.")
        return "Contemplation" # Default fallback
        
    emotion_lower = emotion.strip().lower()
    
    # --- Direct Match Check (Case-Insensitive) ---
    if emotion_lower in VALID_ART_SYSTEM_EMOTIONS_LOWER:
        try:
            # Find the index in the lowercase list to get the correctly capitalized version
            idx = VALID_ART_SYSTEM_EMOTIONS_LOWER.index(emotion_lower)
            # Ensure index is within bounds of the original capitalized list
            if idx < len(VALID_ART_SYSTEM_EMOTIONS):
                 mapped_emotion = VALID_ART_SYSTEM_EMOTIONS[idx]
                 print(f"[_map_emotion_to_art_system] Direct match found: {emotion} -> {mapped_emotion}")
                 return mapped_emotion
            else:
                # Handle edge case where lowercase list might be longer (if extended)
                # Find the first occurrence in the original list that matches lowercase
                 for valid_emotion in VALID_ART_SYSTEM_EMOTIONS:
                      if valid_emotion.lower() == emotion_lower:
                           print(f"[_map_emotion_to_art_system] Direct match found (fallback search): {emotion} -> {valid_emotion}")
                           return valid_emotion
                 # If somehow still not found, proceed to LLM mapping
                 print(f"[_map_emotion_to_art_system] Lowercase match index out of bounds for original list. Proceeding to LLM.")
        except ValueError:
            # Should not happen if emotion_lower is in the list, but handle defensively
            print(f"[_map_emotion_to_art_system] ValueError during direct match lookup. Proceeding to LLM.")
            pass # Proceed to LLM mapping

    # --- LLM-Based Mapping (if no direct match) ---
    print(f"[_map_emotion_to_art_system] No direct match for '{emotion}'. Using LLM for mapping.")
    try:
        from __main__ import llm
    except ImportError:
        print("[_map_emotion_to_art_system] LLM not found in global scope. Cannot perform LLM mapping. Returning default.")
        return "Contemplation" # Default fallback
        
    if not llm:
        print("[_map_emotion_to_art_system] LLM instance is None. Cannot perform LLM mapping. Returning default.")
        return "Contemplation"
        
    valid_emotions_str = ", ".join(VALID_ART_SYSTEM_EMOTIONS)
    # Refined prompt: Focus on finding the direct equivalent or best fit from the valid list.
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are an emotion mapping assistant. 
Your task is to map the given input emotion/concept (like 'happy', 'peaceful', 'relax') to the single most appropriate and equivalent emotion from the following strictly defined list:
**Valid Emotions:** {valid_emotions_str}

If the input is 'calm' or 'relax', map it to 'Calmness'.
If the input is 'happy', map it to 'Joy'.
If the input is 'sad', map it to 'Sadness'.
For other inputs, find the best fit from the list based on meaning.

Respond ONLY with the single chosen emotion name from the valid list, and nothing else."""),
        ("human", f"Input emotion/concept: '{emotion}'\nMost appropriate valid emotion from the list:")
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    try:
        llm_mapped_emotion = chain.invoke({}).strip()
        print(f"[_map_emotion_to_art_system] LLM suggested map: {llm_mapped_emotion}")
        
        # Validate LLM output against the original capitalized list
        if llm_mapped_emotion in VALID_ART_SYSTEM_EMOTIONS:
            print(f"[_map_emotion_to_art_system] LLM mapping successful: {emotion} -> {llm_mapped_emotion}")
            return llm_mapped_emotion
        else:
            print(f"[_map_emotion_to_art_system] LLM returned invalid emotion '{llm_mapped_emotion}'. Returning default.")
            return "Calmness" # Default fallback
            
    except Exception as e:
        print(f"[_map_emotion_to_art_system] Error during LLM mapping: {e}. Returning default.")
        return "Calmness" # Default fallback

# Function to categorize emotion (simple example)
def _get_emotion_category(emotion: str) -> str:
    """Categorize emotion into Positive, Negative, or Other"""
    # This should ideally use the mappings from load_emotion_mappings if available
    # Simplified version for demonstration:
    positive = ["joy", "love", "serenity", "amusement", "gratitude", "hope", "admiration", "happy"]
    negative = ["sadness", "anger", "fear", "disgust", "dread", "confusion", "anxiety", "boredom", "sad"]
    if emotion.lower() in positive:
        return "Positive Emotions"
    elif emotion.lower() in negative:
        return "Negative Emotions"
    else:
        return "Other Emotions"

# ============================================================================
# Testing Function (Keep if used for standalone testing of this script)
# ============================================================================
def test_art_suggestion_flow():
    # ... (original test function implementation) ...
    pass

# Example usage (keep if running this script directly)
if __name__ == "__main__":
    test_art_suggestion_flow() 