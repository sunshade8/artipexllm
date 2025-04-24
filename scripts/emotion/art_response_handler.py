"""
Handler for processing art-related user responses in follow-up messages.

This module helps process multi-turn conversations where users might
directly ask for art recommendations in follow-up messages.
"""

from typing import Dict, List, Optional, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# Import the valid emotion list needed for the intent analysis prompt
try:
    # Assumes this script is run where art_suggestion is importable
    from .art_suggestion import VALID_ART_SYSTEM_EMOTIONS 
except ImportError:
    print("Warning (art_response_handler): Could not import VALID_ART_SYSTEM_EMOTIONS. Using fallback list.")
    # Fallback list - ensure it matches the one in art_suggestion.py
    VALID_ART_SYSTEM_EMOTIONS = [
        "Joy", "Love", "Serenity", "Amusement", "Gratitude", "Hope",
        "Admiration", "Sadness", "Anger", "Fear", "Disgust", "Confusion",
        "Boredom", "Dreamy", "Whimsical", "Mystical", "Spiritual",
        "Nostalgia", "Contemplation", "Wonder", "Awe", "Connectedness",
        "Calmness"
    ]

# ============================================================================
# Process Art Request
# ============================================================================
def process_art_request(state: Dict) -> Dict:
    """
    Process a direct art request from a user's follow-up message.
    
    This node:
    1. Detects if the current message contains a direct art request
    2. Extracts emotion information if mentioned in the request using the passed LLM
    3. Updates state for art recommendation
    
    Args:
        state: Current graph state containing user message
        
    Returns:
        Updated state ready for art recommendation
    """
    print("==== [Node: Processing Art Request] ====")
    
    # --- Access LLM from global scope ---
    try:
        from __main__ import llm
    except ImportError:
        print("Error: LLM instance not found in global scope (__main__). Cannot process art request effectively.")
        # Handle the error appropriately - maybe return state with an error flag or use a fallback?
        # For now, let's proceed but expect potential failure or use default logic.
        llm = None 
    # -----------------------------------

    # Extract the necessary information
    question = state.get("question", "")
    # Get originally detected emotion if available, might be None
    original_detected_emotion = state.get("detected_emotion") 
    
    # Prepare default state update flags for a direct request
    base_update = {
        "direct_art_request": True,
        "offer_art": True,
        "processed_suggestion": True,
        "art_suggestion_accepted": True
    }
    
    if not llm:
        print("Warning: LLM not available. Using existing detected emotion or default.")
        updated_state = {
            **state,
            **base_update,
            "detected_emotion": original_detected_emotion or "joy" # Use existing or fallback
        }
        return updated_state
    
    # Check if specific emotion is mentioned in the art request
    specific_emotion_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an assistant that analyzes art requests.
        Determine if the user has requested artwork related to a specific emotion.
        If they have, extract that emotion. If not, respond with "none".
        
        Only return emotions from this list: joy, happiness, love, serenity, calm, peaceful, 
        amusement, gratitude, hope, admiration, sadness, grief, anger, fear, anxiety, disgust, 
        dread, confusion, boredom, loneliness, dreamy, curious, mystical, spiritual, whimsical, 
        creative, nostalgia, contemplative, wonder, awe, connectedness.
        
        Return only the single emotion name or "none" with no other text."""),
        ("human", f"User message: {question}")
    ])
    
    # Use the passed LLM
    emotion_chain = specific_emotion_prompt | llm | StrOutputParser()
    
    extracted_emotion_val = None
    try:
        extracted_emotion_val = emotion_chain.invoke({}).strip().lower()
        if extracted_emotion_val != "none" and len(extracted_emotion_val) > 2:
            print(f"Extracted emotion from art request: {extracted_emotion_val}")
            original_detected_emotion = extracted_emotion_val # Overwrite if specific emotion found
        else:
            print("No specific emotion extracted from request.")
    except Exception as e:
        print(f"Error extracting emotion from request: {e}")
    
    # If no specific emotion was extracted or originally detected, try to infer a default
    if not original_detected_emotion:
        print("Attempting to determine default emotion...")
        default_emotion_prompt = ChatPromptTemplate.from_messages([
            ("system", """Based on the user's art request, suggest ONE appropriate default emotion 
            from this list: joy, calm, hope, wonder, serenity.
            Return only the emotion name with no other text."""),
            ("human", f"User art request: {question}")
        ])
        
        # Use the passed LLM again
        default_chain = default_emotion_prompt | llm | StrOutputParser()
        
        try:
            default_emotion_val = default_chain.invoke({}).strip().lower()
            # Basic validation for the default emotion
            if default_emotion_val in ["joy", "calm", "hope", "wonder", "serenity"]:
                original_detected_emotion = default_emotion_val
                print(f"Using default emotion for art request: {original_detected_emotion}")
            else:
                print(f"LLM provided invalid default emotion '{default_emotion_val}'. Using fallback.")
                original_detected_emotion = "joy" # Absolute fallback
        except Exception as e:
            print(f"Error determining default emotion: {e}")
            original_detected_emotion = "joy" # Absolute fallback
    
    # Determine the primary emotion context (prefer mentioned in request, fallback to original detection)
    primary_emotion_context = extracted_emotion_val or original_detected_emotion
    print(f"Primary emotion context for intent analysis: {primary_emotion_context}")

    # --- Step 2: Analyze Intent & Determine Target Emotion --- 
    # Use LLM to determine if the user wants to *see* the primary emotion 
    # or *feel different* from it, and select the final target emotion.
    final_target_emotion = primary_emotion_context # Default to primary context

    if primary_emotion_context: # Only run intent analysis if we have some emotion context
        valid_emotions_str = ", ".join(VALID_ART_SYSTEM_EMOTIONS)
        intent_prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are an intent analysis expert for an art recommendation system. 
Analyze the user's request to understand if they want art that **expresses** their current feeling OR art designed to make them **feel better/different**.

User's Mentioned/Contextual Emotion: {primary_emotion_context}
Valid Art System Emotions: {valid_emotions_str}

1. If the request is simply to see art about '{primary_emotion_context}' (e.g., "show me art about sadness"), respond with: **EXPRESS:{primary_emotion_context}**
2. If the request implies wanting to feel *better* or *different* from '{primary_emotion_context}' (e.g., "art to make me feel better from sadness", "art to lift my mood"), suggest a suitable contrasting/positive emotion from the Valid list. Respond with: **COUNTER:[Suggested Emotion from Valid List]** (e.g., COUNTER:Joy, COUNTER:Hope, COUNTER:Calmness).

Choose the BEST single suggested emotion from the valid list if countering. Consider these examples:
- Sadness -> Hope, Joy, Serenity
- Anger -> Calmness, Serenity
- Anxiety/Fear -> Calmness, Serenity, Hope
- Boredom -> Amusement, Wonder, Joy

Respond ONLY in the format EXPRESS:[Emotion] or COUNTER:[Emotion]."""),
            ("human", f"User request: \"{question}\"\nAnalysis (EXPRESS or COUNTER):")
        ])
        
        intent_chain = intent_prompt | llm | StrOutputParser()
        
        try:
            intent_result = intent_chain.invoke({}).strip()
            print(f"Intent analysis result: {intent_result}")
            
            if intent_result.startswith("COUNTER:"):
                counter_emotion = intent_result.split(":", 1)[1]
                # Validate the suggested counter emotion
                if counter_emotion in VALID_ART_SYSTEM_EMOTIONS:
                    final_target_emotion = counter_emotion
                    print(f"Intent is to counter. Using target emotion: {final_target_emotion}")
                else:
                    print(f"Warning: LLM suggested invalid counter emotion '{counter_emotion}'. Falling back to primary context: {primary_emotion_context}")
                    final_target_emotion = primary_emotion_context # Fallback
            elif intent_result.startswith("EXPRESS:"):
                 # Keep the originally identified primary emotion
                 final_target_emotion = primary_emotion_context
                 print(f"Intent is to express. Using target emotion: {final_target_emotion}")
            else:
                 print(f"Warning: Unexpected intent analysis format '{intent_result}'. Falling back to primary context: {primary_emotion_context}")
                 final_target_emotion = primary_emotion_context # Fallback
                 
        except Exception as e:
            print(f"Error during intent analysis: {e}. Falling back to primary context: {primary_emotion_context}")
            final_target_emotion = primary_emotion_context # Fallback
    else:
        # If no primary emotion context could be determined initially, use a generic positive default
        print("No primary emotion context found. Using default target: Joy")
        final_target_emotion = "Joy"

    # --- Step 3: Update State --- 
    # Update the state for art recommendation using the final target emotion
    updated_state = {
        **state,
        **base_update,
        # IMPORTANT: Set detected_emotion to the FINAL target emotion determined by intent analysis
        "detected_emotion": final_target_emotion, 
        "suggestion_message": "" # Ensure suggestion message is cleared
    }
    
    print(f"Ready to recommend art for final target emotion: {final_target_emotion}")
    return updated_state

# ============================================================================
# Process Art Request (Moved to src/art_processing_nodes.py)
# ============================================================================
# def process_art_request(state: Dict, llm: Any) -> Dict:
#     """
#     Process a direct art request from a user's follow-up message.
#     ...
#     """
#     print("==== [Node: Processing Art Request] ====")
#     # ... (original implementation)
#     pass 