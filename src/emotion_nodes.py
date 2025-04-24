# src/emotion_nodes.py

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
import random

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# Assuming emotion_detection.py is also moved or its contents merged
# For now, let's assume it stays in scripts/emotion for the import
# If moved, this import needs adjustment (e.g., from .emotion_detection import ...)
try:
    # Try importing from scripts first (if not moved yet)
    from scripts.emotion.emotion_detection import DETECTABLE_EMOTIONS, EmotionDetect
except ImportError:
    # Fallback if emotion_detection was moved to src (adjust path as needed)
    try:
        from .emotion_detection import DETECTABLE_EMOTIONS, EmotionDetect
    except ImportError:
         print("Warning: Could not import DETECTABLE_EMOTIONS. Using fallback list.")
         DETECTABLE_EMOTIONS = ["happy", "sad", "angry", "calm", "neutral", "contemplative"] 
         # Define a fallback EmotionDetect or handle its absence
         class EmotionDetect(BaseModel):
             binary_score: str
             emotion_name: str
             emotion_confidence: float

# Pydantic model used for the initial check: Can the LLM potentially detect
# an emotion from the predefined list in the user's message?
class EmotionGrader(BaseModel):
    """Detect user's emotion during the interaction"""
    binary_score: str = Field(
        description=f"If you can detect user's emotion among the list below, say 'yes', otherwise say 'no': {', '.join(DETECTABLE_EMOTIONS)}"
    )

# LangGraph node responsible for detecting the user's emotion based on their input.
# It employs a multi-step process for robustness:
# 1. Initial check: Ask the LLM if an emotion *might* be present from a predefined list.
# 2. Specific identification: If yes, ask the LLM to name the specific emotion from the list.
# 3. Fallback matching: If steps 1 or 2 fail or yield invalid results, ask the LLM to find the *closest* match from the list.
# 4. Default assignment: If all else fails, assign a default neutral/calm emotion.
def emotion_detect_node(state: Dict) -> Dict:
    """
    Node for emotion detection with a two-step workflow:
    1. Check if model can detect user's emotion in the list
    2. If not, use API to match to closest emotion
    
    Args:
        state: Current graph state containing user's question and LLM generation
        
    Returns:
        Updated state with emotion detection results
    """
    print("==== [Node: emotion_detect_node (from src)] ====")
    
    # Extract the necessary information
    question = state.get("question", "")
    # generation = state.get("generation", "") # Generation might not be needed here
    
    # Initialize LLM. Attempts to use a globally defined 'llm' (e.g., from a notebook environment)
    # or initializes a default one if not found.
    try:
        from __main__ import llm # Try importing from notebook scope
    except ImportError:
        print("LLM not found in notebook scope, initializing default in emotion_detect_node.")
        llm = ChatOpenAI(temperature=0.1, model="gpt-3.5-turbo") 
    
    # --- Step 1: Initial Check --- 
    # Use the EmotionGrader model to get a simple 'yes'/'no' if an emotion from the list is detectable.
    detection_prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are an emotion detection assistant.
        Analyze if the user's message contains any emotions from this list: {', '.join(DETECTABLE_EMOTIONS)}
        Respond with 'yes' if you can detect an emotion from the list, otherwise respond with 'no'."""),
        ("human", f"User message: {question}")
    ])
    
    # Create the emotion grader
    structured_llm_answer = llm.with_structured_output(EmotionGrader)
    emotion_check = detection_prompt | structured_llm_answer
    
    # Invoke the chain to perform the initial check.
    can_detect = False # Default
    try:
        emotion_result = emotion_check.invoke({})
        can_detect = emotion_result.binary_score.lower() == "yes"
    except Exception as e:
        print(f"Error during emotion detection check: {e}")
        can_detect = False
    
    # --- Step 2: Specific Emotion Identification --- 
    # If the initial check was positive, try to identify the *exact* emotion from the list.
    detected_emotion = None
    confidence = 0.0
    
    if can_detect:
        print("Attempting to identify specific emotion.")
        emotion_prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are an emotion detection assistant.
            Identify the most prominent emotion in the user's message from this list: {', '.join(DETECTABLE_EMOTIONS)}
            Return only the single best matching emotion from the list, with no other text."""),
            ("human", f"User message: {question}")
        ])
        
        emotion_chain = emotion_prompt | llm | StrOutputParser()
        try:
            identified_emotion = emotion_chain.invoke({}).strip().lower()
            # Validate that the emotion is in our list
            if identified_emotion in DETECTABLE_EMOTIONS:
                detected_emotion = identified_emotion
                confidence = 0.9
                print(f"✓ Directly Detected: {detected_emotion} (Confidence: {confidence:.2f})")
            else:
                print(f"✗ LLM identified '{identified_emotion}' which is not in {DETECTABLE_EMOTIONS}. Will try matching.")
                detected_emotion = None # Reset if not valid
        except Exception as e:
            print(f"Error identifying specific emotion: {e}")
            detected_emotion = None
    else:
        print("Emotion not directly detected from list.")

    # --- Step 3: Fallback Matching --- 
    # If no valid emotion was identified in Step 2 (or Step 1 was 'no'),
    # ask the LLM to find the *closest* matching emotion from the list.
    if detected_emotion is None:
        print("Using LLM to match closest emotion from list.")
        emotion_match_prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are an emotion matching API.
            Given a user message, match it to the closest emotion from this list: {', '.join(DETECTABLE_EMOTIONS)}
            Even if the emotion is subtle or implied, try to find the best match.
            Return only the single best matching emotion from the list, with no other text."""),
            ("human", f"User message: {question}")
        ])
        
        emotion_match_chain = emotion_match_prompt | llm | StrOutputParser()
        try:
            matched_emotion = emotion_match_chain.invoke({}).strip().lower()
            if matched_emotion in DETECTABLE_EMOTIONS:
                detected_emotion = matched_emotion
                confidence = 0.7  # Lower confidence for LLM-matched emotions
                print(f"✓ LLM Matched: {detected_emotion} (Confidence: {confidence:.2f})")
            else:
                print(f"✗ LLM match '{matched_emotion}' not in {DETECTABLE_EMOTIONS}. Falling back to default.")
                detected_emotion = None # Reset again
        except Exception as e:
            print(f"Error matching emotion with LLM: {e}")
            detected_emotion = None

    # --- Step 4: Default Fallback --- 
    # If no emotion could be detected or matched, assign a default (neutral/calm).
    if detected_emotion is None:
        detected_emotion = random.choice([e for e in DETECTABLE_EMOTIONS if e in ["calm", "neutral", "contemplative"]])
        confidence = 0.3
        print(f"✓ Using Default Fallback: {detected_emotion} (Confidence: {confidence:.2f})")

    # --- Format Output --- 
    # Construct the output dictionary, including the structured EmotionDetect result if possible.
    # Handles potential import errors for the EmotionDetect class gracefully.
    try:
        emotion_output = EmotionDetect(
            binary_score="yes", # Always yes if we return *any* emotion
            emotion_name=detected_emotion,
            emotion_confidence=confidence
        )
        emotion_result_dict = emotion_output.dict()
    except NameError: # If EmotionDetect class wasn't imported correctly
        emotion_result_dict = {
            "binary_score": "yes",
            "emotion_name": detected_emotion,
            "emotion_confidence": confidence
        }
    except Exception as e:
         print(f"Error creating EmotionDetect object: {e}")
         emotion_result_dict = {
             "binary_score": "error",
             "emotion_name": "error",
             "emotion_confidence": 0.0
         }

    # Return the fields that need to be updated in the graph's state.
    update_fields = {
        "detected_emotion": detected_emotion,
        "emotion_confidence": confidence,
        "emotion_result": emotion_result_dict # Include the structured result if needed downstream
    }
    
    return update_fields 

# # Placeholder for art selection based on negative emotion - REMOVED
# def select_art_for_negative_emotion(state: Dict) -> Dict:
#     """
#     Selects an art piece based on the detected negative emotion.
#     (Placeholder implementation)
#     """
#     print("==== [Node: select_art_for_negative_emotion] ====")
#     detected_emotion = state.get("detected_emotion")
#     print(f"Received negative emotion: {detected_emotion}")
#
#     # Placeholder logic: Just select a default "negative art" identifier
#     # TODO: Implement actual art selection logic (e.g., based on emotion mapping)
#     selected_art_id = "default_sad_art" # Example
#     if detected_emotion == "angry":
#         selected_art_id = "default_angry_art"
#
#     print(f"Selected art ID: {selected_art_id}")
#
#     return {"selected_art_id": selected_art_id}

# # Function to handle branching based on emotion (Modified) - COMMENTED OUT (replaced by art_suggestion_router logic)
# def handle_emotion_response(state: Dict) -> str:
#     """
#     Determines the next node based on the overall emotion category.
#     Routes negative emotions to art selection.
#     """
#     print("==== [Node: handle_emotion_response] ====")
#     emotion = state.get("detected_emotion") # Using detected_emotion now
#     confidence = state.get("emotion_confidence", 0.0)
#
#     # Simple categorization (example threshold)
#     # This logic might need refinement based on how DETECTABLE_EMOTIONS are defined
#     if emotion in ["happy"]: # Add other positive emotions if any
#         print(f"Emotion '{emotion}' classified as positive. Transitioning to generate_positive_response")
#         return "generate_positive_response"
#     elif emotion in ["sad", "angry"]: # Add other negative emotions if any
#         print(f"Emotion '{emotion}' classified as negative. Transitioning to select_art_for_negative_emotion")
#         return "select_art_for_negative_emotion" # <<< CHANGE HERE
#     else: # neutral, calm, contemplative, etc.
#         print(f"Emotion '{emotion}' classified as neutral/other. Transitioning to generate_neutral_response")
#         return "generate_neutral_response" 