"""
Emotion detection node for ArtiTech LangGraph system.

This module provides a function for detecting emotions in user messages
with a two-step approach:
1. Check if model can detect user's emotion in the list
2. If not, use API to match to closest emotion
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
import random

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# Import the emotion list from emotion_detection.py
from .emotion_detection import DETECTABLE_EMOTIONS, EmotionDetect

class EmotionGrader(BaseModel):
    """Detect user's emotion during the interaction"""

    binary_score: str = Field(
        description=f"If you can detect user's emotion among the list below, say 'yes', otherwise say 'no': {', '.join(DETECTABLE_EMOTIONS)}"
    )

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
    print("==== [Emotion Detection Node] ====")
    
    # Extract the necessary information
    question = state.get("question", "")
    generation = state.get("generation", "")
    
    # Initialize LLM
    llm = ChatOpenAI(temperature=0.1, model="gpt-3.5-turbo")
    
    # Step 1: Check if model can detect user's emotion from the predefined list
    detection_prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are an emotion detection assistant.
        Analyze if the user's message contains any emotions from this list: {', '.join(DETECTABLE_EMOTIONS)}
        Respond with 'yes' if you can detect an emotion from the list, otherwise respond with 'no'."""),
        ("human", f"User message: {question}")
    ])
    
    # Create the emotion grader
    structured_llm_answer = llm.with_structured_output(EmotionGrader)
    emotion_check = detection_prompt | structured_llm_answer
    
    # Check if emotion can be detected
    try:
        emotion_result = emotion_check.invoke({})
        can_detect = emotion_result.binary_score == "yes"
    except Exception as e:
        print(f"Error during emotion detection: {e}")
        can_detect = False
    
    # Step 2: If emotion can be detected, identify the specific emotion
    detected_emotion = None
    confidence = 0.0
    
    if can_detect:
        # Get the specific emotion
        emotion_prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are an emotion detection assistant.
            Identify the most prominent emotion in the user's message from this list: {', '.join(DETECTABLE_EMOTIONS)}
            Return only the single best matching emotion from the list, with no other text."""),
            ("human", f"User message: {question}")
        ])
        
        emotion_chain = emotion_prompt | llm | StrOutputParser()
        try:
            detected_emotion = emotion_chain.invoke({}).strip().lower()
            # Validate that the emotion is in our list
            if detected_emotion in DETECTABLE_EMOTIONS:
                confidence = 0.9
                print(f"Detected emotion: {detected_emotion} (confidence: {confidence:.2f})")
            else:
                detected_emotion = None
                print(f"Detected emotion '{detected_emotion}' not in our list, will try matching API")
        except Exception as e:
            print(f"Error identifying specific emotion: {e}")
            detected_emotion = None
    
    # Step 3: If no emotion detected, use API to match to closest emotion
    if not detected_emotion:
        print("No emotion detected directly, using API to match closest emotion")
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
                confidence = 0.7  # Lower confidence for API-matched emotions
                print(f"API matched emotion: {detected_emotion} (confidence: {confidence:.2f})")
            else:
                # As a fallback, select a default emotion
                detected_emotion = random.choice(["calm", "neutral", "contemplative"])
                confidence = 0.3
                print(f"Using default emotion: {detected_emotion} (confidence: {confidence:.2f})")
        except Exception as e:
            print(f"Error matching emotion with API: {e}")
            detected_emotion = "neutral"  # Default fallback
            confidence = 0.1
    
    # Build structured emotion detection result
    emotion_result = EmotionDetect(
        binary_score="yes" if detected_emotion else "no",
        emotion_name=detected_emotion if detected_emotion else "",
        emotion_confidence=confidence
    )
    
    # Update the graph state with detection results
    updated_state = {
        **state, 
        "detected_emotion": detected_emotion,
        "emotion_confidence": confidence,
        "emotion_result": emotion_result.dict()
    }
    
    return updated_state

#" Test function
def test_emotion_detect_node():
    """Test the emotion_detect_node with example inputs."""
    test_inputs = [
        "I'm feeling really happy today!", 
        "I'm so angry about what happened at work",
        "I'm not sure how I feel today",
        "What's the weather like today?",
        "Can you help me solve this math problem?"
    ]
    
    for text in test_inputs:
        print("\n" + "-"*50)
        print(f"Testing with: '{text}'")
        
        # Create test state
        test_state = {
            "question": text,
            "generation": "This is a test response."
        }
        
        # Run emotion detection
        result_state = emotion_detect_node(test_state)
        
        # Print results
        print("\nResults:")
        print(f"Detected emotion: {result_state.get('detected_emotion')}")
        print(f"Confidence: {result_state.get('emotion_confidence')}")
        print(f"Emotion result: {result_state.get('emotion_result')}")

if __name__ == "__main__":
    test_emotion_detect_node() 