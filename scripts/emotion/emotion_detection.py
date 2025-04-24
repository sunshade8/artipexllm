"""
Emotion detection module for ArtiTech LangGraph system.

This module provides functions and components for:
1. Detecting emotions in user messages
2. Determining when to recommend artwork based on emotions
3. Supporting multilingual emotion detection (English and Korean)
"""

from typing import Dict, List, Optional, Any, TypedDict
from pydantic import BaseModel, Field
import re
import random

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# Emotions that can be detected and mapped to artwork recommendations
DETECTABLE_EMOTIONS = [
    # Positive emotions - generally uplifting feelings
    "joy", "happiness", "love", "serenity", "calm", "peaceful", 
    "amusement", "gratitude", "hope", "admiration",
    
    # Negative emotions - feelings that may benefit from supportive responses
    "sadness", "grief", "anger", "fear", "anxiety", "disgust", 
    "dread", "confusion", "boredom", "loneliness",
    
    # Imaginative emotions - feelings that connect to creativity and inspiration
    "dreamy", "curious", "mystical", "spiritual", "whimsical", "creative",
    
    # Blended/Neutral emotions - more complex or balanced emotional states
    "nostalgia", "contemplative", "wonder", "awe", "connectedness"
]

# ============================================================================
# Data Models
# ============================================================================

class EmotionDetect(BaseModel):
    """Detect user's emotion during the interaction"""
    
    binary_score: str = Field(
        description=f"If you can detect user's emotion among the list below, say 'yes', otherwise say 'no': {', '.join(DETECTABLE_EMOTIONS)}"
    )
    
    emotion_name: str = Field(
        description="The specific emotion detected from the list above, if binary_score is 'yes'. Otherwise, leave empty."
    )
    
    emotion_confidence: float = Field(
        default=0.0,
        description="Confidence level in emotion detection (0.0 to 1.0)"
    )


def detect_emotion_keywords(text: str) -> List[str]:
    """
    Detect emotions in text using keyword matching, with multilingual support.
    
    Args:
        text: Text to analyze for emotion keywords
        
    Returns:
        List of detected emotions (English normalized)
    """
    text_lower = text.lower()
    found_emotions = []
    
    # Check for English emotion keywords
    for emotion in DETECTABLE_EMOTIONS:
        if emotion in text_lower:
            found_emotions.append(emotion)
    
    # Special case for "happy" which should map to "happiness"
    if "happy" in text_lower and "happiness" not in found_emotions:
        found_emotions.append("happiness")
    
    # Special case for "mad" which should map to "anger"
    if "mad" in text_lower and "anger" not in found_emotions:
        found_emotions.append("anger")
    
    return found_emotions


def map_emotion_with_llm(text: str, llm) -> Optional[str]:
    """
    Use an LLM to map text to a recognized emotion.
    
    Args:
        text: User's text to analyze
        llm: Language model to use
        
    Returns:
        Mapped emotion from predefined list, or None if mapping failed
    """
    emotions_list = ", ".join(DETECTABLE_EMOTIONS)
    
    mapping_emotion_prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are an emotion detection assistant. Map emotions to predefined categories.
        Map the emotion in the following text to one of these predefined emotions:
        {emotions_list}
        
        Return only the single best matching emotion from the list above, with no other text.
        If no emotion can be confidently detected, respond with "none"."""),
        ("human", f"{text}")
    ])
    
    chain = mapping_emotion_prompt | llm | StrOutputParser()

    try:
        mapped_emotion = chain.invoke({}).strip().lower()
        
        if mapped_emotion in DETECTABLE_EMOTIONS:
            print(f"LLM detected emotion: {mapped_emotion}")
            return mapped_emotion
        elif mapped_emotion != "none":
            # Try to find closest match in case of slight mismatch
            for emotion in DETECTABLE_EMOTIONS:
                if emotion in mapped_emotion:
                    print(f"LLM detected emotion (partial match): {emotion}")
                    return emotion
        
        return None
    except Exception as e:
        print(f"Error using LLM for emotion mapping: {e}")
        return None