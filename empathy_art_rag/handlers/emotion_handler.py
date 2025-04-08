"""
Emotion detection and art recommendation decision handlers.
"""

from ..models import GraphState
import re
from typing import Dict, List, Optional, Any

# Emotions that can be detected and mapped to artwork recommendations
DETECTABLE_EMOTIONS = [
    # Positive emotions
    "joy", "happiness", "love", "serenity", "calm", "peaceful", 
    "amusement", "gratitude", "hope", "admiration",
    
    # Negative emotions
    "sadness", "grief", "anger", "fear", "anxiety", "disgust", 
    "dread", "confusion", "boredom", "loneliness",
    
    # Imaginative emotions
    "dreamy", "curious", "mystical", "spiritual", "whimsical", "creative",
    
    # Blended/Neutral emotions
    "nostalgia", "contemplative", "wonder", "awe", "connectedness"
]

# Korean emotion keywords (for multilingual support)
KOREAN_EMOTION_MAP = {
    "기쁨": "joy",
    "행복": "happiness",
    "사랑": "love",
    "평온": "serenity",
    "차분": "calm",
    "평화": "peaceful",
    "즐거움": "amusement",
    "감사": "gratitude",
    "희망": "hope",
    "존경": "admiration",
    "슬픔": "sadness",
    "비통": "grief",
    "분노": "anger",
    "화남": "anger",
    "두려움": "fear",
    "불안": "anxiety",
    "역겨움": "disgust",
    "혼란": "confusion",
    "지루함": "boredom",
    "외로움": "loneliness",
    "꿈같은": "dreamy",
    "호기심": "curious",
    "신비": "mystical",
    "영적": "spiritual",
    "기발한": "whimsical",
    "창의적": "creative",
    "향수": "nostalgia",
    "사색적": "contemplative",
    "경이": "wonder",
    "경외": "awe",
    "연결": "connectedness"
}

def detect_emotion_keywords(text: str) -> List[str]:
    """
    Detect emotions in text using keyword matching, with multilingual support.
    
    Args:
        text: Text to analyze for emotion keywords
        
    Returns:
        List of detected emotions
    """
    text_lower = text.lower()
    found_emotions = []
    
    # Check for English emotion keywords
    for emotion in DETECTABLE_EMOTIONS:
        if emotion in text_lower:
            found_emotions.append(emotion)
    
    # Check for Korean emotion keywords
    for kr_emotion, en_emotion in KOREAN_EMOTION_MAP.items():
        if kr_emotion in text:
            found_emotions.append(en_emotion)
    
    return found_emotions

def detect_emotion(state: GraphState) -> GraphState:
    """
    Detect emotions in user question and add to state.
    Uses simple keyword matching first, then LLM analysis for complex cases.
    
    Args:
        state: Current graph state
        
    Returns:
        Updated state with detected emotion
    """
    question = state.get("question", state.get("query", ""))
    conversation_history = state.get("conversation_history", [])
    
    print("==== [Detecting Emotions] ====")
    
    # Check for explicit art requests first
    art_request_patterns = [
        r"(?i)recommend.*art",
        r"(?i)show me.*art",
        r"(?i)suggest.*artwork",
        r"(?i)find.*painting",
        r"(?i)art.*recommend",
        r"(?i)artwork.*help",
        r"(?i)healing.*art",
        r"(?i)therapeutic.*art",
        # Korean patterns
        r"작품.*추천",
        r"그림.*보여",
        r"예술.*추천",
        r"미술.*치유"
    ]
    
    is_art_request = any(re.search(pattern, question) for pattern in art_request_patterns)
    
    # Simple keyword-based emotion detection
    detected_emotions = detect_emotion_keywords(question)
    
    # If no emotions detected by keywords, try inference from context
    if not detected_emotions and conversation_history:
        # Use the conversation agent to infer emotion (simplified for this example)
        # In actual implementation, would use LLM with prompt
        last_few_messages = conversation_history[-3:] if len(conversation_history) >= 3 else conversation_history
        combined_text = " ".join([msg.get("content", "") for msg in last_few_messages])
        detected_emotions = detect_emotion_keywords(combined_text)
    
    # Use most specific emotion if multiple detected
    detected_emotion = detected_emotions[0] if detected_emotions else None
    
    # Determine if we should offer art 
    if is_art_request:
        # Direct art request - automatically offer art
        print(f"Art request detected: {question}")
        if not detected_emotion:
            # If no emotion detected but user requests art, default to a neutral emotion
            detected_emotion = "calm"
        
        return {
            **state,
            "detected_emotion": detected_emotion,
            "offer_art": True
        }
    elif detected_emotion:
        print(f"Detected emotion: {detected_emotion}")
        # We detected an emotion but no direct art request
        # Will decide whether to offer art in the next node
        return {
            **state,
            "detected_emotion": detected_emotion
        }
    else:
        # No emotion detected
        print("No specific emotion detected")
        return {
            **state,
            "detected_emotion": None
        }

def should_offer_art(state: GraphState) -> GraphState:
    """
    Decides whether to offer art recommendation based on context.
    
    Args:
        state: Current graph state
        
    Returns:
        Updated state with art offering decision
    """
    print("==== [Checking If Should Offer Art] ====")
    
    detected_emotion = state.get("detected_emotion")
    conversation_history = state.get("conversation_history", [])
    
    # Default to not offering art
    offer_art = False
    
    # If explicit art offering flag is set, always offer
    if state.get("offer_art", False):
        offer_art = True
        print("Explicit art request detected - will offer art")
    # If no emotion detected, skip art
    elif not detected_emotion:
        offer_art = False
        print("No emotion detected - skipping art recommendation")
    else:
        # Check if we've recently offered art (avoid being repetitive)
        recent_messages = conversation_history[-5:] if len(conversation_history) >= 5 else conversation_history
        art_recently_offered = False
        
        for message in recent_messages:
            if isinstance(message, dict) and message.get("role") == "assistant" and "artwork" in message.get("content", "").lower():
                art_recently_offered = True
                break
        
        if art_recently_offered:
            offer_art = False
            print("Art recently offered - skipping to avoid repetition")
        else:
            # Strong emotions warrant art recommendations more often
            strong_emotions = ["grief", "anger", "anxiety", "joy", "sadness", "fear", "love"]
            if detected_emotion in strong_emotions:
                # 80% chance of offering art for strong emotions
                import random
                offer_art = random.random() < 0.8
                print(f"Strong emotion detected ({detected_emotion}) - {'will' if offer_art else 'will not'} offer art")
            else:
                # 40% chance of offering art for milder emotions
                import random
                offer_art = random.random() < 0.4
                print(f"Mild emotion detected ({detected_emotion}) - {'will' if offer_art else 'will not'} offer art")
    
    # Add routing result to state
    updated_state = {
        **state,
        "offer_art": offer_art,
        "check_should_offer_art_routing": 0 if offer_art else 1  # 0=offer art, 1=skip art
    }
    
    return updated_state 