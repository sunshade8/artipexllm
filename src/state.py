from typing import List, Dict, Any, Optional, Tuple
from typing_extensions import TypedDict
from langchain_core.documents import Document

# Defines the structure of the state object that is passed between nodes in the LangGraph.
# This TypedDict holds all the information the graph needs to track during its execution.
class GraphState(TypedDict):
    """Data model indicating the GraphState (Enhanced)

    Attributes:
        question: The current user question.
        documents: List of retrieved documents relevant to the question.
        generation: The generated response string from the LLM.
        conversation_history: List of previous interactions (question-answer pairs).
        detected_emotion: The primary emotion detected in the user's message (e.g., 'sadness').
        emotion_confidence: Confidence score for the detected emotion.
        primary_user_need: Inferred user need based on emotion/context (e.g., 'comfort'). # Added for context
        art_suggestion_offered: Flag indicating if an art suggestion was proactively offered. # Renamed from suggest_art for clarity
        art_suggestion_accepted: Flag indicating if the user accepted the art suggestion.
        art_recommendations: Optional[List[Dict]] = None # 생성된 예술 추천 목록
        art_display_message: Optional[str] = None # 예술 추천 표시 메시지
        direct_art_request: bool = False # 직접적인 예술 요청 여부
        processed_suggestion: bool = False # Process suggestion response
        offer_art: bool = False # If we should offer art based on emotion
        suggestion_message: str = "" # Message asking the user if they want art
        art_suggestion_routing: Optional[str] = None # Routing decision after suggestion node
        skip_art: bool = False # Whether to skip art suggestion/recommendation
    """

    # --- Core RAG State Attributes ---
    # These attributes manage the fundamental Request-Augmented Generation process.
    question: str                    # The user's most recent question.
    documents: List[Document]        # Documents retrieved relevant to the question.
    generation: str                  # The LLM's generated response.
    conversation_history: List[Dict] = [] # History of user questions and assistant answers [{question: str, answer: str}]. Initialized empty.

    # --- Emotion and Art Recommendation State Attributes ---
    # These attributes track the emotional context and the art recommendation process.
    detected_emotion: Optional[str] = None     # The primary emotion detected (e.g., 'happy', 'sad').
    emotion_confidence: Optional[float] = None # Confidence score (0.0-1.0) for the detected emotion.
    primary_user_need: Optional[str] = None    # Inferred user need driving the interaction (e.g., 'comfort', 'inspiration').

    # --- Art Suggestion Flow Control Attributes ---
    # These boolean flags and strings manage the conditional flow related to offering and showing art.
    direct_art_request: bool = False           # True if the user explicitly asked for art.
    offer_art: bool = False                    # True if the system decides to proactively offer art based on emotion.
    suggestion_message: str = ""               # The message generated asking the user if they want an art suggestion.
    processed_suggestion: bool = False         # True once the user's response ('yes'/'no') to the offer has been processed.
    art_suggestion_accepted: Optional[bool] = None # User's response: True for 'yes', False for 'no'.
    art_suggestion_routing: Optional[str] = None # Internal routing decision after processing suggestion ('show_art' or 'skip_art').
    skip_art: bool = False                     # Master flag to bypass the entire art suggestion/recommendation flow.

    # --- Art Recommendation Output Attributes ---
    # These attributes store the results and status of the art recommendation generation and display.
    art_recommendations: Optional[List[Dict]] = None # List of recommended artworks, typically dictionaries with details.
    art_similarities: Optional[List[Tuple[Any, float]]] = None # Raw similarity scores and indices from the search (e.g., from FAISS).
    art_emotion: Optional[str] = None          # The specific emotion used as input for the recommendation query.
    art_recommendation_success: Optional[bool] = None # True if recommendations were successfully generated.
    art_recommendation_error: Optional[str] = None # Error message if recommendation generation failed.
    art_display_message: Optional[str] = None        # Formatted message intended for displaying the recommendations.
    displayed_art_recommendations: Optional[bool] = None # True if the display function was executed (doesn't guarantee visual success).
    display_error: Optional[str] = None        # Error message if the display function encountered a problem.

    # --- Legacy/Redundant Attributes (Review Recommended) ---
    # 'art_suggestion_offered' might be redundant if 'suggestion_message' exists and is non-empty.
    # Consider cleanup if confirmed redundant during testing.

    # Additional attributes can be added below as needed.
    # ... (keep the existing attributes)

    # New attributes
    # ... (add any new attributes here)

    # ... (keep the rest of the existing code) 