# graph.py
# Code extracted from artitech_final.ipynb

import os
import sys
from pathlib import Path
import re
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple

# --- Setup API Keys ---
# Add parent directory to path if necessary (adjust as needed)
# sys.path.insert(0, str(Path(__file__).parent.parent))

# Try using fix_api_key first
try:
    from fix_api_key import setup_api_key
    api_keys_set = setup_api_key()
except ImportError:
    print("Warning: fix_api_key module not found. Trying .env...")
    api_keys_set = False

# Fallback to .env
if not api_keys_set:
    try:
        from dotenv import load_dotenv
        env_path = Path(__file__).resolve().parent / '.env' # Assumes .env is in the same dir as graph.py
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
            print(f"Loaded environment variables from {env_path}")
            # Check for essential keys
            if os.environ.get("OPENAI_API_KEY") and os.environ.get("TAVILY_API_KEY"):
                api_keys_set = True
        else:
             print(f"Warning: .env file not found at {env_path}")
    except ImportError:
        print("Warning: python-dotenv not installed. Cannot load .env file.")

# Final check for required keys
if not os.environ.get("OPENAI_API_KEY"):
    print("ERROR: OPENAI_API_KEY environment variable is not set. Please set it.")
    # sys.exit(1) # Consider exiting if essential keys are missing
if not os.environ.get("TAVILY_API_KEY"):
    print("Warning: TAVILY_API_KEY environment variable is not set. Web search will be disabled.")
    # Potentially disable features or use a placeholder if Tavily is crucial

# --- Langchain and Core Components ---
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnablePassthrough,
    RunnableConfig,
    RunnableParallel
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain_community.vectorstores import FAISS
from langchain_community.tools.tavily_search import TavilySearchResults

# --- Local Modules & Configuration ---
# Assuming these modules are in the EmpathyArtRAG directory or accessible
try:
    from empathy_art_rag.config import (
        ART_CSV_PATH, ART_IMAGE_FOLDER, ART_CACHE_DIR,
        DEFAULT_ART_MODEL, FAISS_INDEX_PATH, METADATA_JSON_PATH
    )
    from empathy_art_rag.art.artwork_recommender import ArtworkRecommender
    from empathy_art_rag.handlers.emotion_handler import detect_emotion_keywords
    from empathy_art_rag.utils import print_color # Assuming print_color is still useful
except ImportError as e:
    print(f"Error importing local modules: {e}")
    print("Please ensure the EmpathyArtRAG package is correctly installed and structured.")
    sys.exit(1)


# --- Initialize Components ---

# LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0.7) # Using gpt-4o as specified

# Embeddings
embeddings = OpenAIEmbeddings()

# Vector Store and Retriever (Load from disk)
try:
    print(f"Loading FAISS index from: {FAISS_INDEX_PATH}")
    vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 5}) # Get top 5 initially
    print("FAISS index loaded successfully.")
except Exception as e:
    print(f"Error loading FAISS index from {FAISS_INDEX_PATH}: {e}")
    print("Please ensure the index was created correctly (e.g., using build_vector_store.py).")
    sys.exit(1)

# Add contextual compression for relevance filtering
embeddings_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.75)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=embeddings_filter,
    base_retriever=base_retriever
)
retriever = compression_retriever # Use the compressed retriever

# Artwork Recommender
try:
    print("Initializing ArtworkRecommender...")
    artwork_recommender = ArtworkRecommender(
        model_name=DEFAULT_ART_MODEL,
        cache_dir=ART_CACHE_DIR
    )
    # Pre-load dataset for recommendations (optional, but might speed up first request)
    artwork_recommender.load_dataset(csv_path=ART_CSV_PATH, image_folder=ART_IMAGE_FOLDER)
    print("ArtworkRecommender initialized.")
except Exception as e:
    print(f"Error initializing ArtworkRecommender: {e}")
    sys.exit(1)

# Tavily Search Tool
try:
    tavily_tool = TavilySearchResults(max_results=5) # Get more results
    print("Tavily search tool initialized.")
except Exception as e:
    print(f"Warning: Failed to initialize Tavily search tool: {e}. Web search unavailable.")
    tavily_tool = None # Set to None if initialization fails

# --- Helper Functions ---

def format_docs(docs: List[Document]) -> str:
    """Convert Documents to a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

def format_artwork(artwork: pd.Series) -> Dict[str, Any]:
    """Format artwork pandas Series into a dictionary for display/LLM context."""
    if artwork is None or artwork.empty:
        return {}
    # Ensure keys exist and handle potential NaN/None values gracefully
    details = {
        "Title": artwork.get('Title', 'N/A'),
        "Artist Display Name": artwork.get('Artist Display Name', 'N/A'),
        "Object Date": artwork.get('Object Date', 'N/A'),
        "Culture": artwork.get('Culture', 'N/A') or 'N/A', # Handle None/empty strings
        "Department": artwork.get('Department', 'N/A') or 'N/A',
        "image_path": artwork.get('image_path', None), # Keep path if available
        # Add other potentially relevant fields if needed
        "Medium": artwork.get('Medium', 'N/A') or 'N/A',
        # Dominant colors might be too verbose for the LLM context sometimes
        # "dominant_colors": artwork.get('dominant_colors', [])
    }
    return details

# Function to load specific artwork data (used by art_chain)
def load_artwork_data_from_meta(artwork_details: Dict[str, Any]) -> str:
    """Loads full metadata for a recommended artwork if needed (or formats existing details)."""
    # For now, just format the details we already have from the recommender
    # In a more complex system, this could fetch more data based on an ID
    if not artwork_details:
        return "No artwork details found."

    formatted_string = f"Artwork Title: {artwork_details.get('Title', 'N/A')}\n"
    formatted_string += f"Artist: {artwork_details.get('Artist Display Name', 'N/A')}\n"
    formatted_string += f"Date: {artwork_details.get('Object Date', 'N/A')}\n"
    formatted_string += f"Culture: {artwork_details.get('Culture', 'N/A')}\n"
    formatted_string += f"Medium: {artwork_details.get('Medium', 'N/A')}\n"
    # Add a summary or key characteristics if available in artwork_details
    # formatted_string += f"Description/Style: {artwork_details.get('description', 'N/A')}\n"
    return formatted_string


# --- Emotion Detection & Routing Logic ---

# Use the imported handler
# detect_emotion_keywords is assumed to return a list of detected emotion strings

# Define art request patterns (similar to chat_interface.py)
ART_REQUEST_PATTERNS = [
    r"(?i)recommend.*art", r"(?i)show me.*art", r"(?i)suggest.*artwork",
    r"(?i)find.*painting", r"(?i)art.*recommend", r"(?i)artwork.*help",
    r"(?i)healing.*art", r"(?i)therapeutic.*art", r"(?i)art\s+piece",
    r"(?i)show\s+art", r"(?i)painting", r"(?i)can you recommend",
    # Korean patterns
    r"작품.*추천", r"그림.*보여", r"예술.*추천", r"미술.*치유"
]

ART_CONFIRMATION_PATTERNS = [
    r"(?i)\byes\b", r"(?i)\bsure\b", r"(?i)\bplease\b", r"(?i)\bok\b",
    r"(?i)\byeah\b", r"(?i)show me", r"(?i)recommend", r"(?i)i would",
    r"(?i)i'd like", r"네", r"예", r"좋아요", r"보여줘"
]

def is_art_request(text: str) -> bool:
    """Check if the text is a direct art request."""
    return any(re.search(pattern, text) for pattern in ART_REQUEST_PATTERNS)

def is_art_confirmation(text: str) -> bool:
    """Check if the text confirms a previous art offer."""
    return any(re.search(pattern, text) for pattern in ART_CONFIRMATION_PATTERNS)

# Emotion mapping (similar to chat_interface.py) - used for recommender
def get_emotion_map():
    return {
        "joy": "Joy", "happiness": "Joy", "love": "Love", "serenity": "Serenity",
        "calm": "Serenity", "peaceful": "Serenity", "amusement": "Amusement",
        "gratitude": "Gratitude", "hope": "Hope", "admiration": "Admiration",
        "sadness": "Sadness", "grief": "Sadness", "anger": "Anger", "fear": "Fear",
        "anxiety": "Anxiety", "disgust": "Disgust", "dread": "Dread",
        "confusion": "Confusion", "boredom": "Boredom", "loneliness": "Sadness",
        "dreamy": "Dreamy", "curious": "Dreamy", "mystical": "Mystical",
        "spiritual": "Spiritual", "whimsical": "Whimsical", "creative": "Whimsical",
        "nostalgia": "Nostalgia", "contemplative": "Contemplation", "wonder": "Wonder",
        "awe": "Awe", "connectedness": "Connectedness"
    }

# Main Router Function
def route_to_art_or_rag(inputs: Dict[str, Any], config: Optional[RunnableConfig] = None) -> str:
    """
    Determines the primary path: direct RAG, ask about art, or recommend art.
    Returns 'rag', 'suggest_art', or 'recommend_art'.
    Updates the state dictionary with detected emotion if needed.
    """
    question = inputs["question"]
    state = inputs.get("state", {}) # Get current state from input

    detected_emotions = detect_emotion_keywords(question)
    direct_request = is_art_request(question)
    confirmation = is_art_confirmation(question)
    pending_suggestion = state.get("pending_art_suggestion", False)
    pending_emotion = state.get("detected_emotion_for_suggestion", None)

    decision = "rag" # Default route

    if direct_request:
        decision = "recommend_art"
        # If direct request, detect emotion or default
        emotion_to_use = detected_emotions[0] if detected_emotions else "calm" # Default for direct request
        state["detected_emotion_for_recommendation"] = emotion_to_use
        print_color(f"[Router] Decision: {decision} (Direct Request, Emotion: {emotion_to_use})", "magenta")

    elif pending_suggestion and confirmation and pending_emotion:
        decision = "recommend_art"
        # Use the emotion stored when the suggestion was made
        state["detected_emotion_for_recommendation"] = pending_emotion
        state["pending_art_suggestion"] = False # Clear the flag
        state["detected_emotion_for_suggestion"] = None
        print_color(f"[Router] Decision: {decision} (User Confirmed, Emotion: {pending_emotion})", "magenta")

    elif pending_suggestion and not confirmation:
        # User didn't confirm, maybe asked something else or denied. Route to RAG.
        decision = "rag"
        state["pending_art_suggestion"] = False # Clear the flag
        state["detected_emotion_for_suggestion"] = None
        print_color(f"[Router] Decision: {decision} (Suggestion Pending, No Confirmation)", "magenta")

    elif detected_emotions and not direct_request and not pending_suggestion:
        # Emotion detected, not a direct request, no pending suggestion -> Suggest art
        decision = "suggest_art"
        primary_emotion = detected_emotions[0]
        # Store the emotion for potential confirmation later
        state["detected_emotion_for_suggestion"] = primary_emotion
        state["pending_art_suggestion"] = True # Set flag
        print_color(f"[Router] Decision: {decision} (Emotion Detected: {primary_emotion})", "magenta")

    else:
        # Default case: No relevant emotion, not art request/confirmation
        print_color(f"[Router] Decision: {decision} (Default)", "magenta")

    # Update state for the next steps
    inputs["state"] = state
    return decision


# --- Prompt Templates ---

# Standard RAG Prompt
RAG_PROMPT_TEMPLATE = """You are an empathetic AI assistant. Use the following retrieved context to answer the user's question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise and empathetic.

Context:
{context}

Question: {question}

Answer:"""
RAG_PROMPT = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

# Empathy Prompt (when suggesting art)
EMPATHY_PROMPT_TEMPLATE = """You are an empathetic AI assistant. The user has expressed the emotion: {emotion}.
Respond empathetically to their situation based on their last message: "{question}".
After your empathetic response, ask if they would like to see an artwork that might resonate with how they are feeling.
Keep your response conversational and natural. Do not mention the context emotion directly unless it feels natural.
Your response should ONLY contain the empathetic message and the question about recommending art.
"""
EMPATHY_PROMPT = PromptTemplate.from_template(EMPATHY_PROMPT_TEMPLATE)


# Art Recommendation Prompt (when generating the final message with art)
ART_REQUEST_PROMPT_TEMPLATE = """You are an empathetic AI assistant specialized in recommending art.
The user expressed the emotion: {emotion}. You have selected the following artwork for them based on this emotion:

Artwork Details:
{artwork_context}

User's request/confirmation: "{question}"

Task:
1. Briefly acknowledge the user's request or confirmation.
2. Introduce the recommended artwork (Title and Artist).
3. Explain concisely WHY this specific artwork might resonate with their feeling of {emotion}, linking its characteristics (e.g., colors, style, subject) to the emotion.
4. Mention that you will display the artwork details and attempt to show the image.
Keep the tone empathetic, insightful, and conversational. Focus on the connection between the art and the emotion.
"""
ART_REQUEST_PROMPT = PromptTemplate.from_template(ART_REQUEST_PROMPT_TEMPLATE)


# --- Sub-Chains ---

# 1. RAG Chain (including web search if needed)
rag_chain = (
    RunnablePassthrough.assign(context=(lambda x: x["question"]) | retriever | format_docs)
    .assign(answer=RAG_PROMPT | llm | StrOutputParser())
    .pick(["answer"]) # Output just the answer string
)

# Optional: Web Search Chain (using Tavily)
web_search_chain = (
    RunnableLambda(lambda x: x["question"]) # Pass the question directly
    | tavily_tool # Use the initialized Tavily tool
    | RunnableLambda(lambda results: {"web_search_results": results}) # Format output
)

# Decide whether to use web search
def route_to_web_search_or_rag(inputs: Dict[str, Any]) -> RunnableBranch:
    question = inputs["question"]
    # Simple heuristic: if Tavily is available and question seems factual
    # A more complex heuristic like in chat_interface.py could be used here
    if tavily_tool and ("what is" in question.lower() or "who is" in question.lower() or "when was" in question.lower()):
         print_color("[Router] Decided to use Web Search", "cyan")
         # Combine RAG context with Web Search results for the final answer
         return RunnablePassthrough.assign(
             context=(lambda x: x["question"]) | retriever | format_docs,
             web_context=web_search_chain
         ) | ChatPromptTemplate.from_template(
             """Answer the user's question using BOTH the retrieved context and the web search results. Prioritize web search results for factual/current information. Be concise and empathetic.

Retrieved Context:
{context}

Web Search Results:
{web_search_results}

Question: {question}

Answer:"""
         ) | llm | StrOutputParser() | (lambda ans: {"generation": ans}) # Format output

    else:
         print_color("[Router] Decided to use standard RAG", "cyan")
         # Just use the standard RAG chain
         return rag_chain | (lambda ans: {"generation": ans.get("answer", "")}) # Format output

# 2. Empathy + Suggestion Chain (when emotion detected, suggest art)
empathy_chain = (
    RunnablePassthrough.assign(
        emotion=lambda x: x["state"].get("detected_emotion_for_suggestion", "neutral"),
        question=lambda x: x["question"]
    )
    | EMPATHY_PROMPT
    | llm
    | StrOutputParser()
    | (lambda response: {"suggestion_message": response}) # Output key
)

# 3. Art Recommendation Chain (when user asks for or confirms art)
art_chain = RunnableParallel(
    # Pass emotion and question through
    passthrough=RunnablePassthrough(),
    # Get artwork recommendation based on emotion
    artwork_recommendation=RunnableLambda(
        lambda x: artwork_recommender.get_recommendations(
            user_emotion=get_emotion_map().get(x["state"].get("detected_emotion_for_recommendation", "calm").lower(), "Serenity"),
            top_n=1
        )[0].iloc[0] # Get the first recommendation Series
    )
) | RunnablePassthrough.assign(
    # Format the chosen artwork details
    artwork_details=lambda x: format_artwork(x['artwork_recommendation']),
    # Load/Format context string for the LLM prompt
    artwork_context=lambda x: load_artwork_data_from_meta(format_artwork(x['artwork_recommendation']))
) | RunnablePassthrough.assign(
    # Generate the final response using the LLM
    generation=(
        lambda x: {
            "emotion": x["passthrough"]["state"].get("detected_emotion_for_recommendation", "calm"),
            "artwork_context": x["artwork_context"],
            "question": x["passthrough"]["question"] # Use original question/confirmation
        }
        | ART_REQUEST_PROMPT
        | llm
        | StrOutputParser()
    )
# Select final output keys: the generated text and the structured artwork details
) | RunnableLambda(lambda x: {"generation": x["generation"], "artwork_details": x["artwork_details"]})


# --- Main Adaptive RAG Chain ---

adaptive_rag = RunnablePassthrough().assign(
    # Add a state dictionary if it doesn't exist
    state=lambda x: x.get("state", {})
) | RunnablePassthrough.assign(
    # Determine the route based on input and state
    route=RunnableLambda(route_to_art_or_rag)
) | RunnableBranch(
    # Route 1: Emotion detected, suggest art
    (lambda x: x["route"] == "suggest_art", empathy_chain),
    # Route 2: User requested or confirmed art, recommend specific art
    (lambda x: x["route"] == "recommend_art", art_chain),
    # Route 3: Default RAG (potentially with web search)
    # (lambda x: x["route"] == "rag", route_to_web_search_or_rag), # Integrate web search routing
    RunnableLambda(route_to_web_search_or_rag) # Default path
)

# --- Example Usage (for testing within this script if needed) ---
if __name__ == '__main__':
    print("\n--- Testing Adaptive RAG Chain ---")

    # Example Test Function
    def run_test(question: str, initial_state: Optional[Dict] = None):
        print(f"\nUser: {question}")
        config = RunnableConfig(recursion_limit=25, configurable={"thread_id": "test-thread"})
        state = initial_state if initial_state else {}
        inputs = {"question": question, "state": state}
        result = adaptive_rag.invoke(inputs, config=config)

        print("AI Response:")
        if result.get("generation"):
            print(f"  Generation: {result['generation']}")
        if result.get("suggestion_message"):
            print(f"  Suggestion: {result['suggestion_message']}")
        if result.get("artwork_details"):
            print(f"  Artwork: {result['artwork_details'].get('Title', 'N/A')} by {result['artwork_details'].get('Artist Display Name', 'N/A')}")
            # print(f"  Artwork Details: {result['artwork_details']}") # Uncomment for full details

        # Return the state for multi-turn tests
        return result.get("state", {})

    # Test Case 1: Emotion -> Suggestion
    print("\n--- Test Case 1: Emotion -> Suggestion ---")
    current_state = run_test("I'm feeling quite sad today.")

    # Test Case 2: Confirmation -> Recommendation
    print("\n--- Test Case 2: Confirmation -> Recommendation ---")
    if current_state.get("pending_art_suggestion"):
        current_state = run_test("Yes, please show me something.", initial_state=current_state)
    else:
        print("Skipping confirmation test, suggestion was not pending.")

    # Test Case 3: Direct Art Request
    print("\n--- Test Case 3: Direct Art Request ---")
    run_test("Can you recommend art for happiness?")

    # Test Case 4: Standard RAG Question
    print("\n--- Test Case 4: Standard RAG ---")
    run_test("What is impressionism?")

     # Test Case 5: Web Search Question (if Tavily enabled)
    if tavily_tool:
        print("\n--- Test Case 5: Web Search ---")
        run_test("What is the latest news about AI art generation?")
    else:
        print("\n--- Skipping Web Search Test (Tavily not enabled) ---")
