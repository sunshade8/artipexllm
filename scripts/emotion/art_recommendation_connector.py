"""
Art recommendation connector for ArtiTech LangGraph system.

This module connects the LangGraph emotion detection system
with the ArtworkRecommender system to provide art recommendations
based on detected emotions.
"""

from typing import Dict, List, Optional, Any, Tuple
import os
import sys
import pandas as pd
from PIL import Image  # Import PIL.Image for image processing
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- Import IN_NOTEBOOK from src.config ---
# Assume src.config exists relative to this script's execution context
# This might require adjustments depending on how scripts are run
try:
    # Attempt relative import first (if scripts are treated as part of a package)
    from ...src.config import IN_NOTEBOOK
except ImportError:
    # Fallback: Assume src is in sys.path (common for notebooks)
    try:
        from src.config import IN_NOTEBOOK
    except ImportError:
        print("Warning (art_recommendation_connector): Could not import IN_NOTEBOOK from src.config.")
        IN_NOTEBOOK = False
# -----------------------------------------

# Add paths for imports (Potentially redundant if src is in path)
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
# sys.path.append(project_root)

# Import art recommendation components
# These imports use the global ArtworkRecommender class already defined in the notebook
# Also import helper function from art_suggestion (keep for now)
try:
    from scripts.emotion.art_suggestion import _map_emotion_to_art_system
except ImportError:
    print("Warning (art_recommendation_connector): Could not import _map_emotion_to_art_system.")
    def _map_emotion_to_art_system(emotion: str) -> str: return emotion # Dummy fallback

# Define list of valid emotions (copied from original node)
VALID_ART_SYSTEM_EMOTIONS = [
    "Joy", "Love", "Serenity", "Amusement", "Gratitude", "Hope",
    "Admiration", "Sadness", "Anger", "Fear", "Disgust", "Confusion",
    "Boredom", "Dreamy", "Whimsical", "Mystical", "Spiritual",
    "Nostalgia", "Contemplation", "Wonder", "Awe", "Connectedness"
]
VALID_ART_SYSTEM_EMOTIONS_LOWER = [e.lower() for e in VALID_ART_SYSTEM_EMOTIONS]

# ============================================================================
# Art Recommendation Node
# ============================================================================
def generate_art_recommendations_node(state: Dict) -> Dict:
    """
    Node to generate art recommendations using the existing ArtworkRecommender.
    
    This function:
    1. Maps the detected emotion to the art recommendation system
    2. Calls the ArtworkRecommender to get recommendations
    3. Updates the state with the results
    
    Args:
        state: Current graph state containing the detected emotion
        
    Returns:
        Updated state with art recommendations
    """
    print("==== [Art Recommendation Node] ====")
    
    # --- Get Detected Emotion and Request Type from State ---
    detected_emotion = state.get("detected_emotion")
    is_direct_request = state.get("direct_art_request", False) # Check if this was a direct request
    
    # If no emotion detected, use a default
    if not detected_emotion:
        print("No emotion detected, using default emotion for art")
        detected_emotion = "calm" 
    
    print(f"Detected Emotion: {detected_emotion}")
    print(f"Is Direct Art Request: {is_direct_request}")

    # --- Determine Target Emotion --- 
    target_emotion = detected_emotion # Start with the detected emotion
    
    # Only use LLM to find a *different* target emotion if it was NOT a direct request
    if not is_direct_request:
        print("Not a direct request, determining helpful target emotion via LLM...")
        llm = ChatOpenAI(temperature=0.1, model="gpt-3.5-turbo")
        valid_emotions_str = ", ".join(VALID_ART_SYSTEM_EMOTIONS)

        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are an expert in art therapy and emotion regulation. 
Your goal is to select a helpful target emotion for an art recommendation based on the user's current detected emotion.
The recommendation should aim to balance or improve the user's state.
For example:
- If the user feels 'sadness', suggest 'hope' or 'joy'.
- If the user feels 'anger', suggest 'serenity' or 'calm'.
- If the user feels 'joy', suggest 'serenity' or 'contemplation' to savor it.
- If the user feels 'anxiety', suggest 'serenity' or 'peaceful'.

You MUST choose the target emotion from the following list of emotions that the art recommendation system understands:
{valid_emotions_str}

Respond ONLY with the single chosen target emotion from the list, and nothing else."""),
            ("human", f"""The user's detected emotion is: {detected_emotion}
What target emotion from the list would be most helpful for an art recommendation?""")
        ])

        chain = prompt | llm | StrOutputParser()
        
        try:
            llm_suggested_target = chain.invoke({}).strip()
            print(f"LLM suggested target emotion: {llm_suggested_target}")
            
            # Validate LLM output and update target_emotion if valid
            if llm_suggested_target.lower() in VALID_ART_SYSTEM_EMOTIONS_LOWER:
                target_emotion_index = VALID_ART_SYSTEM_EMOTIONS_LOWER.index(llm_suggested_target.lower())
                target_emotion = VALID_ART_SYSTEM_EMOTIONS[target_emotion_index]
                print(f"Using LLM target emotion: {target_emotion}")
            else:
                print(f"Warning: LLM returned invalid target '{llm_suggested_target}'. Using original detected emotion '{detected_emotion}'.")
                target_emotion = detected_emotion # Revert to original if LLM output invalid
                
        except Exception as e:
            print(f"Error getting target emotion from LLM: {e}. Using original detected emotion '{detected_emotion}'.")
            target_emotion = detected_emotion # Revert to original on error
    else:
        # If it IS a direct request, explicitly state we are using the requested emotion
        print(f"Direct art request detected. Using requested emotion '{detected_emotion}' directly.")
        target_emotion = detected_emotion

    # --- Map Target Emotion and Get Recommendations ---
    # Map the final chosen target emotion (either direct or LLM-suggested) 
    # to the exact string the recommender expects and validate it.
    mapped_target_emotion = _map_emotion_to_art_system(target_emotion)
    print(f"Final mapped target emotion for recommendation: '{mapped_target_emotion}'")
    
    # Get art recommendations using the existing ArtworkRecommender
    try:
        # Try to access the global recommender in various ways
        recommender = None
        try:
            # First try the builtins approach (most reliable)
            import builtins
            if hasattr(builtins, 'recommender') and builtins.recommender:
                recommender = builtins.recommender
                print(f"Found recommender in builtins.")
        except Exception as e:
            print(f"Error accessing recommender from builtins: {e}")
            
        if recommender is None:
            try:
                # Try importing from __main__
                from __main__ import recommender as main_recommender # Use alias to avoid potential clash
                if main_recommender:
                    recommender = main_recommender
                    print(f"Found recommender in __main__.")
            except ImportError:
                print("Recommender not found in __main__. This node might fail.")
                recommender = None # Ensure it's None
        
        # Handle case where recommender is still None
        if recommender is None:
             raise ValueError("ArtworkRecommender instance not found. Cannot generate recommendations.")

        # Call the recommender to get artworks for the *mapped target* emotion.
        print(f"Getting art recommendations for '{mapped_target_emotion}' (Final Mapped Target Emotion)")
        recommendations, similarities = recommender.get_recommendations(mapped_target_emotion, top_n=1)
        
        # If recommendations were found, process them
        if recommendations is not None:
            # Save the recommendations and similarities for the node response
            print(f"Found {len(recommendations)} recommendations for target emotion '{mapped_target_emotion}'")
            
            # Convert DataFrame to list of dictionaries for serialization
            recommendations_list = recommendations.to_dict('records')
            
            # --- Convert similarity scores to standard Python floats ---
            if similarities:
                similarities_serializable = [(idx, float(score)) for idx, score in similarities]
            else:
                similarities_serializable = []
            # ----------------------------------------------------------
            
            # Return updated state with recommendations
            updated_state = {
                **state,
                "art_recommendations": recommendations_list,
                "art_similarities": similarities_serializable,
                "art_emotion": mapped_target_emotion, # Store the target emotion used for search
                "art_recommendation_success": True
            }
        else:
            print(f"No recommendations found for target emotion '{mapped_target_emotion}'")
            updated_state = {
                **state,
                "art_recommendation_success": False,
                "art_recommendation_error": "No recommendations found",
                "art_recommendations": [],
                "art_emotion": mapped_target_emotion # Still store the target emotion
            }
    except Exception as e:
        print(f"Error generating art recommendations for target emotion '{mapped_target_emotion}': {e}")
        updated_state = {
            **state,
            "art_recommendation_success": False,
            "art_recommendation_error": str(e),
            "art_recommendations": [],
            "art_emotion": mapped_target_emotion # Still store the target emotion
        }
    
    return updated_state

# ============================================================================
# Display Art Recommendations Node
# ============================================================================
def display_art_recommendations_node(state: Dict) -> Dict:
    """
    Node to display art recommendations.
    
    This function:
    1. Checks if recommendations are in the state
    2. Displays them using the existing display_recommendations function
    
    Args:
        state: Current graph state containing the art recommendations
        
    Returns:
        Updated state
    """
    print("==== [Displaying Art Recommendations] ====")
    
    # Extract the recommendations from the state
    recommendations = state.get("art_recommendations")
    similarities = state.get("art_similarities")
    
    # Check if recommendations are available
    if recommendations is not None:
        try:
            # Try to access the display function in various ways
            display_func = None
            
            # Try builtins first
            try:
                import builtins
                if hasattr(builtins, 'display_recommendations'):
                    display_func = builtins.display_recommendations
                    print("Found display_recommendations in builtins")
            except Exception as e:
                print(f"Error accessing display_recommendations from builtins: {e}")
            
            # Try main module if not in builtins
            if display_func is None:
                try:
                    from __main__ import display_recommendations
                    display_func = display_recommendations
                    print("Found display_recommendations in __main__")
                except ImportError:
                    print("display_recommendations not found in __main__. Display will be text-based.")
            
            # Create a fallback display function if needed
            if display_func is None:
                print("Creating fallback text display function")
                def display_func(recommendations, similarities=None, display_images=False):
                    print("\n==== Art Recommendations ====\n")
                    for i, row in recommendations.iterrows():
                        print(f"Art #{i+1}: {row.get('Title', 'Untitled')}")
                        if similarities and i < len(similarities):
                            sim_score = similarities[i][1] * 100
                            print(f"Match score: {sim_score:.1f}%\n")
                    return True
            
            # Display the recommendations
            if display_func:
                print("Displaying art recommendations...")
                # Convert list of dicts back to DataFrame before displaying
                if isinstance(recommendations, list) and recommendations:
                    # import pandas as pd # Already imported at top
                    recommendations_df = pd.DataFrame(recommendations)
                elif isinstance(recommendations, pd.DataFrame): # Already a DataFrame
                    recommendations_df = recommendations
                else: # Handle unexpected type or empty list
                    print("Warning: Recommendations are not in the expected list or DataFrame format.")
                    recommendations_df = pd.DataFrame() # Create empty df to avoid error

                # Use the IN_NOTEBOOK flag for display_images
                display_func(recommendations_df, similarities, display_images=IN_NOTEBOOK)
            
            # Return updated state
            updated_state = {
                **state,
                "displayed_art_recommendations": True
            }
        except Exception as e:
            print(f"Error displaying art recommendations: {e}")
            updated_state = {
                **state,
                "displayed_art_recommendations": False,
                "display_error": str(e)
            }
    else:
        print("No art recommendations to display")
        updated_state = {
            **state,
            "displayed_art_recommendations": False,
            "display_error": "No recommendations available"
        }
    
    return updated_state

# ============================================================================
# Initialization Function (Keep for reference / potential use outside graph)
# ============================================================================
def initialize_art_recommendation_system(csv_path, image_folder, cache_dir="./emotion_art_cache"):
    """
    Initializes the ArtworkRecommender system.

    Args:
        csv_path (str): Path to the CSV file with artwork data.
        image_folder (str): Path to the folder containing artwork images.
        cache_dir (str): Directory for caching embeddings.

    Returns:
        ArtworkRecommender | None: An initialized recommender instance or None if failed.
    """
    print("--- Initializing Art Recommendation System --- ")
    try:
        # Ensure src is in the path to import ArtworkRecommender
        # Need project_root defined relative to this script if run directly
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
        src_path = os.path.join(project_root, "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
            print(f"Added {src_path} to sys.path")

        from src.artwork_recommender import ArtworkRecommender
        from src.config import RECOMMENDER_MODEL_NAME # Get model name from config

        recommender = ArtworkRecommender(
            model_name=RECOMMENDER_MODEL_NAME,
            cache_dir=cache_dir
        )

        print(f"Loading dataset from CSV: {csv_path}, Images: {image_folder}")
        artwork_data = recommender.load_dataset(csv_path=csv_path, image_folder=image_folder)

        if artwork_data is not None:
            print("Artwork Recommender initialized successfully.")
            return recommender
        else:
            print("Failed to load dataset. Recommender initialization failed.")
            return None

    except ImportError as e:
        print(f"ImportError during recommender initialization: {e}")
        print("Please ensure src/artwork_recommender.py exists and dependencies are installed.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during recommender initialization: {e}")
        # Fallback or Minimal Recommender for testing
        print("Using MinimalRecommender as fallback.")
        # return MinimalRecommender() # Example fallback
        return None

# Placeholder/Fallback Recommender classes (Keep for reference)
class TestRecommender:
    def __init__(self, model_name=None, cache_dir=None):
        print("Initialized TestRecommender (Placeholder)")
        self.model_name = model_name
        self.cache_dir = cache_dir

    def load_dataset(self, csv_path, image_folder):
        print(f"TestRecommender: Loading fake data (CSV: {csv_path}, Images: {image_folder})")
        # Simulate loading some data
        data = {
            "Object ID": [1, 2, 3],
            "Title": ["Test Art 1", "Test Art 2", "Test Art 3"],
            "Artist Display Name": ["Test Artist", "Test Artist", "Test Artist"],
            "Classification": ["Painting", "Drawing", "Sculpture"],
            "image_path": ["path/1.jpg", "path/2.jpg", "path/3.jpg"],
            "dominant_colors": [["Blue"], ["Red"], ["Green"]]
        }
        self.artwork_data = pd.DataFrame(data)
        return self.artwork_data

    def get_recommendations(self, emotion, top_n=1):
        print(f"TestRecommender: Getting {top_n} recommendations for '{emotion}'")
        # if self.artwork_data is None: # Simplified: assume loaded
        #     self.load_dataset("dummy.csv", "dummy_images") # Load fake data if needed

        # Return top_n items as a sample
        sample_df = pd.DataFrame({"Object ID": [1], "Title": [f"Test for {emotion}"]}) # Simplified placeholder
        # sample_df = self.artwork_data.sample(min(top_n, len(self.artwork_data)))
        similarities = [(1, 0.9)] # Simplified placeholder
        # similarities = [(idx, 0.9 - i*0.1) for i, idx in enumerate(sample_df.index)]
        return sample_df, similarities

class MinimalRecommender:
    def get_recommendations(self, emotion, top_n=1):
        print(f"MinimalRecommender: Returning placeholder for '{emotion}'")
        # Return minimal placeholder structure
        data = {"Object ID": [999], "Title": [f"Placeholder for {emotion}"]}
        df = pd.DataFrame(data)
        return df, [(999, 0.5)] 