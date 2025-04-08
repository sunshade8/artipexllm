"""
Art recommendation generator based on detected emotions.
"""

from ..models import GraphState
from ..art.artwork_recommender import ArtworkRecommender, display_recommendations
import os
from ..config import ART_CACHE_DIR, ART_CSV_PATH, ART_IMAGE_FOLDER, DEFAULT_ART_MODEL

# Initialize the artwork recommender (singleton instance)
_art_recommender = None

def get_art_recommender() -> ArtworkRecommender:
    """Get or create the artwork recommender instance."""
    global _art_recommender
    if _art_recommender is None:
        # Create art recommender
        _art_recommender = ArtworkRecommender(
            model_name=DEFAULT_ART_MODEL,
            cache_dir=ART_CACHE_DIR
        )
        
        # Load dataset if CSV exists
        if os.path.exists(ART_CSV_PATH):
            _art_recommender.load_dataset(
                csv_path=ART_CSV_PATH,
                image_folder=ART_IMAGE_FOLDER,
                rebuild_cache=False
            )
            print(f"Loaded artwork dataset from {ART_CSV_PATH}")
        else:
            print(f"Warning: Artwork dataset not found at {ART_CSV_PATH}")
    
    return _art_recommender

def format_artwork_for_response(artwork, emotion, similarity=None):
    """Format artwork information for inclusion in a response."""
    title = artwork.get("Title", "Untitled")
    artist = artwork.get("Artist Display Name", "Unknown Artist")
    date = artwork.get("Object Date", "Unknown date")
    medium = artwork.get("Medium", "Unknown medium")
    
    # Get description
    description = ""
    if "General_Text_Description" in artwork and isinstance(artwork["General_Text_Description"], str):
        desc = artwork["General_Text_Description"]
        # Truncate long descriptions
        if len(desc) > 200:
            desc = desc[:197] + "..."
        description = f"\n\nAbout this artwork: {desc}"
    
    # Get reason for recommendation
    reason = f"I selected this artwork because its qualities resonate with {emotion}."
    
    # If we have dominant colors
    color_info = ""
    if "dominant_colors" in artwork:
        colors = ", ".join(artwork["dominant_colors"])
        color_info = f"\n\nDominant colors: {colors}"
    
    # Build the complete message
    message = f"""
**{title}** by {artist} ({date})
{medium}
{color_info}
{description}

{reason}

Would you like me to explain more about how this artwork connects with {emotion}?
"""
    return message

def generate_art_recommendation(state: GraphState) -> GraphState:
    """
    Generate artwork recommendations based on detected emotion.
    
    Args:
        state: Current graph state
        
    Returns:
        Updated state with artwork recommendation
    """
    print("==== [Generating Art Recommendation] ====")
    
    emotion = state.get("detected_emotion")
    if not emotion:
        # If no emotion detected but we're in this node, use a default
        emotion = "calm"
    
    print(f"Finding artwork for emotion: {emotion}")
    
    # Get art recommender
    recommender = get_art_recommender()
    
    # Get recommendations
    try:
        recommendations, similarities = recommender.get_recommendations(
            user_emotion=emotion.capitalize(),  # Capitalize for the art system
            top_n=1  # Just get the top recommendation
        )
        
        if recommendations is not None and len(recommendations) > 0:
            # Get the top artwork
            artwork = recommendations.iloc[0]
            
            # Format for response
            art_info = format_artwork_for_response(
                artwork, 
                emotion,
                similarities[0][1] if similarities else None
            )
            
            # Create art recommendation object
            art_recommendation = {
                "emotion": emotion,
                "artwork": artwork.to_dict(),
                "formatted_text": art_info,
                "similarity": similarities[0][1] if similarities else None
            }
            
            # Return updated state
            return {
                **state, 
                "art_recommendation": art_recommendation,
                "detected_emotion": emotion
            }
        else:
            print(f"No artwork found for emotion: {emotion}")
    except Exception as e:
        print(f"Error generating art recommendation: {e}")
    
    # If we get here, something went wrong
    return {
        **state,
        "art_recommendation": None
    } 