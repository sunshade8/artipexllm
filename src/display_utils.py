import os
import sys
import unicodedata
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd

# Import functions/constants from other modules in src
from .config import IN_NOTEBOOK
from .artwork_recommender import load_emotion_mappings, rgb_to_named_color # Need mappings for display logic

# --- Text Formatting Utility ---

# Helper function to format text for display, particularly handling CJK characters
# and potential encoding issues in different environments (terminal vs. notebook).
def format_text_for_display(text):
    """
    Format text for better display, handling multi-language characters.
    
    Args:
        text: The text to format
        
    Returns:
        Formatted text suitable for display
    """
    if text is None:
        return "Unknown"
    
    # Convert to string if it's not already (e.g., handle NaN)
    text = str(text)
    if pd.isna(text) or text.lower() == 'nan':
         return "Unknown"

    # Check if the text contains CJK characters
    has_cjk = False
    for char in text:
        try:
            # Check for specific CJK Unicode blocks
            name = unicodedata.name(char, '')
            if 'CJK' in name or 'HANGUL' in name or 'HIRAGANA' in name or 'KATAKANA' in name:
                has_cjk = True
                break
        except ValueError:
            # Characters like spaces might not have a name
            continue
    
    # Handle platform-specific display issues, especially for CJK
    # This part might need adjustment based on the specific terminal/environment
    if has_cjk and sys.platform == 'win32':
        try:
            # Attempt to encode/decode for Windows console
            return text.encode('utf-8', errors='replace').decode('utf-8')
        except Exception:
             return text # Fallback if encoding fails
    elif has_cjk and not IN_NOTEBOOK:
         # In non-notebook terminals on non-Windows, CJK might still need care
         # Often works, but providing a fallback
         try:
              # Directly return, assuming terminal supports UTF-8
              return text
         except Exception:
              return text.encode(sys.stdout.encoding, errors='replace').decode(sys.stdout.encoding)
    
    # Default return for non-CJK or notebook environments
    return text

# --- Recommendation Display Function ---

# Main function to display artwork recommendations in a user-friendly format.
# It prints metadata, similarity scores, and optionally tries to display images.
def display_recommendations(recommendations, similarities=None, display_images=True):
    """Display the recommended artworks with their metadata and similarity scores
    
    Args:
        recommendations: DataFrame of recommended artworks
        similarities: List of (DataFrame index, score) tuples 
        display_images: Whether to attempt to display images (set False if images not available or in non-graphical env)
    """
    
    if recommendations is None or recommendations.empty:
         print("No recommendations to display.")
         return

    # Ensure matplotlib uses the correct font set in config.py
    # (This assumes configure_matplotlib_for_cjk() was run at import)
    
    # Load emotion mappings to provide context in the "Why this artwork?" section.
    try:
        emotion_categories, emotion_to_colors, emotion_to_subjects_styles = load_emotion_mappings()
        all_emotions = [e for sublist in emotion_categories.values() for e in sublist]
    except Exception as e:
        print(f"Warning: Could not load emotion mappings for display explanation: {e}")
        emotion_to_colors, emotion_to_subjects_styles = {}, {}
        all_emotions = []

    # Create a dictionary for quick lookup of similarity scores by DataFrame index.
    similarity_map = {idx: score for idx, score in similarities} if similarities else {}

    # Loop through each recommended artwork (row in the DataFrame).
    for i, (idx, row_data) in enumerate(recommendations.iterrows()):
        
        print("\n" + "=" * 60)    
        print(f"🎨 Recommendation #{i+1}")
        print("=" * 60)
        
        # --- Display Basic Artwork Info ---
        # Print core metadata using the formatting helper.
        print(f"📌 Title: {format_text_for_display(row_data.get('Title', 'Unknown'))}")
        print(f"👩‍🎨 Artist: {format_text_for_display(row_data.get('Artist Display Name', 'Unknown'))}")
        print(f"🖌️  Medium: {format_text_for_display(row_data.get('Medium', 'Unknown'))}")
        print(f"📅 Created: {format_text_for_display(row_data.get('Object Date', 'Unknown date'))}")
        print(f"🏛️  Style/Classification: {format_text_for_display(row_data.get('Classification', 'Unknown'))}")
        
        # --- Display Color Information ---
        # Show dominant named colors if available in the data.
        if 'dominant_colors_named' in row_data and isinstance(row_data['dominant_colors_named'], list):
            colors = [format_text_for_display(color) for color in row_data['dominant_colors_named']]
            if colors:
                 print(f"🎨 Dominant Colors: { ', '.join(colors)}")
            # Optionally display RGB too if needed
            # if 'dominant_colors_rgb' in row_data:
            #      print(f"   RGB: {row_data['dominant_colors_rgb']}")
        
        # --- Display Similarity Score ---
        # Show the match score if available.
        if idx in similarity_map:
            similarity = similarity_map[idx] * 100  # Convert to percentage
            print(f"✨ Match Score: {similarity:.1f}%")
        
        # --- Display Artwork Description ---
        # Show the general text description, truncated if too long.
        desc = row_data.get('General_Text_Description')
        if desc and isinstance(desc, str) and not pd.isna(desc):
            # Truncate long descriptions
            if len(desc) > 300:
                desc = desc[:297] + "..."
            print("\n📝 Description:")
            print(f"  {format_text_for_display(desc)}")
        
        # --- Display "Why this artwork?" Explanation ---
        # Attempt to explain the relevance based on emotion mappings (style, color) and score.
        print("\n💭 Why this artwork for your emotion?")
        emotion = row_data.get('emotion_used', None) # Get emotion used for this specific recommendation
        
        if emotion and emotion in all_emotions: 
            reasons = []
            
            # 1. Check for style match
            artwork_style = str(row_data.get('Classification', '')).lower()
            if emotion in emotion_to_subjects_styles and artwork_style:
                 emotion_styles = [s.lower() for s in emotion_to_subjects_styles[emotion].get('styles', [])]
                 for style in emotion_styles:
                     if style in artwork_style:
                         reasons.append(f"The '{style}' style often evokes {emotion}.")
                         break # Found a style match
            
            # 2. Check for color match
            if 'dominant_colors_named' in row_data and emotion in emotion_to_colors:
                emotion_color_list = emotion_to_colors.get(emotion, [])
                artwork_colors = row_data['dominant_colors_named']
                matching_colors = [c for c in artwork_colors if c in emotion_color_list]
                if matching_colors:
                    # Prioritize the first matching color found
                    reasons.append(f"The presence of '{matching_colors[0]}' is associated with {emotion}.")

            # 3. Check for subject match (more complex - simple keyword check for now)
            # This requires a text field containing subjects or relies on the prompt generation logic
            # For now, we'll stick to style and color which are more directly available.
            
            # 4. Generic reason if no specific match found
            if not reasons:
                reasons.append(f"Its overall visual qualities align with feelings of {emotion}.")
            
            # 5. Add similarity score context
            if idx in similarity_map:
                 score_percent = similarity_map[idx] * 100
                 if score_percent > 80:
                      reasons.append(f"It scored highly ({score_percent:.0f}%) against our understanding of '{emotion}' in art.")
                 else:
                      reasons.append(f"It visually matches our '{emotion}' prompt reasonably well ({score_percent:.0f}%)." )
            else:
                 reasons.append(f"The artwork's visual features were found to be relevant to '{emotion}'.")

            # Print the collected reasons, limited to a few for brevity.
            for reason in reasons[:3]:
                print(f"  • {reason}")
        else:
             print(f"  • Could not determine specific emotional link (Emotion: {emotion}). The artwork was selected based on visual similarity.")

        print("-" * 60)
        
        # --- Attempt to Display Image --- 
        # If requested, running in a notebook, and the image file exists, display it using Matplotlib.
        image_path = row_data.get('image_path')
        if display_images and IN_NOTEBOOK and image_path and os.path.isfile(image_path):
            try:
                img = Image.open(image_path)
                plt.figure(figsize=(6, 6)) # Smaller figure size
                plt.imshow(img)
                plt.axis('off')
                
                # Format title for display
                title_text = format_text_for_display(row_data.get('Title', 'Untitled'))
                plt.title(title_text, fontsize=12) # Smaller font size
                plt.show()
                print("\n")
            except Exception as e:
                print(f"Error displaying image {image_path}: {e}\n")
        elif display_images and image_path and not os.path.isfile(image_path):
             print(f"[Image not found at: {image_path}]\n")
        elif display_images and not IN_NOTEBOOK:
             print(f"[Image display skipped in non-notebook environment: {image_path}]\n") 