import sys
import locale
import matplotlib as mpl
import matplotlib.pyplot as plt

# --- Optional Dependency Check: FAISS ---
# FAISS provides efficient similarity search, crucial for large datasets.
# The application can function without it, but performance will be degraded.
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    print("FAISS not available. Using slower similarity search.")
    print("Install FAISS for faster similarity search with: pip install faiss-cpu")
    FAISS_AVAILABLE = False

# --- Locale Configuration ---
# Attempt to set the system locale to ensure correct handling of
# language-specific settings (e.g., number formatting, character encoding).
try:
    locale.setlocale(locale.LC_ALL, '')
except:
    pass

def configure_matplotlib_for_cjk():
    """Configure Matplotlib to better handle CJK (Chinese, Japanese, Korean) characters"""
    
    # Check if we're in a Jupyter notebook or IPython environment
    in_notebook = False
    try:
        # This will only be defined in IPython/Jupyter
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':  # Jupyter notebook or qtconsole
            in_notebook = True
    except NameError:
        pass  # Regular Python interpreter or environment where get_ipython is not defined
        
    # Try to use a font that supports CJK characters
    cjk_fonts = []
    
    # Platform-specific font suggestions
    if sys.platform == 'darwin':  # macOS
        cjk_fonts = ['AppleGothic', 'Hiragino Sans GB', 'Heiti TC', 'STHeiti', 'Arial Unicode MS']
    elif sys.platform == 'win32':  # Windows
        cjk_fonts = ['Microsoft YaHei', 'SimHei', 'FangSong', 'SimSun', 'NSimSun', 'Arial Unicode MS']
    else:  # Linux and others
        cjk_fonts = ['Noto Sans CJK JP', 'Noto Sans CJK SC', 'Noto Sans CJK TC', 'Noto Sans CJK KR', 
                     'WenQuanYi Micro Hei', 'Droid Sans Fallback', 'Arial Unicode MS']
    
    # Try each font until we find one that exists
    font_found = False
    for font in cjk_fonts:
        try:
            # Check if matplotlib can use this font
            mpl.font_manager.findfont(font, fallback_to_default=False)
            # If we get here, the font exists
            mpl.rcParams['font.family'] = [font, 'sans-serif']
            font_found = True
            print(f"Using font: {font} for CJK character support")
            break
        except:
            continue
    
    if not font_found:
        print("Could not find a suitable font for CJK characters. Text display may be limited.")
        
    # Additional configuration for better text rendering
    mpl.rcParams['axes.unicode_minus'] = False  # Fix minus sign display
    
    # Return whether we're in a notebook (useful for display decisions)
    return in_notebook

# Execute the Matplotlib configuration and store whether running in a notebook.
IN_NOTEBOOK = configure_matplotlib_for_cjk()

# === Application Configuration Constants ===

# --- Artwork Recommender Configuration ---
# Model used by the ArtworkRecommender for generating image and text embeddings.
# 'openai/clip-vit-large-patch14' is a powerful model offering good accuracy.
RECOMMENDER_MODEL_NAME = "openai/clip-vit-large-patch14" # Model for image/text embeddings
# RECOMMENDER_MODEL_NAME = "openai/clip-vit-base-patch32" # Smaller/faster alternative

# Path to the CSV file containing artwork metadata (title, artist, description, etc.).
# Assumes a 'data/art_csv' directory relative to the project root.
CSV_PATH = "data/art_csv/painting_info_with_descriptions.csv"
# Path to the directory containing the artwork image files.
# Assumes a 'data/images' directory relative to the project root.
IMAGE_FOLDER = "data/images"
# Directory used to cache computed embeddings and potentially other processed data.
# This speeds up subsequent runs by avoiding re-computation.
# Assumes a directory named 'emotion_art_cache' relative to the project root.
CACHE_DIR = "emotion_art_cache" # Directory for caching embeddings
# ----------------------------------------- 