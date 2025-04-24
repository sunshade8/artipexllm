import os
import sys
import pandas as pd
from .artwork_recommender import ArtworkRecommender
from .config import CSV_PATH, IMAGE_FOLDER, CACHE_DIR, RECOMMENDER_MODEL_NAME, FAISS_AVAILABLE

# Factory function to create and initialize the artwork recommender system.
# Encapsulates the setup logic, including loading data and building the index.
def initialize_recommender(rebuild_cache: bool = False, image_extension: str = ".jpg"):
    """
    Initializes the ArtworkRecommender and loads the dataset.

    Args:
        rebuild_cache (bool): Whether to force regeneration of embeddings cache.
        image_extension (str): The file extension for the artwork images.

    Returns:
        ArtworkRecommender | None: An initialized recommender instance, or None if initialization fails.
    """
    # Configuration values are imported from config.py for centralized management.
    # --- Ensure src is in path (important if run as script, maybe less needed in notebook) ---
    # Use absolute path from known project structure if needed, but relative imports should work if structured correctly
    # project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Assuming src is one level down
    # if project_root not in sys.path:
    #     sys.path.insert(0, project_root)
    # print(f"Ensured '{project_root}' is in sys.path") # Less verbose

    # Initialize the core recommender class with the specified model and cache directory.
    try:
        print(f"Initializing ArtworkRecommender with model: {RECOMMENDER_MODEL_NAME}")
        recommender = ArtworkRecommender(
            model_name=RECOMMENDER_MODEL_NAME,
            cache_dir=CACHE_DIR
        )

        # Load the artwork metadata (from CSV) and generate/load image embeddings.
        print(f"Attempting to load dataset from CSV: {CSV_PATH} and Images: {IMAGE_FOLDER}")
        artwork_data = recommender.load_dataset(
            csv_path=CSV_PATH,
            image_folder=IMAGE_FOLDER,
            image_extension=image_extension,
            rebuild_cache=rebuild_cache
        )

        # Check if data loading was successful and the recommender is ready.
        if artwork_data is not None and not artwork_data.empty:
            print(f"Dataset loaded successfully. Recommender ready.")
            print(f"Total artworks loaded: {len(artwork_data)}")
            if FAISS_AVAILABLE and recommender.faiss_index:
                print(f"FAISS index active with {recommender.faiss_index.ntotal} items.")
            elif FAISS_AVAILABLE:
                print("FAISS is available but index is not active (maybe no embeddings?).")
            else:
                print("FAISS not available, using standard search.")
            return recommender # Return the fully initialized recommender
        else:
            print("Failed to load dataset or dataset is empty. Recommender cannot be initialized.")
            return None

    except ImportError as e:
        print(f"ImportError during recommender setup: {e}")
        print("Please ensure all dependencies are installed and the project structure is correct.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during recommender setup: {e}")
        return None 