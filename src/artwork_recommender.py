"""
Implements the ArtworkRecommender class, responsible for loading artwork data,
generating embeddings using CLIP, and providing recommendations based on user emotions.
It utilizes color analysis and emotion mappings to enhance recommendations and
can leverage FAISS for efficient similarity search if installed.
"""

import os
import pickle
import hashlib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from sklearn.cluster import KMeans
from collections import Counter

# Import FAISS availability check from config
from .config import FAISS_AVAILABLE

# Try to import FAISS if available
if FAISS_AVAILABLE:
    import faiss
else:
    faiss = None # Define faiss as None if not available

# === Emotion and Color Mappings ===

# Loads predefined mappings between emotions, descriptive colors, and relevant subjects/styles.
# These mappings are used to generate text prompts for CLIP and explain recommendations.
def load_emotion_mappings():
    """Load emotion mappings to be used by the recommender"""
    emotion_categories = {
        "Positive Emotions": [
            "Joy", "Love", "Serenity", "Amusement", 
            "Gratitude", "Hope", "Admiration"
        ],
        "Negative Emotions": [
            "Sadness", "Anger", "Fear", "Disgust", 
            "Dread", "Confusion", "Anxiety", "Boredom"
        ],
        "Imaginative / Fantastical States": [
            "Dreamy", "Psychedelic", "Mystical", 
            "Spiritual", "Whimsical", "Cosmic"
        ],
        "Blended / Neutral Emotions": [
            "Nostalgia", "Contemplation", "Wonder", 
            "Awe", "Calmness", "Connectedness"
        ]
    }

    emotion_to_colors = {
        "Joy": ["Yellow", "Light Orange", "Peach", "Bright Pink"],
        "Love": ["Red", "Pink", "Warm Beige"],
        "Serenity": ["Sky Blue", "Mint", "Pastel Green"],
        "Amusement": ["Orange", "Lime Green", "Bright Blue"],
        "Gratitude": ["Golden Yellow", "Warm Cream", "Soft Brown"],
        "Hope": ["Sunshine Yellow", "Mint", "Soft Blue"],
        "Admiration": ["Royal Blue", "White", "Lavender"],
        "Sadness": ["Navy", "Cool Blue", "Gray", "Muted Purple"],
        "Anger": ["Red", "Crimson", "Black", "Dark Orange"],
        "Fear": ["Dark Blue", "Charcoal", "Desaturated Teal"],
        "Disgust": ["Sick Green", "Moss", "Brown"],
        "Dread": ["Charcoal Gray", "Dark Green"],
        "Confusion": ["Grayish Purple", "Beige", "Muted Yellow"],
        "Anxiety": ["Pale Green", "Gray", "Off-White"],
        "Boredom": ["Beige", "Dusty Brown", "Pale Gray"],
        "Dreamy": ["Pastel Pink", "Lavender", "Pale Blue", "Cloud White"],
        "Psychedelic": ["Neon Pink", "Cyan", "Magenta", "Yellow-Green"],
        "Mystical": ["Deep Purple", "Indigo", "Silver"],
        "Spiritual": ["Ivory", "Sky Blue", "Gold"],
        "Whimsical": ["Pastel Rainbow", "Bright Mint", "Soft Pink"],
        "Cosmic": ["Black", "Electric Blue", "Star Gold"],
        "Nostalgia": ["Sepia", "Dusty Blue", "Beige", "Olive"],
        "Contemplation": ["Cool Gray", "Deep Blue", "Sage Green"],
        "Wonder": ["Cobalt", "Turquoise", "Light Purple"],
        "Awe": ["Dark Blue", "Emerald", "Shimmering Gold"],
        "Calmness": ["Light Blue", "Soft Green", "Cloud White"],
        "Connectedness": ["Soft Yellow", "Lavender", "Dusty Pink"]
    }

    emotion_to_subjects_styles = {
        "Joy": {
            "subjects": ["flowers", "sunlight", "children", "gardens", "dancing", "celebration"],
            "styles": ["Impressionism", "Bright composition", "Balanced layout"]
        },
        "Sadness": {
            "subjects": ["lonely figure", "rain", "night", "window", "abandoned place"],
            "styles": ["Monochromatic", "Muted tones", "Solitary focus"]
        },
        "Anger": {
            "subjects": ["fire", "chaos", "abstract expression", "explosion", "broken shapes"],
            "styles": ["Expressionism", "Aggressive brushstrokes", "High contrast"]
        },
        "Fear": {
            "subjects": ["shadows", "dark forest", "isolation", "unknown creature"],
            "styles": ["Surrealism", "High chiaroscuro", "Claustrophobic composition"]
        },
        "Serenity": {
            "subjects": ["lake", "clouds", "horizon", "peaceful landscape", "yoga"],
            "styles": ["Watercolor", "Minimalism", "Soft gradients"]
        },
        "Love": {
            "subjects": ["couples", "embrace", "mother and child", "roses"],
            "styles": ["Romanticism", "Warm lighting", "Soft edges"]
        },
        "Nostalgia": {
            "subjects": ["old village", "vintage object", "family photo", "farmhouse"],
            "styles": ["Sepia tone", "Realism", "Still life"]
        },
        "Dreamy": {
            "subjects": ["floating", "moon", "stars", "clouds", "fantasy creature"],
            "styles": ["Symbolism", "Low contrast", "Blurred outlines"]
        },
        "Mystical": {
            "subjects": ["ritual", "temple", "mythical being", "portal", "halo"],
            "styles": ["Byzantine", "Spiritual iconography", "Ornamented design"]
        },
        "Whimsical": {
            "subjects": ["fantasy", "play", "magic", "balloons", "impossible scenes"],
            "styles": ["Naive Art", "Bright pastels", "Asymmetric layout"]
        },
        "Contemplation": {
            "subjects": ["monk", "still water", "reading", "interior room"],
            "styles": ["Muted realism", "Japanese ink painting", "Zen minimalism"]
        },
        "Awe": {
            "subjects": ["cathedrals", "mountains", "cosmic sky", "large scale"],
            "styles": ["Baroque", "Grand scale", "Detailed composition"]
        }
    }
    
    return emotion_categories, emotion_to_colors, emotion_to_subjects_styles

# === Color Extraction Utilities ===

# Extracts the most dominant colors from a PIL Image object using KMeans clustering.
# Resizes the image for faster processing before clustering pixel colors.
def extract_dominant_colors(image, n_colors=3):
    """Extract dominant colors from an image using KMeans clustering"""
    # Resize image to speed up processing
    img = image.copy()
    img = img.resize((100, 100))

    # Convert to numpy array and reshape
    img_array = np.array(img)
    img_array = img_array.reshape((img_array.shape[0] * img_array.shape[1], 3))

    # Use KMeans to find dominant colors
    # Added n_init='auto' to suppress future warning
    kmeans = KMeans(n_clusters=n_colors, n_init='auto', random_state=0) 
    kmeans.fit(img_array)

    # Get RGB values of the centroids
    colors = kmeans.cluster_centers_

    # Convert to integers
    colors = colors.astype(int)

    # Get counts of pixels in each cluster
    labels = kmeans.labels_
    count = Counter(labels)

    # Sort colors by count
    sorted_colors = [colors[i] for i in sorted(count, key=count.get, reverse=True)]

    return sorted_colors

# Converts an RGB color tuple to the name of the closest predefined color in a dictionary.
# Uses Euclidean distance in RGB space to find the nearest match.
def rgb_to_named_color(rgb):
    """Convert RGB value to closest named color"""
    # Dictionary of RGB values for named colors (copied from notebook)
    color_dict = {
        "Beige": [245, 245, 220], "Black": [0, 0, 0], "Bright Blue": [0, 153, 255], 
        "Bright Mint": [189, 252, 201], "Bright Pink": [255, 105, 180], "Brown": [165, 42, 42], 
        "Charcoal": [54, 69, 79], "Charcoal Gray": [80, 80, 80], "Cloud White": [255, 255, 250], 
        "Cobalt": [0, 71, 171], "Cool Blue": [100, 149, 237], "Cool Gray": [140, 146, 172], 
        "Crimson": [220, 20, 60], "Cyan": [0, 255, 255], "Dark Blue": [0, 0, 139], 
        "Dark Green": [0, 100, 0], "Dark Orange": [255, 140, 0], "Deep Blue": [0, 0, 139], 
        "Deep Purple": [102, 0, 153], "Desaturated Teal": [88, 110, 117], "Dusty Blue": [96, 130, 182], 
        "Dusty Brown": [170, 120, 100], "Dusty Pink": [203, 144, 153], "Electric Blue": [125, 249, 255], 
        "Emerald": [80, 200, 120], "Gold": [255, 215, 0], "Golden Yellow": [255, 223, 0], 
        "Gray": [128, 128, 128], "Grayish Purple": [145, 130, 150], "Indigo": [75, 0, 130], 
        "Ivory": [255, 255, 240], "Lavender": [230, 230, 250], "Light Blue": [173, 216, 230], 
        "Light Orange": [255, 204, 153], "Light Purple": [200, 162, 200], "Lime Green": [50, 205, 50], 
        "Magenta": [255, 0, 255], "Mint": [170, 255, 195], "Moss": [113, 139, 94], 
        "Muted Purple": [147, 112, 219], "Muted Yellow": [240, 230, 140], "Navy": [0, 0, 128], 
        "Neon Pink": [255, 20, 147], "Off-White": [250, 250, 240], "Olive": [128, 128, 0], 
        "Orange": [255, 165, 0], "Pale Blue": [175, 238, 238], "Pale Gray": [211, 211, 211], 
        "Pale Green": [152, 251, 152], "Pastel Pink": [255, 209, 220], "Pastel Rainbow": [255, 204, 229], 
        "Peach": [255, 229, 180], "Pink": [255, 192, 203], "Red": [255, 0, 0], 
        "Royal Blue": [65, 105, 225], "Sage Green": [157, 193, 131], "Sepia": [112, 66, 20], 
        "Shimmering Gold": [255, 236, 139], "Sick Green": [148, 191, 77], "Silver": [192, 192, 192], 
        "Sky Blue": [135, 206, 235], "Soft Blue": [173, 216, 230], "Soft Brown": [181, 101, 29], 
        "Soft Green": [144, 238, 144], "Soft Pink": [255, 182, 193], "Soft Yellow": [255, 255, 204], 
        "Star Gold": [255, 215, 0], "Sunshine Yellow": [255, 247, 0], "Turquoise": [64, 224, 208], 
        "Warm Beige": [222, 200, 150], "Warm Cream": [255, 253, 208], "White": [255, 255, 255], 
        "Yellow": [255, 255, 0], "Yellow-Green": [154, 205, 50]
    }

    min_dist = float('inf')
    closest_color = None
    
    for color_name, color_rgb in color_dict.items():
        dist = np.sqrt(sum([(a - b) ** 2 for a, b in zip(rgb, color_rgb)]))
        if dist < min_dist:
            min_dist = dist
            closest_color = color_name

    return closest_color

# === Dataset Loading Functions ===

# Loads artwork metadata from a CSV file and associates it with image paths.
# Specifically designed for datasets where \'Object ID\' links CSV rows to image filenames.
# Checks for the existence of image files and filters the DataFrame accordingly.
def load_art_dataset(csv_path, image_folder, image_extension=".jpg"):
    """
    Load art dataset from a CSV file.
    Customized for the MetMuseum dataset which uses Object ID as identifiers.
    """
    print(f"Loading CSV from: {csv_path}")
    try:
        # Explicitly specify UTF-8 encoding for proper handling of non-Latin characters
        df = pd.read_csv(csv_path, encoding='utf-8')
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        return None
    except Exception as e:
        print(f"Error reading CSV file {csv_path}: {e}")
        return None

    if 'Object ID' not in df.columns:
        print(f"Error: 'Object ID' column not found in {csv_path}")
        return None

    # Create image filename from Object ID
    df['image_path'] = df['Object ID'].astype(str).apply(
        lambda x: os.path.join(image_folder, f"{x.strip()}{image_extension}")
    )
    
    # Check if images exist (with progress bar for large datasets)
    print(f"Checking if images exist in {image_folder}...")
    image_exists = []
    for path in tqdm(df['image_path'], desc="Checking images"):
        image_exists.append(os.path.isfile(path))
    
    df['image_exists'] = image_exists
    
    # Filter to only include rows with existing images
    if df['image_exists'].sum() == 0:
        print(f"WARNING: No images found in {image_folder} matching the Object IDs in {csv_path}!")
        # Return an empty DataFrame with the expected structure if no images are found
        return pd.DataFrame(columns=df.columns).drop(columns=['image_exists'])
    else:
        print(f"Found {df['image_exists'].sum()} images out of {len(df)} records")
    
    filtered_df = df[df['image_exists']].copy()
    filtered_df = filtered_df.drop(columns=['image_exists'])
    
    return filtered_df

# Placeholder/Example function to create a dataset directly from image files in a folder.
# (Implementation details are missing in the provided snippet)
def create_example_dataset(image_folder):
    """Create a simple dataset from a folder of images"""
    print(f"Creating dataset from images in: {image_folder}")
    if not os.path.isdir(image_folder):
        print(f"Error: Image folder not found at {image_folder}")
        return None
        
    image_files = [f for f in os.listdir(image_folder) 
                 if os.path.isfile(os.path.join(image_folder, f)) and 
                 f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"No image files found in {image_folder}")
        return pd.DataFrame(columns=['Object ID', 'Title', 'image_path', 'Artist Display Name', 'Classification'])

    print(f"Found {len(image_files)} images")
    
    # Extract object IDs from filenames (assuming they're at the start of the filename)
    object_ids = []
    for filename in image_files:
        # Try to extract a numeric object ID from the filename
        parts = filename.split('.')
        name_part = parts[0]
        # Extract digits if they exist at the start
        digits = ""
        for char in name_part:
            if char.isdigit():
                digits += char
            else:
                break
        
        if digits:
            object_ids.append(int(digits))
        else:
            # Use index if no digits found - make it unique by adding a prefix maybe?
            object_ids.append(f"idx_{len(object_ids)}") 
    
    data = {
        'Object ID': object_ids,
        'Title': [os.path.splitext(f)[0] for f in image_files],
        'image_path': [os.path.join(image_folder, f) for f in image_files],
        'Artist Display Name': ['Unknown'] * len(image_files),
        'Classification': ['Unknown'] * len(image_files)
    }
    
    return pd.DataFrame(data)


# === Artwork Recommender Class ===

# Core class for handling artwork recommendations based on emotional prompts.
# Manages data loading, embedding generation (CLIP), caching, and similarity search (FAISS or standard).
class ArtworkRecommender:
    """Recommends artworks based on emotional prompts using CLIP embeddings."""

    # Initializes the recommender, loading the CLIP model and processor.
    # Sets up the cache directory and determines the computation device (GPU/CPU).
    def __init__(self, model_name="openai/clip-vit-base-patch32", cache_dir="./emotion_art_cache"):
        """
        Initialize the recommender with a specific CLIP model and cache directory.
        
        Args:
            model_name: CLIP model to use (smaller models are faster)
            cache_dir: Directory to store cached embeddings
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load emotion mappings
        self.emotion_categories, self.emotion_to_colors, self.emotion_to_subjects_styles = load_emotion_mappings()
        
        # Setup device
        self.device = self._get_device()
        print(f"Using device: {self.device}")
        
        # Load model
        self.model_name = model_name
        print(f"Loading CLIP model: {model_name}")
        try:
            self.model = CLIPModel.from_pretrained(model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(model_name)
        except Exception as e:
            print(f"Error loading CLIP model {model_name}: {e}")
            raise # Re-raise exception as model is critical
        
        # Initialize embeddings cache
        self.embeddings_cache = {}
        self.embedding_dimensions = None
        self.faiss_index = None
        self.all_ids = [] # Should store DataFrame indices, not Object IDs
        self.artwork_df = None
    
    # Determines the appropriate device (CUDA GPU or CPU) for model computations.
    def _get_device(self):
        """Get the best available device for processing"""
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    
    # Loads the dataset from CSV and image folder, manages embedding generation/caching,
    # and optionally builds a FAISS index.
    def load_dataset(self, csv_path, image_folder, image_extension=".jpg", rebuild_cache=False):
        """
        Load artwork dataset and prepare for recommendations
        
        Args:
            csv_path: Path to CSV file with artwork metadata
            image_folder: Folder containing artwork images
            image_extension: File extension of the images
            rebuild_cache: Whether to force rebuilding the cache
            
        Returns:
            DataFrame with artwork data
        """
        # Load dataset
        self.artwork_df = load_art_dataset(csv_path, image_folder, image_extension)
        if self.artwork_df is None or self.artwork_df.empty:
             print("Artwork DataFrame is empty or failed to load. Cannot proceed.")
             # Ensure df is an empty DataFrame with expected columns if it failed/is empty
             if self.artwork_df is None: 
                 self.artwork_df = pd.DataFrame(columns=['Object ID', 'Title', 'image_path', 'Artist Display Name', 'Classification']) # Add other expected cols
             return self.artwork_df # Return empty df

        print(f"Loaded {len(self.artwork_df)} artworks with images")
        
        # Set cache filename based on model and data source
        hash_str = hashlib.md5(f"{csv_path}_{image_folder}_{self.model_name}".encode()).hexdigest()[:10]
        cache_path = os.path.join(self.cache_dir, f"image_embeddings_{hash_str}.pkl")
        
        # Try to load cached embeddings
        self.embeddings_cache = {} # Reset cache before loading/generating
        if os.path.exists(cache_path) and not rebuild_cache:
            print(f"Loading cached embeddings from {cache_path}")
            try:
                with open(cache_path, 'rb') as f:
                    self.embeddings_cache = pickle.load(f)
                print(f"Loaded {len(self.embeddings_cache)} cached embeddings")
            except Exception as e:
                print(f"Error loading cache: {e}. Will regenerate.")
                self.embeddings_cache = {}
        
        # Compare cache with dataset and find missing
        all_paths = set(self.artwork_df['image_path'])
        cached_paths = set(self.embeddings_cache.keys())
        
        # Prune cache: remove entries not in the current dataset df
        paths_to_remove = cached_paths - all_paths
        if paths_to_remove:
             print(f"Removing {len(paths_to_remove)} stale entries from cache.")
             for path in paths_to_remove:
                 del self.embeddings_cache[path]
        
        # Identify paths needing generation
        missing_paths = all_paths - set(self.embeddings_cache.keys()) # Recalculate after pruning
        missing = len(missing_paths)
        
        # Generate missing embeddings if needed
        if missing > 0:
            print(f"Generating {missing} missing embeddings...")
            self._generate_missing_embeddings(cache_path, list(missing_paths)) # Pass as list
        elif not self.embeddings_cache and len(all_paths) > 0:
             print("Cache is empty, generating all embeddings...")
             self._generate_missing_embeddings(cache_path, list(all_paths)) # Pass as list
        else:
             print("All required embeddings found in cache.")

        # Ensure embedding dimension is set if cache was loaded
        if self.embeddings_cache and self.embedding_dimensions is None:
             first_key = next(iter(self.embeddings_cache))
             self.embedding_dimensions = self.embeddings_cache[first_key].shape[1]
             print(f"Embedding dimension set from cache: {self.embedding_dimensions}")

        # Create similarity search index
        if FAISS_AVAILABLE:
            self._create_faiss_index()
        else:
            print("FAISS not available, using slower similarity search")
        
        return self.artwork_df
    
    # Generates an embedding for a single image file using the loaded CLIP model.
    # Handles image opening, resizing, preprocessing, and potential errors.
    def _get_image_embedding(self, image_path, max_size=224):
        """Generate CLIP embedding for an image with size limitation"""
        try:
            # Open image
            image = Image.open(image_path).convert("RGB")
            
            # Resize large images to reduce memory usage and speed up processing
            width, height = image.size
            if width > max_size or height > max_size:
                if width > height:
                    new_width = max_size
                    new_height = int(height * (max_size / width))
                else:
                    new_height = max_size
                    new_width = int(width * (max_size / height))
                image = image.resize((new_width, new_height))
            
            # Process the image using the CLIP processor
            inputs = self.processor(
                text=None,
                images=image,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            # Generate image embeddings
            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)
                # Normalize the embeddings
                image_embedding = F.normalize(outputs, p=2, dim=1)
            
            return image_embedding
        except FileNotFoundError:
             print(f"Error: Image file not found at {image_path}")
             return None
        except Exception as e:
            # Handle other errors for corrupted images etc.
            print(f"Error processing {image_path}: {e}")
            return None
    
    # Generates embeddings in batches for images whose embeddings are missing from the cache.
    # Saves the newly generated embeddings back to the cache file.
    def _generate_missing_embeddings(self, cache_path, paths_to_process, batch_size=16):
        """
        Generate embeddings for images not in cache, in batches to save memory
        
        Args:
            cache_path: Path to save cache
            paths_to_process: List of paths to generate embeddings for
            batch_size: Number of images to process in each batch
        """
        if not paths_to_process:
            print("No missing paths to process.")
            return

        # Process in batches
        newly_generated_count = 0
        for i in tqdm(range(0, len(paths_to_process), batch_size), desc="Processing batches"):
            batch_paths = paths_to_process[i:i+batch_size]
            
            for image_path in tqdm(batch_paths, desc=f"Batch {i//batch_size+1}", leave=False):
                if image_path not in self.embeddings_cache: # Double check if generated in parallel somehow
                    try:
                        image_embedding = self._get_image_embedding(image_path)
                        if image_embedding is not None:
                            embedding_numpy = image_embedding.cpu().numpy()
                            self.embeddings_cache[image_path] = embedding_numpy
                            newly_generated_count += 1
                            
                            # Update embedding dimension if not set
                            if self.embedding_dimensions is None:
                                self.embedding_dimensions = embedding_numpy.shape[1]
                                print(f"Embedding dimension set: {self.embedding_dimensions}")
                    except Exception as e:
                        print(f"Error processing image {image_path}: {e}")
            
            # Save cache periodically to prevent data loss on long runs
            if newly_generated_count > 0 and newly_generated_count % (batch_size * 5) == 0: # Save every 5 batches approx
                try:
                    with open(cache_path, 'wb') as f:
                        pickle.dump(self.embeddings_cache, f)
                    print(f"Saved checkpoint with {len(self.embeddings_cache)} embeddings")
                except Exception as e:
                     print(f"Error saving cache checkpoint: {e}")
        
        # Save final updated cache
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(self.embeddings_cache, f)
            print(f"Finished generation. Saved final cache with {len(self.embeddings_cache)} total embeddings.")
        except Exception as e:
             print(f"Error saving final cache: {e}")
    
    # Creates a descriptive text prompt based on an emotion, using predefined mappings.
    # This prompt is used as input to the CLIP text encoder.
    def _generate_emotion_prompt(self, emotion):
        """Create a detailed text prompt for CLIP based on emotion"""
        prompt = f"An artwork that expresses the emotion of {emotion}. "

        # Add subjects
        if emotion in self.emotion_to_subjects_styles:
            subjects = self.emotion_to_subjects_styles[emotion].get("subjects", []) # Use .get for safety
            if subjects:
                subject_str = ", ".join(subjects[:3])  # Take first 3 subjects
                prompt += f"It might contain {subject_str}. "

        # Add styles
        if emotion in self.emotion_to_subjects_styles:
            styles = self.emotion_to_subjects_styles[emotion].get("styles", []) # Use .get for safety
            if styles:
                style_str = ", ".join(styles[:2])  # Take first 2 styles
                prompt += f"In a style that is {style_str}. "

        # Add colors
        if emotion in self.emotion_to_colors:
            colors = self.emotion_to_colors.get(emotion, []) # Use .get for safety
            if colors:
                color_str = ", ".join(colors[:3]) # Take first 3 colors
                prompt += f"Using colors like {color_str}."

        return prompt.strip() # Remove trailing space/newline
    
    # Generates an embedding for a given text string using the loaded CLIP model.
    def _get_text_embedding(self, text):
        """Generate CLIP embedding for text"""
        # Process the text using the CLIP processor
        inputs = self.processor(
            text=[text],
            images=None,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        # Generate text embeddings
        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)
            # Normalize the embeddings
            text_embedding = F.normalize(outputs, p=2, dim=1)
        
        return text_embedding
    
    # Creates and populates a FAISS index for efficient similarity search if FAISS is available.
    # Handles index creation, training (if necessary), and adding embeddings.
    def _create_faiss_index(self):
        """Create FAISS index from cached embeddings for fast similarity search"""
        if not FAISS_AVAILABLE or faiss is None:
            print("FAISS not available or not imported, cannot create index.")
            self.faiss_index = None
            return
            
        # Check if we have a DataFrame and embeddings
        if self.artwork_df is None or self.artwork_df.empty:
             print("Artwork data is not loaded or is empty. Cannot build FAISS index.")
             self.faiss_index = None
             return
        if not self.embeddings_cache:
            print("No embeddings available to create index")
            self.faiss_index = None
            return
        if self.embedding_dimensions is None:
            print("Embedding dimension not determined. Cannot create FAISS index.")
            self.faiss_index = None
            return
            
        # Initialize index
        print(f"Creating FAISS index with dimension {self.embedding_dimensions}")
        # Using IndexFlatIP for cosine similarity (requires normalized vectors)
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dimensions) 
        
        # Prepare embeddings and corresponding DataFrame indices
        all_embeddings_list = []
        self.all_ids = [] # Stores the DataFrame index
        
        # Iterate through the DataFrame to ensure order matches all_ids
        for idx, row in tqdm(self.artwork_df.iterrows(), total=len(self.artwork_df), desc="Building FAISS index"):
            image_path = row['image_path']
            if image_path in self.embeddings_cache:
                embedding = self.embeddings_cache[image_path]
                if embedding is not None and embedding.shape == (1, self.embedding_dimensions):
                    self.all_ids.append(idx) # Store the DataFrame index
                    all_embeddings_list.append(embedding[0].astype('float32')) # Add the 1D embedding vector
                # else: # Optional: log problematic embeddings
                #      print(f"Skipping embedding for {image_path}: shape {embedding.shape if embedding is not None else 'None'}")

        
        if all_embeddings_list:
            all_embeddings_np = np.array(all_embeddings_list)
             # Ensure vectors are normalized (CLIP output should be, but double-check)
            faiss.normalize_L2(all_embeddings_np)
            self.faiss_index.add(all_embeddings_np)
            print(f"Created FAISS index with {self.faiss_index.ntotal} vectors")
        else:
            print("No valid embeddings found to add to the FAISS index")
            self.faiss_index = None # Reset index if empty
    
    # Main method to get artwork recommendations for a given emotion.
    # Generates an emotion prompt, gets its embedding, and performs similarity search
    # using FAISS (if available) or standard cosine similarity.
    # Optionally enhances results with dominant color information.
    def get_recommendations(self, user_emotion, top_n=5, enhance_with_colors=True):
        """
        Get artwork recommendations based on emotion
        
        Args:
            user_emotion: Emotion to search for
            top_n: Number of recommendations to return
            enhance_with_colors: Whether to add color information
            
        Returns:
            Tuple: (DataFrame with recommendations, List of (index, score) tuples) 
                   or (None, None) if errors occur.
        """
        # Check for dataset
        if self.artwork_df is None or self.artwork_df.empty:
            print("Error: No artwork data loaded. Call load_dataset() first.")
            return None, None
            
        # Validate the emotion
        all_emotions = [e for sublist in self.emotion_categories.values() for e in sublist]
        if user_emotion not in all_emotions:
            print(f"Error: Invalid emotion '{user_emotion}'. Valid emotions are: {all_emotions}")
            return None, None
        
        # Generate emotion prompt
        emotion_prompt = self._generate_emotion_prompt(user_emotion)
        print(f"Generated prompt: '{emotion_prompt}'")
        
        # Get text embedding for the emotion
        emotion_embedding = self._get_text_embedding(emotion_prompt)
        
        # Get recommendations based on similarity
        recommendations_df = None
        similarities_list = None
        
        if FAISS_AVAILABLE and self.faiss_index is not None and self.faiss_index.ntotal > 0:
            # Use FAISS for fast similarity search
            print("Using FAISS for similarity search.")
            recommendations_df, similarities_list = self._get_recommendations_faiss(emotion_embedding, top_n)
        else:
            # Fallback to standard similarity search
            if not FAISS_AVAILABLE:
                 print("FAISS not available. Using standard similarity search (might be slow).")
            elif self.faiss_index is None or self.faiss_index.ntotal == 0:
                 print("FAISS index not ready or empty. Using standard similarity search (might be slow).")
                 
            recommendations_df, similarities_list = self._get_recommendations_standard(emotion_embedding, top_n)
        
        # Check if recommendations were found
        if recommendations_df is None or recommendations_df.empty:
             print(f"No recommendations found for emotion '{user_emotion}'.")
             return None, None

        # Enhance with color analysis
        if enhance_with_colors:
             # Pass only the df part to enhance_with_colors
             recommendations_enhanced_df = self._enhance_with_colors(recommendations_df) 
             if recommendations_enhanced_df is not None:
                  recommendations_df = recommendations_enhanced_df # Update if enhancement was successful
        
        # Add the emotion used for these recommendations as a column
        recommendations_df['emotion_used'] = user_emotion
        
        # Store the emotion used as an attribute (less conventional, maybe remove)
        # recommendations_df._emotion_used = user_emotion 
        
        return recommendations_df, similarities_list # Return DF and list of (index, score)
    
    # Performs similarity search using the pre-built FAISS index.
    # Returns the top N closest artwork indices and their distances.
    def _get_recommendations_faiss(self, emotion_embedding, top_n):
        """Get recommendations using FAISS similarity search"""
        if self.faiss_index is None or self.faiss_index.ntotal == 0:
            print("FAISS index is not available or empty.")
            return None, None
            
        emotion_embedding_np = emotion_embedding.cpu().numpy().astype('float32')
        # Ensure query vector is normalized for IndexFlatIP
        faiss.normalize_L2(emotion_embedding_np)
        
        # Make sure we don't request more items than we have in the index
        available_count = self.faiss_index.ntotal
        request_count = min(top_n, available_count)
        
        # Search the index
        try:
            D, I = self.faiss_index.search(emotion_embedding_np, request_count)
        except Exception as e:
             print(f"Error during FAISS search: {e}")
             return None, None

        # Convert FAISS results to original dataframe indices
        if len(I) > 0 and len(I[0]) > 0:
            try:
                # Map FAISS indices (positions in the index) to DataFrame indices (stored in self.all_ids)
                # I[0] contains the indices within the FAISS index structure
                faiss_indices = I[0] 
                # Filter out potential -1 indices if search returns fewer than k results cleanly
                valid_faiss_indices = [i for i in faiss_indices if i != -1] 
                
                if not valid_faiss_indices:
                     print("FAISS search returned no valid indices.")
                     return None, None

                # Get corresponding DataFrame indices from our mapping
                top_df_indices = [self.all_ids[i] for i in valid_faiss_indices] 
                
                # Get scores for the valid indices
                scores = [D[0][j] for j, i in enumerate(faiss_indices) if i != -1]
                
                similarities = list(zip(top_df_indices, scores))
                
                # Use loc with the collected DataFrame indices
                return self.artwork_df.loc[top_df_indices].copy(), similarities 
            except IndexError as e:
                 print(f"Error mapping FAISS indices to DataFrame indices: {e}. Index size: {len(self.all_ids)}, FAISS indices: {I[0]}")
                 return None, None
            except Exception as e:
                print(f"Error processing FAISS results: {e}")
                return None, None
        else:
            print("FAISS search returned no results")
            return None, None
    
    # Performs similarity search using standard cosine similarity calculation.
    # Computes similarity between the emotion embedding and all artwork embeddings.
    # Returns the top N closest artwork indices and their scores.
    def _get_recommendations_standard(self, emotion_embedding, top_n):
        """Get recommendations using standard similarity calculation"""
        similarities = []
        
        # Process each artwork using cached embeddings
        for df_idx, row in tqdm(self.artwork_df.iterrows(), total=len(self.artwork_df), desc="Calculating similarities"):
            try:
                # Get image embedding from cache
                image_path = row['image_path']
                if image_path in self.embeddings_cache:
                    embedding_np = self.embeddings_cache[image_path]
                    if embedding_np is not None:
                        # Convert back to tensor 
                        image_embedding = torch.tensor(embedding_np).to(self.device)
                        
                        # Ensure embeddings are suitable for cosine similarity (e.g., shape [1, dim])
                        if image_embedding.shape == emotion_embedding.shape:
                             # Calculate similarity (expects batch dim, adds it if missing)
                             sim = F.cosine_similarity(emotion_embedding, image_embedding).item()
                             # Store result with DataFrame index
                             similarities.append((df_idx, sim))
                        # else: # Optional: log shape mismatch
                        #      print(f"Shape mismatch: query {emotion_embedding.shape}, image {image_embedding.shape} for {image_path}")

            except Exception as e:
                print(f"Error processing image {row['image_path']} during standard similarity: {e}")
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get top N results
        if similarities:
            top_results = similarities[:top_n]
            top_indices = [idx for idx, _ in top_results]
             # Use .loc[] with the list of indices to preserve original index
            return self.artwork_df.loc[top_indices].copy(), top_results
        else:
            print("No valid similarities found during standard calculation")
            return None, None
    
    # Enhances the recommendation DataFrame by adding dominant color information
    # for each recommended artwork. Extracts colors from images and maps them to names.
    def _enhance_with_colors(self, recommendations_df):
        """Add dominant color information to recommendations DataFrame"""
        if recommendations_df is None or recommendations_df.empty:
             return recommendations_df # Return original if empty or None

        dominant_colors_list = []
        named_colors_list = []

        for idx, row in recommendations_df.iterrows():
            dominant_colors = []
            named_colors = []
            try:
                img = Image.open(row['image_path']).convert("RGB")
                dominant_colors_rgb = extract_dominant_colors(img)
                dominant_colors = [list(map(int, c)) for c in dominant_colors_rgb] # Store as list of lists
                named_colors = [rgb_to_named_color(color) for color in dominant_colors_rgb]
            except FileNotFoundError:
                 print(f"Image not found for color extraction: {row['image_path']}")
            except Exception as e:
                print(f"Error processing colors for {row['image_path']}: {e}")
            
            dominant_colors_list.append(dominant_colors) # List of RGB lists
            named_colors_list.append(named_colors) # List of named colors

        # Add new columns to the DataFrame copy
        recommendations_df['dominant_colors_rgb'] = dominant_colors_list
        recommendations_df['dominant_colors_named'] = named_colors_list
        
        return recommendations_df 