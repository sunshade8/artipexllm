"""
Optimized Artwork Recommender System

This module provides an efficient implementation of the emotion-based
artwork recommendation system with caching and fast similarity search.
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from sklearn.cluster import KMeans
from collections import Counter
import unicodedata
import locale
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt

# Try to import FAISS (optional but recommended)
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    print("FAISS not available. Using slower similarity search.")
    print("Install FAISS for faster similarity search with: pip install faiss-cpu")
    FAISS_AVAILABLE = False

# Try to get the system locale and set it
try:
    locale.setlocale(locale.LC_ALL, '')
except:
    pass

# Configure Matplotlib for CJK text support
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
        pass  # Regular Python interpreter
    
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

# Call this at import time
IN_NOTEBOOK = configure_matplotlib_for_cjk()

# Color and emotion mappings from the original notebook
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


# Color extraction functions
def extract_dominant_colors(image, n_colors=3):
    """Extract dominant colors from an image using KMeans clustering"""
    # Resize image to speed up processing
    img = image.copy()
    img = img.resize((100, 100))

    # Convert to numpy array and reshape
    img_array = np.array(img)
    img_array = img_array.reshape((img_array.shape[0] * img_array.shape[1], 3))

    # Use KMeans to find dominant colors
    kmeans = KMeans(n_clusters=n_colors)
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


def rgb_to_named_color(rgb):
    """Convert RGB value to closest named color"""
    # Dictionary of RGB values for named colors
    color_dict = {
        "Beige": [245, 245, 220],
        "Black": [0, 0, 0],
        "Bright Blue": [0, 153, 255],
        "Bright Mint": [189, 252, 201],
        "Bright Pink": [255, 105, 180],
        "Brown": [165, 42, 42],
        "Charcoal": [54, 69, 79],
        "Charcoal Gray": [80, 80, 80],
        "Cloud White": [255, 255, 250],
        "Cobalt": [0, 71, 171],
        "Cool Blue": [100, 149, 237],
        "Cool Gray": [140, 146, 172],
        "Crimson": [220, 20, 60],
        "Cyan": [0, 255, 255],
        "Dark Blue": [0, 0, 139],
        "Dark Green": [0, 100, 0],
        "Dark Orange": [255, 140, 0],
        "Deep Blue": [0, 0, 139],
        "Deep Purple": [102, 0, 153],
        "Desaturated Teal": [88, 110, 117],
        "Dusty Blue": [96, 130, 182],
        "Dusty Brown": [170, 120, 100],
        "Dusty Pink": [203, 144, 153],
        "Electric Blue": [125, 249, 255],
        "Emerald": [80, 200, 120],
        "Gold": [255, 215, 0],
        "Golden Yellow": [255, 223, 0],
        "Gray": [128, 128, 128],
        "Grayish Purple": [145, 130, 150],
        "Indigo": [75, 0, 130],
        "Ivory": [255, 255, 240],
        "Lavender": [230, 230, 250],
        "Light Blue": [173, 216, 230],
        "Light Orange": [255, 204, 153],
        "Light Purple": [200, 162, 200],
        "Lime Green": [50, 205, 50],
        "Magenta": [255, 0, 255],
        "Mint": [170, 255, 195],
        "Moss": [113, 139, 94],
        "Muted Purple": [147, 112, 219],
        "Muted Yellow": [240, 230, 140],
        "Navy": [0, 0, 128],
        "Neon Pink": [255, 20, 147],
        "Off-White": [250, 250, 240],
        "Olive": [128, 128, 0],
        "Orange": [255, 165, 0],
        "Pale Blue": [175, 238, 238],
        "Pale Gray": [211, 211, 211],
        "Pale Green": [152, 251, 152],
        "Pastel Pink": [255, 209, 220],
        "Pastel Rainbow": [255, 204, 229],
        "Peach": [255, 229, 180],
        "Pink": [255, 192, 203],
        "Red": [255, 0, 0],
        "Royal Blue": [65, 105, 225],
        "Sage Green": [157, 193, 131],
        "Sepia": [112, 66, 20],
        "Shimmering Gold": [255, 236, 139],
        "Sick Green": [148, 191, 77],
        "Silver": [192, 192, 192],
        "Sky Blue": [135, 206, 235],
        "Soft Blue": [173, 216, 230],
        "Soft Brown": [181, 101, 29],
        "Soft Green": [144, 238, 144],
        "Soft Pink": [255, 182, 193],
        "Soft Yellow": [255, 255, 204],
        "Star Gold": [255, 215, 0],
        "Sunshine Yellow": [255, 247, 0],
        "Turquoise": [64, 224, 208],
        "Warm Beige": [222, 200, 150],
        "Warm Cream": [255, 253, 208],
        "White": [255, 255, 255],
        "Yellow": [255, 255, 0],
        "Yellow-Green": [154, 205, 50]
    }

    # Calculate Euclidean distance between RGB values
    min_dist = float('inf')
    closest_color = None
    
    for color_name, color_rgb in color_dict.items():
        dist = np.sqrt(sum([(a - b) ** 2 for a, b in zip(rgb, color_rgb)]))
        if dist < min_dist:
            min_dist = dist
            closest_color = color_name

    return closest_color


# Dataset Loader Functions
def load_art_dataset(csv_path, image_folder, image_extension=".jpg"):
    """
    Load art dataset from a CSV file.
    Customized for the MetMuseum dataset which uses Object ID as identifiers.
    """
    print(f"Loading CSV from: {csv_path}")
    # Explicitly specify UTF-8 encoding for proper handling of non-Latin characters
    df = pd.read_csv(csv_path, encoding='utf-8')
    
    # Create image filename from Object ID
    df['image_path'] = df['Object ID'].astype(str).apply(
        lambda x: os.path.join(image_folder, f"{x.strip()}{image_extension}")
    )
    
    # Check if images exist (with progress bar for large datasets)
    print("Checking if images exist...")
    image_exists = []
    for path in tqdm(df['image_path'], desc="Checking images"):
        image_exists.append(os.path.isfile(path))
    
    df['image_exists'] = image_exists
    
    # Filter to only include rows with existing images
    if df['image_exists'].sum() == 0:
        print(f"WARNING: No images found in {image_folder} matching the Object IDs!")
    else:
        print(f"Found {df['image_exists'].sum()} images out of {len(df)} records")
    
    filtered_df = df[df['image_exists']].copy()
    filtered_df = filtered_df.drop(columns=['image_exists'])
    
    return filtered_df


def create_example_dataset(image_folder):
    """Create a simple dataset from a folder of images"""
    print(f"Creating dataset from images in: {image_folder}")
    image_files = [f for f in os.listdir(image_folder) 
                 if os.path.isfile(os.path.join(image_folder, f)) and 
                 f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
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
            # Use index if no digits found
            object_ids.append(len(object_ids))
    
    data = {
        'Object ID': object_ids,
        'Title': [f.split('.')[0] for f in image_files],
        'image_path': [os.path.join(image_folder, f) for f in image_files],
        'Artist Display Name': ['Unknown'] * len(image_files),
        'Classification': ['Unknown'] * len(image_files)
    }
    
    return pd.DataFrame(data)


class ArtworkRecommender:
    """
    Optimized artwork recommendation system with caching and fast similarity search.
    
    Features:
    - Caches CLIP embeddings to disk for faster repeated use
    - Uses FAISS for fast similarity search (if available)
    - Resizes images for faster processing
    - Batched processing to reduce memory usage
    """
    
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
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        # Initialize embeddings cache
        self.embeddings_cache = {}
        self.embedding_dimensions = None
        self.faiss_index = None
        self.all_ids = []
        self.artwork_df = None
    
    def _get_device(self):
        """Get the best available device for processing"""
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    
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
        print(f"Loaded {len(self.artwork_df)} artworks")
        
        # Set cache filename based on model and data source
        import hashlib
        hash_str = hashlib.md5(f"{csv_path}_{image_folder}_{self.model_name}".encode()).hexdigest()[:10]
        cache_path = os.path.join(self.cache_dir, f"image_embeddings_{hash_str}.pkl")
        
        # Try to load cached embeddings
        if os.path.exists(cache_path) and not rebuild_cache:
            print(f"Loading cached embeddings from {cache_path}")
            try:
                with open(cache_path, 'rb') as f:
                    self.embeddings_cache = pickle.load(f)
                print(f"Loaded {len(self.embeddings_cache)} cached embeddings")
            except Exception as e:
                print(f"Error loading cache: {e}")
                self.embeddings_cache = {}
        
        # Compare cache with dataset
        missing = 0
        all_paths = set(self.artwork_df['image_path'])
        cached_paths = set(self.embeddings_cache.keys())
        missing_paths = all_paths - cached_paths
        missing = len(missing_paths)
        
        # Generate missing embeddings if needed
        if missing > 0:
            print(f"Generating {missing} missing embeddings...")
            self._generate_missing_embeddings(cache_path, missing_paths)
        
        # Create similarity search index
        if FAISS_AVAILABLE:
            self._create_faiss_index()
        else:
            print("FAISS not available, using slower similarity search")
        
        return self.artwork_df
    
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
        except Exception as e:
            # Handle errors for corrupted images
            print(f"Error processing {image_path}: {e}")
            return None
    
    def _generate_missing_embeddings(self, cache_path, missing_paths=None, batch_size=16):
        """
        Generate embeddings for images not in cache, in batches to save memory
        
        Args:
            cache_path: Path to save cache
            missing_paths: Set of paths to generate embeddings for
            batch_size: Number of images to process in each batch
        """
        # Filter to only missing paths if provided
        if missing_paths:
            paths_to_process = list(missing_paths)
        else:
            paths_to_process = [row['image_path'] for _, row in self.artwork_df.iterrows() 
                           if row['image_path'] not in self.embeddings_cache]
        
        # Process in batches
        for i in tqdm(range(0, len(paths_to_process), batch_size), desc="Processing batches"):
            batch_paths = paths_to_process[i:i+batch_size]
            
            for image_path in tqdm(batch_paths, desc=f"Batch {i//batch_size+1}", leave=False):
                if image_path not in self.embeddings_cache:
                    try:
                        image_embedding = self._get_image_embedding(image_path)
                        if image_embedding is not None:
                            self.embeddings_cache[image_path] = image_embedding.cpu().numpy()
                            
                            # Update embedding dimension if not set
                            if self.embedding_dimensions is None:
                                self.embedding_dimensions = image_embedding.shape[1]
                    except Exception as e:
                        print(f"Error processing image {image_path}: {e}")
            
            # Save cache after each batch to prevent data loss
            if i % (batch_size * 10) == 0 and i > 0:
                with open(cache_path, 'wb') as f:
                    pickle.dump(self.embeddings_cache, f)
                print(f"Saved checkpoint with {len(self.embeddings_cache)} embeddings")
        
        # Save final updated cache
        with open(cache_path, 'wb') as f:
            pickle.dump(self.embeddings_cache, f)
        
        print(f"Updated cache with {len(self.embeddings_cache)} embeddings")
    
    def _generate_emotion_prompt(self, emotion):
        """Create a detailed text prompt for CLIP based on emotion"""
        prompt = f"An artwork that expresses the emotion of {emotion}. "

        # Add subjects
        if emotion in self.emotion_to_subjects_styles:
            subjects = self.emotion_to_subjects_styles[emotion]["subjects"]
            if subjects:
                subject_str = ", ".join(subjects[:3])  # Take first 3 subjects
                prompt += f"It might contain {subject_str}. "

        # Add styles
        if emotion in self.emotion_to_subjects_styles:
            styles = self.emotion_to_subjects_styles[emotion]["styles"]
            if styles:
                style_str = ", ".join(styles[:2])  # Take first 2 styles
                prompt += f"In a style that is {style_str}. "

        # Add colors
        if emotion in self.emotion_to_colors:
            colors = self.emotion_to_colors[emotion]
            if colors:
                color_str = ", ".join(colors[:3]) # Take first 3 colors
                prompt += f"Using colors like {color_str}."

        return prompt
    
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
    
    def _create_faiss_index(self):
        """Create FAISS index from cached embeddings for fast similarity search"""
        if not FAISS_AVAILABLE:
            return
            
        # Get dimension from embeddings
        if not self.embeddings_cache:
            print("No embeddings available to create index")
            return
        
        # Get dimension from first valid embedding
        for path in self.embeddings_cache:
            embedding = self.embeddings_cache[path]
            if embedding is not None:
                self.embedding_dimensions = embedding.shape[1]
                break
        
        if self.embedding_dimensions is None:
            print("No valid embeddings found")
            return
            
        # Initialize index
        print(f"Creating FAISS index with dimension {self.embedding_dimensions}")
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dimensions)
        
        # Add embeddings to index
        all_embeddings = []
        self.all_ids = []
        
        for idx, row in tqdm(self.artwork_df.iterrows(), total=len(self.artwork_df), desc="Building FAISS index"):
            if row['image_path'] in self.embeddings_cache:
                embedding = self.embeddings_cache[row['image_path']]
                if embedding is not None:
                    self.all_ids.append(idx)
                    all_embeddings.append(embedding[0])
        
        if all_embeddings:
            all_embeddings = np.array(all_embeddings).astype('float32')
            self.faiss_index.add(all_embeddings)
            print(f"Created FAISS index with {len(all_embeddings)} vectors")
        else:
            print("No valid embeddings to add to index")
    
    def get_recommendations(self, user_emotion, top_n=5, enhance_with_colors=True):
        """
        Get artwork recommendations based on emotion
        
        Args:
            user_emotion: Emotion to search for
            top_n: Number of recommendations to return
            enhance_with_colors: Whether to add color information
            
        Returns:
            DataFrame with recommendations and similarity scores
        """
        # Check for dataset
        if self.artwork_df is None or len(self.artwork_df) == 0:
            print("No artwork data loaded. Call load_dataset() first.")
            return None, None
            
        # Validate the emotion
        all_emotions = [e for sublist in self.emotion_categories.values() for e in sublist]
        if user_emotion not in all_emotions:
            print(f"Invalid emotion. Valid emotions are: {all_emotions}")
            return None, None
        
        # Generate emotion prompt
        emotion_prompt = self._generate_emotion_prompt(user_emotion)
        print(f"Generated prompt: '{emotion_prompt}'")
        
        # Get text embedding for the emotion
        emotion_embedding = self._get_text_embedding(emotion_prompt)
        
        # Get recommendations based on similarity
        if FAISS_AVAILABLE and self.faiss_index is not None:
            # Use FAISS for fast similarity search
            recommendations, similarities = self._get_recommendations_faiss(emotion_embedding, top_n)
        else:
            # Fallback to standard similarity search
            recommendations, similarities = self._get_recommendations_standard(emotion_embedding, top_n)
        
        # Enhance with color analysis
        if enhance_with_colors and recommendations is not None:
            recommendations = self._enhance_with_colors(recommendations, similarities)
        
        # Add the emotion used for these recommendations
        if recommendations is not None:
            # Track which emotion was used for these recommendations
            recommendations._emotion_used = user_emotion
            
            # Also add it as a column
            recommendations['emotion_used'] = user_emotion
        
        return recommendations, similarities
    
    def _get_recommendations_faiss(self, emotion_embedding, top_n):
        """Get recommendations using FAISS similarity search"""
        emotion_embedding_np = emotion_embedding.cpu().numpy()
        
        # Search the index
        D, I = self.faiss_index.search(emotion_embedding_np, min(top_n, len(self.all_ids)))
        
        # Convert FAISS results to original dataframe indices
        if len(I) > 0 and len(I[0]) > 0:
            top_indices = [self.all_ids[i] for i in I[0]]
            similarities = [(idx, score) for idx, score in zip(top_indices, D[0])]
            
            return self.artwork_df.iloc[top_indices], similarities
        else:
            print("FAISS search returned no results")
            return None, None
    
    def _get_recommendations_standard(self, emotion_embedding, top_n):
        """Get recommendations using standard similarity calculation"""
        similarities = []
        
        # Process each artwork using cached embeddings
        for idx, row in tqdm(self.artwork_df.iterrows(), total=len(self.artwork_df), desc="Calculating similarities"):
            try:
                # Get image embedding from cache
                image_path = row['image_path']
                if image_path in self.embeddings_cache:
                    embedding = self.embeddings_cache[image_path]
                    if embedding is not None:
                        # Convert back to tensor if needed
                        image_embedding = torch.tensor(embedding).to(self.device)
                        
                        # Calculate similarity
                        similarity = F.cosine_similarity(emotion_embedding, image_embedding).item()
                        
                        # Store result
                        similarities.append((idx, similarity))
            except Exception as e:
                print(f"Error processing image {row['image_path']}: {e}")
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get top N results
        if similarities:
            top_indices = [idx for idx, _ in similarities[:top_n]]
            return self.artwork_df.iloc[top_indices], similarities[:top_n]
        else:
            print("No valid similarities found")
            return None, None
    
    def _enhance_with_colors(self, recommendations, similarities):
        """Add dominant color information to recommendations"""
        enhanced_recommendations = []
        for idx, row in recommendations.iterrows():
            try:
                img = Image.open(row['image_path']).convert("RGB")
                dominant_colors = extract_dominant_colors(img)
                named_colors = [rgb_to_named_color(color) for color in dominant_colors]
                
                # Add color information to the row
                row_dict = row.to_dict()
                row_dict['dominant_colors'] = named_colors
                
                enhanced_recommendations.append(row_dict)
            except Exception as e:
                print(f"Error processing colors for {row['image_path']}: {e}")
                # Still include the row without color information
                enhanced_recommendations.append(row.to_dict())
        
        return pd.DataFrame(enhanced_recommendations)


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
    
    # Convert to string if it's not already
    text = str(text)
    
    # Check if the text contains CJK characters
    has_cjk = False
    for char in text:
        try:
            if 'CJK' in unicodedata.name(char, ''):
                has_cjk = True
                break
        except ValueError:
            # Some characters might not have a name in unicodedata
            continue
    
    # If we're on Windows, we need to handle CJK differently
    if has_cjk and sys.platform == 'win32':
        # Windows console has issues with CJK chars in some configurations
        # We can try to ensure it's UTF-8, but console font matters too
        return text.encode('utf-8', errors='replace').decode('utf-8')
    
    return text


def display_recommendations(recommendations, similarities=None, display_images=True):
    """Display the recommended artworks with their metadata and similarity scores
    
    Args:
        recommendations: DataFrame of recommended artworks
        similarities: List of (index, score) tuples
        display_images: Whether to attempt to display images (set to False if images not available)
    """
    for i, row in enumerate(recommendations.iterrows() if isinstance(recommendations, pd.DataFrame) else recommendations):
        if isinstance(recommendations, pd.DataFrame):
            idx, row_data = row
        else:
            row_data = row
        
        print("\n" + "=" * 60)    
        print(f"🎨 Recommendation #{i+1}")
        print("=" * 60)
        
        # Basic artwork information - Use format_text_for_display for each text field
        print(f"📌 Title: {format_text_for_display(row_data.get('Title', 'Unknown'))}")
        
        if 'Artist Display Name' in row_data:
            print(f"👩‍🎨 Artist: {format_text_for_display(row_data.get('Artist Display Name', 'Unknown'))}")
        
        # New fields - Medium, Date, and Classification
        if 'Medium' in row_data:
            print(f"🖌️  Medium: {format_text_for_display(row_data.get('Medium', 'Unknown'))}")
            
        if 'Object Date' in row_data:
            print(f"📅 Created: {format_text_for_display(row_data.get('Object Date', 'Unknown date'))}")
            
        if 'Classification' in row_data:
            print(f"🏛️  Style: {format_text_for_display(row_data.get('Classification', 'Unknown'))}")
        
        # Color information
        if 'dominant_colors' in row_data:
            colors = [format_text_for_display(color) for color in row_data['dominant_colors']]
            print(f"🎨 Dominant Colors: {', '.join(colors)}")
            
        # Similarity score
        if similarities and i < len(similarities):
            similarity = similarities[i][1] * 100  # Convert to percentage
            print(f"✨ Match Score: {similarity:.1f}%")
        
        # Add artwork description if available
        if 'General_Text_Description' in row_data and isinstance(row_data['General_Text_Description'], str):
            desc = row_data['General_Text_Description']
            # Truncate long descriptions
            if len(desc) > 300:
                desc = desc[:297] + "..."
            print("\n📝 Description:")
            print(f"  {format_text_for_display(desc)}")
        
        # Add "Why this artwork?" section to explain the emotional connection
        print("\n💭 Why this artwork for your emotion?")
        emotion = None
        # Try to determine which emotion was used for this recommendation
        if hasattr(row_data, '_emotion_used') and row_data._emotion_used:
            emotion = row_data._emotion_used
        else:
            # Fall back to emotion column if it exists
            if 'emotion_used' in row_data:
                emotion = row_data['emotion_used']
            else:
                # Fall back to first emotion in the list if we don't know
                # This is a bit of a hack but should work for demo purposes
                emotion_categories = load_emotion_mappings()[0]
                all_emotions = [e for sublist in emotion_categories.values() for e in sublist]
                if all_emotions:
                    emotion = all_emotions[0]
        
        if emotion:
            # Get emotion mappings
            _, emotion_to_colors, emotion_to_subjects_styles = load_emotion_mappings()
            
            reasons = []
            
            # Check for style match
            if ('Classification' in row_data and 
                emotion in emotion_to_subjects_styles and 
                'styles' in emotion_to_subjects_styles[emotion]):
                
                artwork_style = row_data.get('Classification', '').lower()
                emotion_styles = [s.lower() for s in emotion_to_subjects_styles[emotion]['styles']]
                
                for style in emotion_styles:
                    if style.lower() in artwork_style:
                        reasons.append(f"The {style} style often evokes {emotion}")
                        break
            
            # Check for color match
            if 'dominant_colors' in row_data and emotion in emotion_to_colors:
                emotion_color_list = emotion_to_colors[emotion]
                artwork_colors = row_data['dominant_colors']
                
                matching_colors = [c for c in artwork_colors if c in emotion_color_list]
                if matching_colors:
                    reasons.append(f"The {matching_colors[0]} color is associated with {emotion}")
            
            # If we couldn't find specific reasons, give a generic one
            if not reasons:
                if emotion in emotion_to_subjects_styles:
                    subjects = emotion_to_subjects_styles[emotion].get('subjects', [])
                    if subjects:
                        reasons.append(f"The composition and themes reflect {emotion} through elements like {subjects[0]}")
                    else:
                        reasons.append(f"The visual qualities of this artwork match the emotion of {emotion}")
                else:
                    reasons.append(f"The visual qualities of this artwork match the emotion of {emotion}")
            
            # Print reasons
            for reason in reasons[:2]:  # Limit to 2 reasons
                print(f"  • {reason}")
            
            # Add a visual similarity reason
            print(f"  • The artwork's visual features closely match our {emotion} prompt")
        
        print("-" * 60)
        
        # Display the image only if requested and the file exists
        if display_images:
            try:
                # Check if file exists before trying to open it
                if not os.path.isfile(row_data['image_path']):
                    print(f"Image file not found: {row_data['image_path']}")
                    continue
                    
                img = Image.open(row_data['image_path'])
                
                # Create figure with proper font configuration for title
                plt.figure(figsize=(8, 8))
                plt.imshow(img)
                plt.axis('off')
                
                # Format title for display
                title_text = format_text_for_display(row_data.get('Title', 'Untitled'))
                
                # Set title with appropriate font
                plt.title(title_text, fontsize=14)
                plt.show()
                print("\n")
            except Exception as e:
                print(f"Error displaying image: {e}\n")


# Example usage
if __name__ == "__main__":
    # Example paths
    CSV_PATH = "/Users/ijunhyeong/Desktop/IPEX_Project/Image_Process/painting_info_with_descriptions.csv"
    IMAGE_FOLDER = "/Volumes/X31/ArtiTech_images"
    CACHE_DIR = "./emotion_art_cache"
    
    # Create a recommender (using smaller, faster model)
    recommender = ArtworkRecommender(
        model_name="openai/clip-vit-base-patch32",
        cache_dir=CACHE_DIR
    )
    
    # Load dataset
    artwork_df = recommender.load_dataset(CSV_PATH, IMAGE_FOLDER)
    
    # Example: Get recommendations for "Joy"
    recommendations, similarities = recommender.get_recommendations("Joy", top_n=3)
    
    # Display recommendations
    if recommendations is not None:
        display_recommendations(recommendations, similarities) 