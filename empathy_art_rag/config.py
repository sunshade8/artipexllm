"""
Configuration settings for the EmpathyArtRAG system.

This file contains paths, model settings, and other configuration.
"""

import os
from pathlib import Path

# Project base paths
PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")

# Artwork recommendation configuration
ART_CACHE_DIR = os.path.join(CACHE_DIR, "art_embeddings")
ART_CSV_PATH = os.environ.get(
    "ART_CSV_PATH", 
    os.path.join(DATA_DIR, "art", "painting_info_with_descriptions.csv")
)
ART_IMAGE_FOLDER = os.environ.get(
    "ART_IMAGE_FOLDER",
    os.path.join(DATA_DIR, "images")
)
DEFAULT_ART_MODEL = "openai/clip-vit-base-patch32"  # Smaller model for speed

# RAG Configuration 
VECTOR_DB_PATH = os.path.join(DATA_DIR, "vector_db")
DOCUMENT_PATH = os.path.join(DATA_DIR, "documents")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

# LLM Configuration
DEFAULT_MODEL = "gpt-3.5-turbo" 
LLM_TEMPERATURE = 0.2

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FILE = os.path.join(PROJECT_ROOT, "logs", "empathy_art_rag.log")

# Create necessary directories
for directory in [DATA_DIR, CACHE_DIR, ART_CACHE_DIR, 
                 VECTOR_DB_PATH, DOCUMENT_PATH,
                 os.path.join(PROJECT_ROOT, "logs")]:
    os.makedirs(directory, exist_ok=True)

# Package root directory auto-detection
PACKAGE_ROOT = Path(__file__).parent.parent.absolute()

# Data and model directory settings (from environment variables or defaults)
DATA_DIR = os.environ.get("EMPATHY_ART_RAG_DATA_DIR", os.path.join(PACKAGE_ROOT, "data"))
MODEL_DIR = os.environ.get("EMPATHY_ART_RAG_MODEL_DIR", os.path.join(PACKAGE_ROOT, "models"))

# Vector store path
VECTOR_STORE_PATH = os.environ.get("EMPATHY_ART_RAG_VECTOR_STORE_PATH", 
                                 os.path.join(MODEL_DIR, "vector_store"))

# Art cache directory
ART_CACHE_DIR = os.environ.get("EMPATHY_ART_RAG_ART_CACHE_DIR",
                             os.path.join(MODEL_DIR, "art_cache"))

# Data subdirectories
SAMPLE_DATA_DIR = os.path.join(DATA_DIR, "sample")
PDF_DIR = os.path.join(DATA_DIR, "pdf")
ART_DATA_DIR = os.path.join(DATA_DIR, "art")

# API key settings
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")

# Other settings
DEFAULT_TEMPERATURE = float(os.environ.get("EMPATHY_ART_RAG_TEMPERATURE", "0.0"))

def check_environment():
    """Check environment setup and warn about potential issues."""
    warnings = []
    
    if not OPENAI_API_KEY:
        warnings.append("⚠️ OPENAI_API_KEY environment variable is not set.")
    
    if not TAVILY_API_KEY:
        warnings.append("⚠️ TAVILY_API_KEY environment variable is not set.")
    
    if not os.path.exists(SAMPLE_DATA_DIR):
        warnings.append(f"⚠️ Sample data directory does not exist: {SAMPLE_DATA_DIR}")
    
    if not os.path.exists(VECTOR_STORE_PATH):
        warnings.append(f"⚠️ Vector store not found: {VECTOR_STORE_PATH}")
        warnings.append("   Run scripts/create_vector_store.py to create it.")
    
    if not os.path.exists(ART_CSV_PATH):
        warnings.append(f"⚠️ Art metadata CSV not found: {ART_CSV_PATH}")
        warnings.append("   Copy painting_info_with_descriptions.csv to data/art/")
    
    # Check if the art images folder exists
    if not ART_IMAGE_FOLDER or not os.path.exists(ART_IMAGE_FOLDER):
        warnings.append(f"⚠️ Art image folder not found: {ART_IMAGE_FOLDER}")
        warnings.append("   Update ART_IMAGE_FOLDER in config.py or set the ART_IMAGE_FOLDER environment variable.")
    
    # Check for art cache files
    cache_files = [f for f in os.listdir(ART_CACHE_DIR) if f.endswith('.pkl')] if os.path.exists(ART_CACHE_DIR) else []
    if not cache_files:
        warnings.append(f"⚠️ No art embedding cache files found in: {ART_CACHE_DIR}")
        warnings.append("   Run scripts/generate_art_cache.py to create them.")
    
    return warnings 