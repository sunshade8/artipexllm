#!/usr/bin/env python3
"""
Script to generate art embedding cache for faster artwork recommendations.

This script processes the artwork images and creates CLIP embeddings,
saving them to a cache file for later use.
"""

import os
import sys
import time
import argparse
from pathlib import Path

# Add parent directory to path so we can import the module
sys.path.insert(0, str(Path(__file__).parent.parent))

from empathy_art_rag.art.artwork_recommender import ArtworkRecommender
from empathy_art_rag.config import ART_CSV_PATH, ART_IMAGE_FOLDER, ART_CACHE_DIR, DEFAULT_ART_MODEL

def main(args):
    """Generate art embedding cache."""
    # Print configuration
    print(f"Art CSV Path: {ART_CSV_PATH}")
    print(f"Art Image Folder: {ART_IMAGE_FOLDER}")
    print(f"Art Cache Directory: {ART_CACHE_DIR}")
    print(f"CLIP Model: {args.model_name}")
    
    # Check if the CSV file exists
    if not os.path.exists(ART_CSV_PATH):
        print(f"Error: Art CSV file not found at {ART_CSV_PATH}")
        return 1
    
    # Check if the image folder exists
    if not os.path.exists(ART_IMAGE_FOLDER):
        print(f"Error: Art image folder not found at {ART_IMAGE_FOLDER}")
        return 1
    
    # Create cache directory if it doesn't exist
    os.makedirs(ART_CACHE_DIR, exist_ok=True)
    
    # Initialize ArtworkRecommender
    print("\nInitializing ArtworkRecommender...")
    art_recommender = ArtworkRecommender(
        model_name=args.model_name,
        cache_dir=ART_CACHE_DIR
    )
    
    # Load dataset and rebuild cache
    print("\nLoading dataset and generating cache...")
    start_time = time.time()
    
    try:
        dataset = art_recommender.load_dataset(
            csv_path=ART_CSV_PATH,
            image_folder=ART_IMAGE_FOLDER,
            rebuild_cache=args.force_rebuild
        )
        
        # Print dataset info
        print(f"\nLoaded {len(dataset)} artwork items")
        
        # Get cache info - replacing the get_cache_stats() method that doesn't exist
        try:
            # Try to access the embeddings directly if available
            if hasattr(art_recommender, 'embeddings') and art_recommender.embeddings is not None:
                cache_size = len(art_recommender.embeddings)
                print(f"Cache size: {cache_size} embeddings")
            elif hasattr(art_recommender, 'clip_embeddings') and art_recommender.clip_embeddings is not None:
                cache_size = len(art_recommender.clip_embeddings)
                print(f"Cache size: {cache_size} embeddings")
            else:
                print("Cache created, but couldn't determine exact size")
        except Exception as e:
            print(f"Note: Unable to get exact cache stats: {e}")
        
        # Time taken
        elapsed_time = time.time() - start_time
        print(f"Process completed in {elapsed_time:.2f} seconds")
        
        # Test a sample recommendation
        if args.test:
            print("\nTesting artwork recommendation...")
            for emotion in ["Joy", "Serenity", "Nostalgia"]:
                print(f"\nGetting recommendations for '{emotion}'...")
                recommendations, _ = art_recommender.get_recommendations(
                    user_emotion=emotion,
                    top_n=1
                )
                if recommendations is not None and len(recommendations) > 0:
                    artwork = recommendations.iloc[0]
                    print(f"Recommended: {artwork.get('Title', 'Untitled')} by {artwork.get('Artist Display Name', 'Unknown')}")
                else:
                    print(f"No recommendations found for '{emotion}'")
        
        return 0
    
    except Exception as e:
        print(f"Error generating cache: {e}")
        return 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate art embedding cache for faster artwork recommendations"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_ART_MODEL,
        help=f"CLIP model name to use (default: {DEFAULT_ART_MODEL})"
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force rebuild of the cache even if it already exists"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test the cache with sample recommendations after generation"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("   ART EMBEDDING CACHE GENERATOR")
    print("=" * 80)
    print()
    
    exit_code = main(args)
    sys.exit(exit_code) 