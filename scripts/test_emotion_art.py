#!/usr/bin/env python3
"""
Test Script for Emotion Detection and Art Recommendation

This script tests the emotion detection and art recommendation components
directly to help diagnose and fix issues.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path so we can import the module
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import API key setup
try:
    from fix_api_key import setup_api_key
    api_key = setup_api_key()
except ImportError:
    print("Warning: fix_api_key module not found.")
    api_key = None

# Import components
from empathy_art_rag.art.artwork_recommender import ArtworkRecommender, display_recommendations
from empathy_art_rag.config import ART_CSV_PATH, ART_IMAGE_FOLDER, ART_CACHE_DIR, DEFAULT_ART_MODEL
from empathy_art_rag.handlers.emotion_handler import detect_emotion_keywords

def test_emotion_detection():
    """Test emotion detection with various test cases"""
    print("\n===== Testing Emotion Detection =====")
    
    test_cases = [
        "I feel so happy today!",
        "I'm feeling sad after what happened.",
        "This makes me angry!",
        "I'm having a really bad day.",
        "I fought with my boss and now I feel terrible.",
        "Can you recommend some art?",
        "I'm feeling nostalgic about my childhood.",
        "I'm so stressed about this project.",
        "I'm really nervous about my presentation tomorrow."
    ]
    
    for i, test in enumerate(test_cases, 1):
        emotions = detect_emotion_keywords(test)
        print(f"\nTest {i}: \"{test}\"")
        print(f"Detected emotions: {', '.join(emotions) if emotions else 'None'}")

def test_art_recommendation():
    """Test artwork recommendations for different emotions"""
    print("\n===== Testing Art Recommendation =====")
    
    print("Initializing ArtworkRecommender...")
    recommender = ArtworkRecommender(
        model_name=DEFAULT_ART_MODEL,
        cache_dir=ART_CACHE_DIR
    )
    
    print("Loading artwork dataset...")
    dataset = recommender.load_dataset(
        csv_path=ART_CSV_PATH,
        image_folder=ART_IMAGE_FOLDER
    )
    print(f"Loaded {len(dataset)} artwork items")
    
    # Get all valid emotions
    all_emotions = [e for sublist in recommender.emotion_categories.values() for e in sublist]
    print(f"\nValid emotions in recommender: {all_emotions}")
    
    # Map lowercase emotions to correctly capitalized emotions
    emotion_map = {e.lower(): e for e in all_emotions}
    print(f"\nEmotion case mapping: {emotion_map}")
    
    # Test emotions with correct capitalization
    test_emotions = [
        "Sadness", 
        "Joy", 
        "Anger", 
        "Fear", 
        "Nostalgia",
        "Love",
        "Serenity"
    ]
    
    for emotion in test_emotions:
        print(f"\nGetting recommendations for emotion: {emotion}")
        recommendations, similarities = recommender.get_recommendations(
            user_emotion=emotion,
            top_n=1
        )
        
        if recommendations is not None:
            for i, row in enumerate(recommendations.iterrows(), 1):
                idx, item = row
                print(f"\nRecommendation #{i} for {emotion}:")
                print(f"Title: {item.get('Title', 'Unknown')}")
                print(f"Artist: {item.get('Artist Display Name', 'Unknown')}")
                print(f"Period: {item.get('Object Date', 'Unknown')}")
                print(f"Department: {item.get('Department', 'Unknown')}")
                if 'dominant_colors' in item:
                    print(f"Colors: {', '.join(item['dominant_colors'])}")
        else:
            print(f"No recommendations found for {emotion}")

def main():
    """Main entry point"""
    print("==================================================")
    print("  EMOTION DETECTION AND ART RECOMMENDATION TESTER  ")
    print("==================================================")
    
    test_emotion_detection()
    test_art_recommendation()

if __name__ == "__main__":
    main() 