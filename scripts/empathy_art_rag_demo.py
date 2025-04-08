#!/usr/bin/env python3
"""
Demo script for the EmpathyArtRAG system.

This script demonstrates how to use the integrated system that combines
RAG capabilities with emotion-based artwork recommendations.
"""

import os
import sys
import time
from pathlib import Path
import argparse

# Add parent directory to path so we can import the module
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the art recommender component
from empathy_art_rag.art.artwork_recommender import ArtworkRecommender
from empathy_art_rag.config import ART_CSV_PATH, ART_IMAGE_FOLDER, ART_CACHE_DIR, DEFAULT_ART_MODEL
from empathy_art_rag.handlers.emotion_handler import detect_emotion, should_offer_art

# Sample conversation history
SAMPLE_CONVERSATIONS = [
    # Simple emotion expression
    [
        {"role": "user", "content": "I'm feeling really happy today! Everything is going well."}
    ],
    # Direct request for art
    [
        {"role": "user", "content": "Can you recommend some artwork that represents love?"}
    ],
    # Conversation with emotional context
    [
        {"role": "user", "content": "I've been thinking about my childhood a lot lately."},
        {"role": "assistant", "content": "Childhood memories can be quite powerful. Is there something specific about your childhood that's been on your mind?"},
        {"role": "user", "content": "Just feeling nostalgic about simpler times."}
    ],
    # Non-emotional question
    [
        {"role": "user", "content": "What's the capital of France?"}
    ],
    # Mixed emotional and informational
    [
        {"role": "user", "content": "I'm feeling anxious about my upcoming trip to Paris. Can you tell me about some peaceful places to visit there?"}
    ]
]

def format_message(message):
    """Format a message for display."""
    role = message["role"].upper()
    content = message["content"]
    return f"{role}: {content}"

def display_conversation(conversation):
    """Display a conversation history."""
    print("\nConversation:")
    print("-" * 40)
    for message in conversation:
        print(format_message(message))
    print("-" * 40)

def process_with_emotion_art(conversation, art_recommender):
    """Process a conversation with emotion detection and art recommendation."""
    # Get the latest user message
    latest_message = conversation[-1]["content"]
    
    # Detect emotions in the message
    emotions, emotion_debug = detect_emotion(latest_message, conversation)
    
    # Determine if we should offer art
    offer_art = should_offer_art(latest_message, conversation, emotions)
    
    print("\nEmotion Analysis:")
    print(f"Detected Emotions: {', '.join(emotions) if emotions else 'None'}")
    print(f"Should Offer Art: {offer_art}")
    
    # Get art recommendations if needed
    if offer_art and emotions:
        primary_emotion = emotions[0]
        print(f"\nGetting art recommendations for emotion: {primary_emotion}")
        
        start_time = time.time()
        recommendations, prompt = art_recommender.get_recommendations(
            user_emotion=primary_emotion,
            top_n=1
        )
        elapsed = time.time() - start_time
        
        print(f"Generated prompt: '{prompt}'")
        print(f"Found recommendations in {elapsed:.2f} seconds")
        
        if recommendations is not None and len(recommendations) > 0:
            artwork = recommendations.iloc[0]
            print("\nRecommended Artwork:")
            print(f"Title: {artwork.get('Title', 'Untitled')}")
            print(f"Artist: {artwork.get('Artist Display Name', 'Unknown')}")
            print(f"Period: {artwork.get('Object Date', 'Unknown')}")
            
            # Simulate assistant response with art recommendation
            art_response = (
                f"Based on the emotions in your message, you might appreciate this artwork: "
                f"\"{artwork.get('Title', 'Untitled')}\" by {artwork.get('Artist Display Name', 'Unknown')} "
                f"({artwork.get('Object Date', 'Unknown')}). "
                f"This piece captures the essence of {primary_emotion}."
            )
            
            print("\nGenerated Response:")
            print(art_response)
        else:
            print("No artwork recommendations found for this emotion.")
    else:
        print("\nNo art recommendations offered for this conversation.")
        
        # Simulate a regular response without art
        regular_response = "I would process this through the RAG system to generate a relevant response based on retrieved information."
        print("\nGenerated Response (simulated):")
        print(regular_response)

def main(args):
    """Run the EmpathyArtRAG demo."""
    # Print configuration
    print(f"Art CSV Path: {ART_CSV_PATH}")
    print(f"Art Image Folder: {ART_IMAGE_FOLDER}")
    print(f"Art Cache Directory: {ART_CACHE_DIR}")
    
    # Initialize ArtworkRecommender
    print("\nInitializing ArtworkRecommender...")
    art_recommender = ArtworkRecommender(
        model_name=DEFAULT_ART_MODEL,
        cache_dir=ART_CACHE_DIR
    )
    
    # Load dataset with cached embeddings
    print("Loading dataset with cached embeddings...")
    dataset = art_recommender.load_dataset(
        csv_path=ART_CSV_PATH,
        image_folder=ART_IMAGE_FOLDER
    )
    print(f"Loaded {len(dataset)} artwork items")
    
    # Process each sample conversation
    for i, conversation in enumerate(SAMPLE_CONVERSATIONS):
        print(f"\n\n{'='*80}")
        print(f"Sample {i+1}/{len(SAMPLE_CONVERSATIONS)}")
        print(f"{'='*80}")
        
        display_conversation(conversation)
        process_with_emotion_art(conversation, art_recommender)
        
        if i < len(SAMPLE_CONVERSATIONS) - 1:
            input("\nPress Enter to continue to the next sample...")
    
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Demo for the EmpathyArtRAG system"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("   EMPATHY ART RAG DEMO")
    print("=" * 80)
    print()
    
    exit_code = main(args)
    sys.exit(exit_code) 