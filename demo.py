#!/usr/bin/env python3
"""
Demo script for testing the EmpathyArtRAG system.

This script demonstrates the system's emotion detection and artwork recommendation
capabilities with various example queries.
"""

import os
import sys
from empathy_art_rag import process_query
from empathy_art_rag.config import ART_CSV_PATH, ART_IMAGE_FOLDER

def print_colored(text, color=None):
    """Print colored text in terminal."""
    colors = {
        'blue': '\033[94m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'red': '\033[91m',
        'purple': '\033[95m',
        'cyan': '\033[96m',
        'end': '\033[0m'
    }
    if color in colors and sys.stdout.isatty():
        print(f"{colors[color]}{text}{colors['end']}")
    else:
        print(text)

def check_environment():
    """Check if the environment is properly configured for the demo."""
    issues = []
    
    if not os.path.exists(ART_CSV_PATH):
        issues.append(f"Art metadata CSV file not found at {ART_CSV_PATH}")
    
    if not ART_IMAGE_FOLDER or not os.path.exists(ART_IMAGE_FOLDER):
        issues.append(f"Art image folder not found at {ART_IMAGE_FOLDER}")
    
    return issues

def run_example_conversation():
    """Run an example conversation demonstrating various system capabilities."""
    print_colored("=" * 80, "blue")
    print_colored("    EmpathyArtRAG SYSTEM DEMO", "blue")
    print_colored("=" * 80, "blue")
    print()
    
    # Check environment
    issues = check_environment()
    if issues:
        print_colored("WARNING: Environment issues detected:", "yellow")
        for issue in issues:
            print_colored(f"  - {issue}", "yellow")
        print()
        response = input("Continue with the demo anyway? (y/n): ").lower()
        if response != 'y':
            print("Exiting demo.")
            return
    
    print("This demo will run through an example conversation to demonstrate the system.")
    print("It shows emotion detection, adaptive retrieval, and artwork recommendations.")
    print()
    
    # Example conversation
    examples = [
        "Hello, how are you today?",
        "I'm feeling a bit sad today.",
        "Tell me about how art can help with emotions.",
        "Can you recommend art that might help me feel calm?",
        "I'm curious about art that represents joy.",
        "What's the relationship between art and emotional wellbeing?",
        "I'm feeling anxious about my upcoming presentation.",
        "Thanks for your help!"
    ]
    
    # Run conversation
    conversation_id = None
    
    for i, example in enumerate(examples):
        print_colored(f"\n=== Example {i+1}/{len(examples)} ===", "cyan")
        print_colored(f"User: {example}", "green")
        
        # Process query
        response = process_query(example, conversation_id)
        conversation_id = response.conversation_id
        
        # Print response
        print_colored(f"Assistant: {response.answer}", "purple")
        
        # Print metadata
        print_colored("\nMetadata:", "blue")
        print_colored(f"  Detected emotion: {response.metadata.get('detected_emotion')}", "blue")
        print_colored(f"  Retrieval needed: {response.metadata.get('retrieval_needed')}", "blue")
        print_colored(f"  Offered art: {response.metadata.get('offer_art')}", "blue")
        print_colored(f"  Execution path: {response.metadata.get('execution_path')}", "blue")
        
        if i < len(examples) - 1:
            input("\nPress Enter to continue to the next example...\n")
    
    print_colored("\n" + "=" * 80, "blue")
    print_colored("    DEMO COMPLETED", "blue")
    print_colored("=" * 80, "blue")

def run_interactive_demo():
    """Run an interactive demo where the user can input their own queries."""
    print_colored("=" * 80, "blue")
    print_colored("    EmpathyArtRAG INTERACTIVE DEMO", "blue")
    print_colored("=" * 80, "blue")
    print()
    
    # Check environment
    issues = check_environment()
    if issues:
        print_colored("WARNING: Environment issues detected:", "yellow")
        for issue in issues:
            print_colored(f"  - {issue}", "yellow")
        print()
    
    print("Enter your messages below. Type 'exit', 'quit', or 'bye' to end the demo.")
    print()
    
    conversation_id = None
    
    while True:
        user_input = input("> ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            break
        
        # Process query
        response = process_query(user_input, conversation_id)
        conversation_id = response.conversation_id
        
        # Print response
        print_colored(f"\n{response.answer}\n", "purple")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run a demo of the EmpathyArtRAG system")
    parser.add_argument(
        "--interactive", "-i", 
        action="store_true",
        help="Run in interactive mode (user inputs their own queries)"
    )
    args = parser.parse_args()
    
    if args.interactive:
        run_interactive_demo()
    else:
        run_example_conversation() 