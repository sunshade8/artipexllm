#!/usr/bin/env python3
"""
API Key Helper for EmpathyArtRAG

This script provides functions to properly set and retrieve API keys from environment variables,
primarily loaded from a .env file.
It ensures the API keys are correctly set in the environment variables.
"""

import os
from pathlib import Path

def fix_tavily_api_key():
    """
    Try to load and set the Tavily API key from the environment.
    
    Returns:
        str: The Tavily API key if found and set, None otherwise.
    """
    api_key = get_api_key_from_env_file(key_name="TAVILY_API_KEY")
    
    if not api_key:
        print("⚠️ Tavily API key not found in environment/.env. Web search functionality will be limited.")
        return None
    
    # Set the API key in the environment (redundant if load_dotenv worked, but safe)
    os.environ["TAVILY_API_KEY"] = api_key
    print(f"Tavily API Key loaded and set: {api_key[:5]}...") # Show fewer chars
    return api_key
    
def get_api_key_from_env_file(key_name="OPENAI_API_KEY"):
    """
    Try to load an API key from the .env file located in the project root.
    Loads the .env file and returns the value of the specified key.
    
    Args:
        key_name (str): The name of the environment variable to load (e.g., "OPENAI_API_KEY").
    
    Returns:
        str | None: The API key if found and valid, None otherwise.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        print("Error: python-dotenv package not installed. Cannot load keys from .env file.")
        print("Please install it: pip install python-dotenv")
        return None
        
    # Determine the path to the .env file (assumes it's in the project root)
    # Assumes this script (fix_api_key.py) is in scripts/ directory
    project_root = Path(__file__).resolve().parent.parent 
    env_path = project_root / '.env'
        
    if env_path.exists():
        # Load environment variables from .env into os.environ
        loaded = load_dotenv(dotenv_path=env_path, override=True) # Override ensures .env takes precedence
        if loaded:
            print(f"Loaded environment variables from: {env_path}")
        else:
             print(f"Warning: dotenv found .env file but failed to load variables from {env_path}")
             # Continue, maybe keys are set directly in environment
    else:
        print(f"Warning: .env file not found at {env_path}. Attempting to use existing environment variables.")

    # Get the API key from the environment (either pre-existing or loaded from .env)
    api_key = os.environ.get(key_name)
    
    # Basic validation
    if not api_key:
        print(f"❌ Error: API key '{key_name}' not found in environment variables or .env file.")
        return None
        
    if key_name == "OPENAI_API_KEY":
        if "sk-" not in api_key or len(api_key) < 50:
             print(f"❌ Error: OpenAI API Key ('{key_name}') loaded from environment/.env appears invalid.")
             return None
        else:
            print(f"✓ Found valid {key_name} in environment/.env: {api_key[:5]}...{api_key[-4:]}")
            return api_key
    elif key_name == "TAVILY_API_KEY":
        if "tvly-" not in api_key or len(api_key) < 10:
            print(f"❌ Error: Tavily API Key ('{key_name}') loaded from environment/.env appears invalid.")
            return None
        else:
            print(f"✓ Found valid {key_name} in environment/.env: {api_key[:5]}...")
            return api_key
    else:
        # Generic check for other keys
        if len(api_key) < 10: # Arbitrary minimum length
             print(f"❌ Error: API Key ('{key_name}') loaded from environment/.env appears invalid (too short).")
             return None
        else:
            print(f"✓ Found {key_name} in environment/.env.")
            return api_key

def setup_api_keys(): # Renamed for clarity
    """
    Main function to load and set up the required API keys (OpenAI, Tavily)
    from the environment/.env file.
    
    Raises:
        ValueError: If a required API key (OpenAI) is missing or invalid.

    Returns:
        tuple: (openai_api_key, tavily_api_key) - Tavily key can be None.
    """
    print("--- Setting up API Keys --- ")
    # Load OpenAI API Key (Required)
    openai_key = get_api_key_from_env_file(key_name="OPENAI_API_KEY")
    if not openai_key:
        # Raise an error if the essential OpenAI key is missing
        raise ValueError("CRITICAL ERROR: OpenAI API Key (OPENAI_API_KEY) is missing or invalid. \n" \
                         "Please create a .env file in the project root with your key \n" \
                         "or set the environment variable directly.")
    # Ensure it's set in the environment (might be redundant but safe)
    os.environ["OPENAI_API_KEY"] = openai_key
    
    # Load Tavily API Key (Optional, functionality degrades if missing)
    tavily_key = fix_tavily_api_key() # This already tries to load and set
    
    print("--- API Key Setup Complete --- ")
    # Return the keys (Tavily might be None)
    return openai_key, tavily_key

if __name__ == "__main__":
    # When run as a script, attempt setup and report status
    try:
        openai_key, tavily_key = setup_api_keys()
        print("\nAPI Key Status:")
        print(f"- OpenAI Key: {'Loaded' if openai_key else 'MISSING/INVALID'}")
        print(f"- Tavily Key: {'Loaded' if tavily_key else 'MISSING/INVALID'}")
    except ValueError as e:
         print(f"\nAPI Key Setup Failed: {e}")
    except Exception as e:
         print(f"\nAn unexpected error occurred during API key setup: {e}") 