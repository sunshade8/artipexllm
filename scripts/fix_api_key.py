#!/usr/bin/env python3
"""
API Key Helper for EmpathyArtRAG

This script provides functions to properly set and retrieve the OpenAI API key.
It ensures the API key is correctly set in the environment variables.
"""

import os
from pathlib import Path

def fix_openai_api_key():
    """
    Set and return the OpenAI API key.
    
    This function tries to load the API key from the environment variables
    or prompts the user to enter it.
    
    Returns:
        str: The OpenAI API key
    """
    # Try to get the API key from environment variables
    api_key = os.environ.get("OPENAI_API_KEY", "")
    
    if not api_key:
        # If not available, try to load from .env file
        api_key = get_api_key_from_env_file("OPENAI_API_KEY")
        
        # If still not available, prompt the user
        if not api_key:
            print("\n⚠️ OpenAI API key not found in environment variables or .env file.")
            api_key = input("Please enter your OpenAI API key: ").strip()
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
                print(f"OpenAI API Key has been set: {api_key[:5]}...")
    else:
        print(f"OpenAI API Key found in environment: {api_key[:5]}...")
    
    return api_key

def fix_tavily_api_key():
    """
    Set and return the Tavily API key.
    
    This function sets the Tavily API key from the .env file or uses a default if needed.
    
    Returns:
        str: The Tavily API key if found, None otherwise
    """
    # Try to get the API key from the environment or .env file
    tavily_key = get_api_key_from_env_file("TAVILY_API_KEY")
    
    if tavily_key:
        # Set the API key in the environment
        os.environ["TAVILY_API_KEY"] = tavily_key
        print(f"Tavily API Key has been set: {tavily_key[:5]}...")
        return tavily_key
    else:
        print("⚠️ Tavily API key not found. Web search functionality will be limited.")
        return None

def get_api_key_from_env_file(key_name="OPENAI_API_KEY"):
    """
    Try to load an API key from the .env file.
    
    Args:
        key_name: The name of the environment variable to load (default: OPENAI_API_KEY)
    
    Returns:
        str: The API key if found, None otherwise
    """
    try:
        from dotenv import load_dotenv
        
        # Determine the path to the .env file (in the project root)
        project_root = Path(__file__).parent.parent
        env_path = project_root / '.env'
        
        if env_path.exists():
            # Load environment variables from .env
            load_dotenv(dotenv_path=env_path)
            print(f"Loaded environment from {env_path}")
            
            # Get the API key from environment
            api_key = os.environ.get(key_name)
            
            if key_name == "OPENAI_API_KEY":
                if api_key and api_key != "your-openai-api-key-here" and len(api_key) > 20:
                    print(f"Found valid {key_name} in .env: {api_key[:5]}...")
                    return api_key
                else:
                    print(f"{key_name} in .env file is invalid or missing")
                    return None
            else:
                # For other API keys, just check if they exist and have some content
                if api_key and len(api_key) > 5:
                    print(f"Found valid {key_name} in .env: {api_key[:5]}...")
                    return api_key
                else:
                    print(f"{key_name} in .env file is invalid or missing")
                    return None
        else:
            print(f".env file not found at {env_path}")
            return None
    except ImportError:
        print("dotenv package not installed. Cannot load from .env file.")
        return None

def setup_api_key():
    """
    Main function to set up the API keys.
    
    This function sets up both OpenAI and Tavily API keys.
    
    Returns:
        str: The OpenAI API key
    """
    # Try to set up the OpenAI API key
    openai_key = fix_openai_api_key()
    
    # Also set up the Tavily API key
    tavily_key = fix_tavily_api_key()
    
    # Return the OpenAI key for backward compatibility
    return openai_key

if __name__ == "__main__":
    # When run as a script, just set up the API keys
    api_key = setup_api_key()
    if api_key:
        print(f"API key setup successful. Key length: {len(api_key)}")
    else:
        print("Failed to set up API key. Please check your configuration.") 