#!/usr/bin/env python3
"""
Interactive Chat Interface for EmpathyArtRAG

This script provides a simple terminal-based chat interface to test the
complete EmpathyArtRAG system, including RAG functionality and emotion-based
art recommendations.
"""

import os
import sys
import time
import re
from pathlib import Path
import argparse
import textwrap
from typing import List, Dict, Any

# Add parent directory to path so we can import the module
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import our API key fix module first, before any other imports
# to ensure keys are set before environment checks
try:
    from fix_api_key import setup_api_key
    # Set up the API key
    api_key = setup_api_key()
except ImportError:
    print("Warning: fix_api_key module not found. Will try alternative methods for API key setup.")
    api_key = None

# Try to load .env file as a fallback if API key is still not set
if not api_key:
    try:
        from dotenv import load_dotenv
        # Load .env file from project root
        env_path = Path(__file__).parent.parent / '.env'
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
            print(f"Loaded environment variables from {env_path}")
            api_key = os.environ.get("OPENAI_API_KEY")
    except ImportError:
        print("python-dotenv not installed. You can install it with: pip install python-dotenv")
        print("Continuing without loading .env file...")

# Final check if API key is set
if not api_key:
    print("\n" + "=" * 80)
    print("⚠️  OpenAI API Key is not set! Chat functionality will not work.")
    print("You can set it in one of these ways:")
    print("  1. Create a .env file in the project root with: OPENAI_API_KEY=your_key_here")
    print("  2. Set the OPENAI_API_KEY environment variable")
    print("  3. Enter your API key below")
    print("=" * 80)
    user_api_key = input("\nEnter your OpenAI API key (or press Enter to skip): ").strip()
    if user_api_key:
        api_key = user_api_key
        os.environ["OPENAI_API_KEY"] = user_api_key
    else:
        print("No API key provided. Chat functionality will be limited.")

# Now that API keys are set, import components
from empathy_art_rag.art.artwork_recommender import ArtworkRecommender
from empathy_art_rag.config import ART_CSV_PATH, ART_IMAGE_FOLDER, ART_CACHE_DIR, DEFAULT_ART_MODEL
from empathy_art_rag.utils import print_color  # Import the print_color function from our utils module

# Import OpenAI for chat functionality
try:
    from openai import OpenAI
except ImportError:
    print("Error: OpenAI package not installed. Please install with: pip install openai")
    sys.exit(1)

# Import Tavily for web search
try:
    from tavily import TavilyClient
    tavily_api_key = os.environ.get("TAVILY_API_KEY")
    if not tavily_api_key:
        print("Warning: TAVILY_API_KEY environment variable is not set. Web search will be disabled.")
        has_tavily = False
    else:
        tavily_client = TavilyClient(api_key=tavily_api_key)
        has_tavily = True
        print_color("Tavily search client initialized successfully with API key", "green")
except ImportError:
    print("Warning: Tavily package not installed. Web search functionality will be limited.")
    print("You can install it with: pip install tavily-python")
    has_tavily = False
except Exception as e:
    print(f"Error initializing Tavily client: {e}")
    has_tavily = False

# Terminal colors for better readability
COLORS = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "white": "\033[37m",
}

# Using imported print_color function from utils module instead
# def print_color(text, color="reset", end="\n"):
#     """Print text with specified color."""
#     print(f"{COLORS.get(color, '')}{text}{COLORS['reset']}", end=end)

def print_header():
    """Print the chat interface header."""
    print("\n" + "=" * 80)
    print_color("   EMPATHY ART RAG INTERACTIVE CHAT", "cyan")
    print("=" * 80)
    print_color("\nType your message to chat with the AI assistant.", "yellow")
    print_color("Type 'exit', 'quit', or 'q' to end the conversation.", "yellow")
    print_color("Type 'clear' or 'c' to clear the conversation history.", "yellow")
    print("=" * 80 + "\n")

def format_artwork(artwork):
    """Format artwork information."""
    title = artwork.get('Title', 'Untitled')
    artist = artwork.get('Artist Display Name', 'Unknown')
    period = artwork.get('Object Date', 'Unknown')
    culture = artwork.get('Culture', 'Unknown')
    department = artwork.get('Department', 'Unknown')
    # Format text fields to avoid None values
    culture = culture if culture else "Unknown"
    department = department if department else "Unknown"
    
    return f"\"{title}\" by {artist} ({period})\nCulture: {culture}, Department: {department}"

def is_art_request(text):
    """Determine if the text is requesting art recommendations."""
    art_request_patterns = [
        r"(?i)recommend.*art",
        r"(?i)show me.*art",
        r"(?i)suggest.*artwork",
        r"(?i)find.*painting",
        r"(?i)art.*recommend",
        r"(?i)artwork.*help",
        r"(?i)healing.*art",
        r"(?i)therapeutic.*art",
        r"(?i)art\s+piece",
        r"(?i)show\s+art",
        r"(?i)painting",
        r"(?i)can you recommend",
        # Korean patterns
        r"작품.*추천",
        r"그림.*보여",
        r"예술.*추천",
        r"미술.*치유"
    ]
    
    return any(re.search(pattern, text) for pattern in art_request_patterns)

class ChatInterface:
    """Interactive chat interface for EmpathyArtRAG."""
    
    def __init__(self, model_name="gpt-3.5-turbo"):
        """Initialize the chat interface."""
        self.conversation_history = []
        self.model_name = model_name
        self.terminal_width = 80
        
        # State tracking for art recommendations
        self.pending_art_recommendation = False
        self.detected_emotion = None
        self.capitalized_emotion = None
        
        # Web search capability
        self.has_tavily = has_tavily
        
        # Check if API key is available
        self.has_valid_api_key = bool(api_key)
        
        # Initialize OpenAI client if API key is available
        if self.has_valid_api_key:
            try:
                self.client = OpenAI(api_key=api_key)
                print_color("OpenAI client initialized successfully", "green")
            except Exception as e:
                print_color(f"Error initializing OpenAI client: {e}", "red")
                self.has_valid_api_key = False
        
        # Initialize ArtworkRecommender
        print_color("\nInitializing ArtworkRecommender...", "blue")
        self.art_recommender = ArtworkRecommender(
            model_name=DEFAULT_ART_MODEL,
            cache_dir=ART_CACHE_DIR
        )
        
        # Load dataset with cached embeddings
        print_color("Loading artwork dataset with cached embeddings...", "blue")
        self.dataset = self.art_recommender.load_dataset(
            csv_path=ART_CSV_PATH,
            image_folder=ART_IMAGE_FOLDER
        )
        print_color(f"Loaded {len(self.dataset)} artwork items", "green")
        
        # System message that defines the assistant's behavior
        self.system_message = {
            "role": "system",
            "content": """You are an empathetic AI assistant that combines conversational
            capabilities with art recommendations. When a user expresses emotions or asks
            directly about art, you should respond empathetically to their situation.
            
            If they express emotions like happiness, sadness, anger, etc., but don't directly
            ask for art, you should ask if they'd like to see artwork that might resonate with
            their emotional state. For example: "Would you like me to recommend an artwork that
            might resonate with how you're feeling right now?"
            
            If they ask directly for art or confirm they want art recommendations, recommend the
            specific artwork that is provided to you and explain why it resonates with their
            emotional state. Be helpful, informative, and emotionally intelligent."""
        }
    
    def wrap_text(self, text, initial_indent="", subsequent_indent="  "):
        """Wrap text to fit terminal width with improved paragraph formatting."""
        # Split text into paragraphs
        paragraphs = text.split('\n')
        formatted_paragraphs = []
        
        for paragraph in paragraphs:
            # For empty paragraphs (just line breaks), add them as-is
            if not paragraph.strip():
                formatted_paragraphs.append('')
                continue
                
            # Lists detection
            is_list_item = False
            list_match = re.match(r'^(\s*[\d]+\.|\s*[\*\-•]) ', paragraph)
            if list_match:
                is_list_item = True
                list_indent = list_match.group(0)
                this_subsequent_indent = " " * len(list_indent)
            else:
                this_subsequent_indent = subsequent_indent
            
            # Wrap this paragraph
            if is_list_item:
                wrapped_lines = textwrap.wrap(
                    paragraph,
                    width=self.terminal_width - len(initial_indent),
                    initial_indent=initial_indent,
                    subsequent_indent=initial_indent + this_subsequent_indent
                )
            else:
                wrapped_lines = textwrap.wrap(
                    paragraph,
                    width=self.terminal_width - len(initial_indent),
                    initial_indent=initial_indent,
                    subsequent_indent=initial_indent + this_subsequent_indent
                )
            
            formatted_paragraphs.append("\n".join(wrapped_lines))
        
        # Join paragraphs with double newlines for better readability
        return "\n\n".join(formatted_paragraphs)
    
    def add_message(self, role, content):
        """Add a message to the conversation history."""
        self.conversation_history.append({"role": role, "content": content})
    
    def display_message(self, role, content):
        """Display a message with proper formatting and improved readability."""
        if role == "user":
            # Print a divider line before user message
            print("\n" + "-" * 80)
            print_color("You: ", "green", end="")
            print(content)
            
        elif role == "assistant":
            # Print a divider line before assistant's response
            print("\n" + "-" * 80)
            print_color("AI: ", "blue", end="")
            
            # Process the content to enhance formatting
            enhanced_content = self.enhance_text_formatting(content)
            
            # Display the formatted content
            print(self.wrap_text(enhanced_content, initial_indent="", subsequent_indent="    "))
            
        elif role == "system":
            print_color(content, "yellow")
            
        # Add a small spacing after every message
        print()
    
    def enhance_text_formatting(self, text):
        """Enhance text formatting for better readability."""
        # Improve list formatting
        text = re.sub(r'(\d+)\.\s+', r'\n\1. ', text)
        text = re.sub(r'(\n\d+\.)', r'\1 ', text)
        
        # Ensure proper spacing after periods
        text = re.sub(r'\.([A-Z])', r'. \1', text)
        
        # Better formatting for sections or points
        text = re.sub(r'(Step \d+:)', r'\n\1', text)
        text = re.sub(r'(Tips?:)', r'\n\1', text)
        text = re.sub(r'(Note:)', r'\n\1', text)
        
        # Handle art descriptions better
        text = re.sub(r'(".*?")\s+by\s+(.*?)\s+\((.*?)\)', r'\n"\1"\nby \2\n(\3)', text)
        
        return text
    
    def get_llm_response(self, messages):
        """Get a response from the LLM."""
        if not self.has_valid_api_key:
            return "I can't process your request because there's no valid OpenAI API key configured. Please set your API key and restart the application."
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            print_color(f"Error getting response from LLM: {e}", "red")
            if "API key" in str(e).lower():
                self.has_valid_api_key = False
                return "I apologize, but there seems to be an issue with the OpenAI API key. Please check your API key and restart the application."
            return "I apologize, but I'm having trouble processing your request right now."
    
    def display_artwork(self, artwork):
        """Display information about the artwork and show the image if possible."""
        try:
            image_path = artwork.get('image_path')
            if image_path and os.path.exists(image_path):
                # Print a decorative box around artwork information
                print_color("\n" + "╔" + "═" * 78 + "╗", "cyan")
                print_color("║" + " " * 24 + "ARTWORK INFORMATION" + " " * 24 + "║", "cyan")
                print_color("╠" + "═" * 78 + "╣", "cyan")
                
                # Title with special formatting
                title = artwork.get('Title', 'Untitled')
                artist = artwork.get('Artist Display Name', 'Unknown')
                period = artwork.get('Object Date', 'Unknown')
                
                print_color("║  " + "Title:".ljust(12) + " " + title.ljust(62) + "║", "cyan")
                print_color("║  " + "Artist:".ljust(12) + " " + artist.ljust(62) + "║", "cyan")
                print_color("║  " + "Period:".ljust(12) + " " + period.ljust(62) + "║", "cyan")
                
                # More details
                culture = artwork.get('Culture', 'Unknown')
                department = artwork.get('Department', 'Unknown')
                culture = culture if culture else "Unknown"
                department = department if department else "Unknown"
                
                print_color("║  " + "Culture:".ljust(12) + " " + culture.ljust(62) + "║", "cyan")
                print_color("║  " + "Department:".ljust(12) + " " + department.ljust(62) + "║", "cyan")
                
                # Colors if available
                if 'dominant_colors' in artwork:
                    colors = ', '.join(artwork['dominant_colors'])
                    # Handle long color lists by wrapping
                    if len(colors) > 62:
                        print_color("║  " + "Colors:".ljust(12) + " " + colors[:62].ljust(62) + "║", "cyan")
                        print_color("║  " + " ".ljust(12) + " " + colors[62:124].ljust(62) + "║", "cyan")
                    else:
                        print_color("║  " + "Colors:".ljust(12) + " " + colors.ljust(62) + "║", "cyan")
                
                # Image path
                if len(image_path) > 62:
                    # Truncate with ellipsis if too long
                    display_path = "..." + image_path[-59:]
                    print_color("║  " + "Image:".ljust(12) + " " + display_path.ljust(62) + "║", "cyan")
                else:
                    print_color("║  " + "Image:".ljust(12) + " " + image_path.ljust(62) + "║", "cyan")
                
                print_color("╚" + "═" * 78 + "╝", "cyan")
                
                # Try to open the image file with the default image viewer
                try:
                    import subprocess
                    import platform
                    
                    system = platform.system()
                    if system == 'Darwin':  # macOS
                        subprocess.run(['open', image_path], check=True)
                        print_color("✓ Image opened in default viewer.", "green")
                    elif system == 'Windows':
                        os.startfile(image_path)
                        print_color("✓ Image opened in default viewer.", "green")
                    elif system == 'Linux':
                        subprocess.run(['xdg-open', image_path], check=True)
                        print_color("✓ Image opened in default viewer.", "green")
                    else:
                        print_color(f"⚠ Unable to open image: Unsupported system ({system})", "yellow")
                except Exception as e:
                    print_color(f"⚠ Failed to open image: {e}", "red")
                    print_color("You can manually view the image at the path shown above.", "yellow")
            else:
                print_color("⚠ Image file not found or path not specified.", "red")
        except Exception as e:
            print_color(f"⚠ Error displaying artwork: {e}", "red")

    def needs_web_search(self, query):
        """
        Determine if a query likely needs real-time information from the web.
        
        Args:
            query: The user's query
            
        Returns:
            bool: True if web search is recommended
        """
        # Keywords that suggest factual or current information is needed
        factual_patterns = [
            r"(?i)what is",
            r"(?i)who is",
            r"(?i)where is",
            r"(?i)when is",
            r"(?i)when was",
            r"(?i)how (many|much|old|long)",
            r"(?i)latest",
            r"(?i)recent",
            r"(?i)current",
            r"(?i)newest",
            r"(?i)today",
            r"(?i)yesterday",
            r"(?i)this (week|month|year)",
            r"(?i)last (week|month|year)",
            r"(?i)2023",
            r"(?i)2024",
            r"(?i)news",
            r"(?i)update",
            r"(?i)president",
            r"(?i)election",
            r"(?i)exhibition",
            r"(?i)event",
            r"(?i)show me",
            r"(?i)information about",
            r"(?i)tell me about",
            r"(?i)did (.*) win",
            r"(?i)Nobel Prize",
            r"(?i)age of",
            r"(?i)born in",
            r"(?i)died in"
        ]
        
        # Topics that likely need real-time information
        factual_topics = [
            "president", "prime minister", "election", "olympics", "awards", 
            "movie", "exhibition", "show", "release", "price", "stock", 
            "latest", "newest", "coronavirus", "covid", "pandemic", 
            "climate", "war", "conflict", "political", "government", 
            "economy", "scientific", "research", "discovery", "technology"
        ]
        
        # Check if any factual topic is in the query
        for topic in factual_topics:
            if topic.lower() in query.lower():
                print_color(f"[DEBUG] Query contains factual topic: {topic}", "magenta")
                return True
        
        # Check if the query matches any factual pattern
        for pattern in factual_patterns:
            if re.search(pattern, query):
                print_color(f"[DEBUG] Query matches factual pattern: {pattern}", "magenta")
                return True
        
        # Special cases for questions that don't fit patterns but need real-time info
        words = query.lower().split()
        if len(words) >= 3 and words[0] in ["what", "who", "where", "when", "why", "how"]:
            print_color("[DEBUG] Query appears to be a specific question that might need real-time info", "magenta")
            return True
            
        return False

    def perform_web_search(self, query):
        """
        Perform a web search using Tavily API to get real-time information.
        
        Args:
            query: The search query
            
        Returns:
            str: Search results summary or error message
        """
        if not self.has_tavily:
            print_color("[DEBUG] Tavily search capability is not available", "red")
            return "Web search is not available. The Tavily API key may not be set correctly."
        
        try:
            print_color(f"[DEBUG] Performing web search for: {query}", "magenta")
            
            # For better results, focus the query by prefixing with "recent information about"
            # only if the query doesn't already have a clear question structure
            words = query.lower().split()
            if len(words) >= 3 and words[0] not in ["what", "who", "where", "when", "why", "how"]:
                search_query = "recent information about " + query
            else:
                search_query = query
                
            # Perform the search with advanced depth for comprehensive results
            search_result = tavily_client.search(
                query=search_query, 
                search_depth="advanced",
                include_domains=["wikipedia.org", "nytimes.com", "bbc.com", "reuters.com", "theguardian.com", 
                                "washingtonpost.com", "cnn.com", "apnews.com", "npr.org"],
                max_results=5
            )
            
            if not search_result or not search_result.get("results"):
                print_color("[DEBUG] No search results found from Tavily", "yellow")
                return "No relevant information found through web search."
            
            # Extract and format the search results
            content = "Here's what I found from searching the web:\n\n"
            
            # Include the search query and timestamp
            from datetime import datetime
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            content += f"Search query: \"{query}\"\n"
            content += f"Search performed at: {current_time}\n\n"
            
            # Include more detailed results (up to 5)
            for i, result in enumerate(search_result.get("results", [])[:5], 1):
                title = result.get("title", "Untitled")
                snippet = result.get("content", "No content available")
                url = result.get("url", "No URL available")
                
                content += f"{i}. {title}\n"
                # Include more of the content (up to 300 chars)
                content += f"   {snippet[:300]}...\n"
                content += f"   Source: {url}\n\n"
            
            print_color(f"[DEBUG] Found {len(search_result.get('results', []))} search results from Tavily", "green")
            return content
        except Exception as e:
            print_color(f"[DEBUG] Error performing web search: {e}", "red")
            return f"Sorry, I encountered an error when trying to search the web: {str(e)}"

    def process_message(self, user_message):
        """Process a user message and generate a response."""
        # Add user message to history
        self.add_message("user", user_message)
        
        # Import emotion handlers here to avoid circular imports
        from empathy_art_rag.handlers.emotion_handler import detect_emotion_keywords
        
        # Check for art confirmation in the user's message
        art_confirmation_patterns = [
            r"(?i)\byes\b",
            r"(?i)\bsure\b",
            r"(?i)\bplease\b",
            r"(?i)\bok\b",
            r"(?i)\byeah\b",
            r"(?i)show me",
            r"(?i)recommend",
            r"(?i)i would",
            r"(?i)i'd like",
            # Korean patterns
            r"네",
            r"예",
            r"좋아요",
            r"보여줘"
        ]
        
        is_art_confirmation = False
        if self.pending_art_recommendation:
            is_art_confirmation = any(re.search(pattern, user_message) for pattern in art_confirmation_patterns)
            
            # Check for denial patterns
            art_denial_patterns = [
                r"(?i)\bno\b",
                r"(?i)\bnot\b",
                r"(?i)\bdon'?t\b",
                r"(?i)later",
                # Korean patterns
                r"아니",
                r"됐어",
                r"괜찮아"
            ]
            is_art_denial = any(re.search(pattern, user_message) for pattern in art_denial_patterns)
            
            if is_art_denial:
                # Clear the pending recommendation
                self.pending_art_recommendation = False
                self.detected_emotion = None
                self.capitalized_emotion = None
                print_color("[DEBUG] User declined art recommendation.", "magenta")
        
        # Check if this is a direct art request
        direct_art_request = is_art_request(user_message)
        
        # Get the full message history for the LLM
        messages_for_llm = [self.system_message] + self.conversation_history
        
        # Check if web search would be helpful for non-art related queries
        should_search_web = False
        web_search_results = None
        
        # Only consider web search if we're not in the middle of an art recommendation flow
        if not (self.pending_art_recommendation and (is_art_confirmation or is_art_denial)) and not direct_art_request:
            # Check if the query might need web search
            should_search_web = self.needs_web_search(user_message)
            
            if should_search_web:
                if self.has_tavily:
                    print_color("[DEBUG] Query likely needs real-time information. Performing web search.", "magenta")
                    web_search_results = self.perform_web_search(user_message)
                    
                    if web_search_results:
                        # Add web search results to the context
                        search_context_message = {
                            "role": "system",
                            "content": f"""The user's question requires up-to-date information.
                            Here are relevant web search results to help answer their query:
                            
                            {web_search_results}
                            
                            IMPORTANT: Use this information to provide an accurate, up-to-date response.
                            When using information from the search results, cite the sources in your response.
                            If the search results don't directly answer the question, acknowledge the limitations
                            of the available information.
                            
                            Do NOT mention that you're using web search in your response - simply
                            incorporate the information naturally as if you knew it."""
                        }
                        
                        # Insert web search results before other messages
                        messages_for_llm = [self.system_message] + [search_context_message] + self.conversation_history
                        print_color("[DEBUG] Added web search results to context.", "magenta")
                else:
                    print_color("[DEBUG] Web search would be helpful but Tavily is not available.", "yellow")
        
        # Process different scenarios for art recommendations
        if self.pending_art_recommendation and (is_art_confirmation or is_art_denial):
            # User has confirmed they want art - use the stored emotion
            print_color("[DEBUG] User confirmed art recommendation.", "magenta")
            self.pending_art_recommendation = False
            
            # Get art recommendation using stored emotion
            try:
                recommendations, prompt = self.art_recommender.get_recommendations(
                    user_emotion=self.capitalized_emotion,
                    top_n=1
                )
                
                if recommendations is not None and len(recommendations) > 0:
                    artwork = recommendations.iloc[0]
                    artwork_info = format_artwork(artwork)
                    
                    # Add art info to the system message for this turn only
                    art_system_message = {
                        "role": "system",
                        "content": f"""The user's emotion is: {self.detected_emotion}.
                        Recommend this specific artwork that resonates with this emotion: {artwork_info}
                        
                        Explain why this artwork might resonate with their emotional state,
                        linking the artwork's characteristics to the emotion of {self.detected_emotion}.
                        
                        IMPORTANT: Only suggest this exact artwork from our collection. Do not 
                        make up or suggest other artworks like "The Scream" or "Starry Night" 
                        unless they are the specific artwork provided in this system message.
                        
                        Mention that you're showing the artwork image if possible, or 
                        that they can find the image at the displayed path.
                        
                        Keep your response conversational and empathetic."""
                    }
                    
                    # Insert art system message before the user's message
                    messages_for_llm = [self.system_message] + [art_system_message] + self.conversation_history
                    
                    print_color(f"[DEBUG] Artwork selected: {artwork.get('Title', 'Unknown')}", "magenta")
                    
                    # Display artwork info and image
                    # Store artwork to display after getting AI response
                    self.current_artwork = artwork
                else:
                    print_color(f"[WARNING] No recommendations found for emotion: {self.capitalized_emotion}", "red")
            except Exception as e:
                print_color(f"[DEBUG] Error getting art recommendations: {e}", "red")
        
        elif direct_art_request:
            # Direct request for art - detect emotions or use default
            detected_emotions = detect_emotion_keywords(user_message)
            
            # If no emotions detected for art request, default to "calm"
            if not detected_emotions:
                detected_emotions = ["calm"]
                print_color("[DEBUG] No emotion detected in art request, using default: calm", "magenta")
            
            # Map to properly capitalized emotion
            emotion_map = self.get_emotion_map()
            primary_emotion = detected_emotions[0]
            capitalized_emotion = emotion_map.get(primary_emotion)
            
            if not capitalized_emotion:
                print_color(f"[WARNING] Could not map emotion '{primary_emotion}' to a valid artwork emotion. Using 'Serenity' as default.", "yellow")
                capitalized_emotion = "Serenity"
            
            print_color(f"[DEBUG] Getting art recommendations for: {primary_emotion} (mapped to {capitalized_emotion})", "magenta")
            
            # Get art recommendations
            try:
                recommendations, prompt = self.art_recommender.get_recommendations(
                    user_emotion=capitalized_emotion,
                    top_n=1
                )
                
                if recommendations is not None and len(recommendations) > 0:
                    artwork = recommendations.iloc[0]
                    artwork_info = format_artwork(artwork)
                    
                    # Add art info to the system message for this turn only
                    art_system_message = {
                        "role": "system",
                        "content": f"""The user has directly requested art recommendations related to the emotion: {primary_emotion}.
                        Recommend this specific artwork that resonates with this emotion: {artwork_info}
                        
                        Explain why this artwork might resonate with their emotional state,
                        linking the artwork's characteristics to the emotion of {primary_emotion}.
                        
                        IMPORTANT: Only suggest this exact artwork from our collection. Do not 
                        make up or suggest other artworks like "The Scream" or "Starry Night" 
                        unless they are the specific artwork provided in this system message.
                        
                        Mention that you're showing the artwork image if possible, or 
                        that they can find the image at the displayed path.
                        
                        Keep your response conversational and empathetic."""
                    }
                    
                    # Insert art system message before the user's message
                    messages_for_llm = [self.system_message] + [art_system_message] + self.conversation_history
                    
                    print_color(f"[DEBUG] Artwork selected: {artwork.get('Title', 'Unknown')}", "magenta")
                    
                    # Store artwork to display after getting AI response
                    self.current_artwork = artwork
                else:
                    print_color(f"[WARNING] No recommendations found for emotion: {capitalized_emotion}", "red")
            except Exception as e:
                print_color(f"[DEBUG] Error getting art recommendations: {e}", "red")
        
        else:
            # Neither confirmation nor direct request - check for emotions
            detected_emotions = detect_emotion_keywords(user_message)
            
            # Check if there's a real emotional expression, not just a greeting
            is_real_emotion_expression = False
            if detected_emotions:
                # Patterns that indicate genuine emotional expression rather than just greeting
                emotion_expression_patterns = [
                    r"(?i)i\s+(?:am|feel|am\s+feeling)\s+",
                    r"(?i)i\'m\s+(?:feeling|feeling\s+so|so)\s+",
                    r"(?i)(?:very|really|quite|extremely|somewhat)\s+",
                    r"(?i)(?:having|had)\s+a\s+(?:good|bad|great|terrible|awful|wonderful)\s+",
                    r"(?i)it\s+makes\s+me\s+",
                    r"(?i)i\s+(?:hate|love|despise|enjoy|dislike)\s+",
                    r"(?i)feeling\s+",
                    # Korean patterns for emotional expressions
                    r"(?:기분|느낌)(?:이|은|는)\s+",
                    r"(?:화|슬픔|기쁨|행복)(?:이|을|를|에)\s+",
                    r"(?:나는|내가)\s+(?:기쁘|슬프|화가|행복하)"
                ]
                # Check if any emotion expression pattern is found
                is_real_emotion_expression = any(re.search(pattern, user_message) for pattern in emotion_expression_patterns)
                
                # Special case for short messages that directly state an emotion
                if not is_real_emotion_expression and len(user_message.split()) <= 5:
                    # If it's a short message and entirely about emotion, consider it an expression
                    primary_emotion = detected_emotions[0]
                    emotion_variants = [
                        f"(?i)\\b{re.escape(variant)}\\b" 
                        for variant in self.get_emotion_variants(primary_emotion)
                    ]
                    
                    # If the message is heavily centered around the emotion term, consider it an expression
                    for pattern in emotion_variants:
                        if re.search(pattern, user_message):
                            word_count = len(user_message.split())
                            if word_count <= 3:  # Very short message like "I'm sad"
                                is_real_emotion_expression = True
                                break
            
            # If emotions detected and it seems like a real emotion expression (not just greeting),
            # prepare to ask for confirmation
            if detected_emotions and is_real_emotion_expression and not self.pending_art_recommendation:
                primary_emotion = detected_emotions[0]
                
                # Map to properly capitalized emotion
                emotion_map = self.get_emotion_map()
                capitalized_emotion = emotion_map.get(primary_emotion)
                
                if not capitalized_emotion:
                    print_color(f"[WARNING] Could not map emotion '{primary_emotion}' to a valid artwork emotion. Using 'Serenity' as default.", "yellow")
                    capitalized_emotion = "Serenity"
                
                # Store the emotions for later use
                self.detected_emotion = primary_emotion
                self.capitalized_emotion = capitalized_emotion
                self.pending_art_recommendation = True
                
                print_color(f"[DEBUG] Detected emotion: {primary_emotion}. Setting up to ask for art confirmation.", "magenta")
                
                # Add system message to ask if user wants art recommendations
                ask_confirmation_message = {
                    "role": "system",
                    "content": f"""The user seems to be expressing the emotion: {primary_emotion}.
                    
                    Respond empathetically to their message, and then ask if they would like you to 
                    recommend an artwork that might resonate with how they're feeling right now.
                    
                    Keep your response conversational and empathetic. DO NOT recommend specific
                    artwork yet - wait for them to confirm they want art recommendations."""
                }
                
                # Insert confirmation message before the user's message
                messages_for_llm = [self.system_message] + [ask_confirmation_message] + self.conversation_history
                
                print_color(f"[DEBUG] Will ask for confirmation to recommend art for emotion: {primary_emotion}", "magenta")
            else:
                if detected_emotions:
                    print_color(f"[DEBUG] Emotion detected ({', '.join(detected_emotions)}), but doesn't seem like a direct emotional expression. Not asking for art recommendation.", "magenta")
            
            # If there's still a pending recommendation but user didn't confirm or deny,
            # continue the conversation normally but keep the pending state
        
        # Get response from LLM
        ai_response = self.get_llm_response(messages_for_llm)
        
        # Add assistant response to history
        self.add_message("assistant", ai_response)
        
        # Display the assistant's response
        self.display_message("assistant", ai_response)
        
        # If we recommended artwork, display it
        if hasattr(self, 'current_artwork'):
            self.display_artwork(self.current_artwork)
            delattr(self, 'current_artwork')  # Clean up after displaying
    
    def get_emotion_variants(self, emotion):
        """Get variants of an emotion for pattern matching."""
        emotion_variants = {
            "joy": ["joy", "joyful", "joyous", "happy", "happier", "happiest", "happiness"],
            "happiness": ["happiness", "happy", "happier", "happiest", "glad", "delighted", "pleased"],
            "love": ["love", "loved", "loving", "lovely", "adore", "cherish"],
            "sadness": ["sadness", "sad", "unhappy", "depressed", "down", "blue", "gloomy", "somber", "sorrow", "sorrowful"],
            "anger": ["anger", "angry", "furious", "mad", "outraged", "upset", "annoyed", "irritated", "frustrated"],
            "fear": ["fear", "afraid", "scared", "terrified", "frightened", "fearful"],
            "anxiety": ["anxiety", "anxious", "nervous", "worried", "stressed", "stress", "uneasy", "tense"],
            "calm": ["calm", "peaceful", "relaxed", "serene", "tranquil"],
            "nostalgia": ["nostalgia", "nostalgic", "reminiscent", "remembrance", "memories", "longing"]
        }
        
        # Return variants for the specific emotion, or just the emotion itself if no variants defined
        return emotion_variants.get(emotion, [emotion])
    
    def get_emotion_map(self):
        """Get the mapping of lowercase emotions to correctly capitalized ones for the art recommender."""
        return {
            "joy": "Joy",
            "happiness": "Joy",  # Map happiness to Joy for the recommender
            "love": "Love",
            "serenity": "Serenity",
            "calm": "Serenity",  # Map calm to Serenity
            "peaceful": "Serenity",
            "amusement": "Amusement",
            "gratitude": "Gratitude",
            "hope": "Hope",
            "admiration": "Admiration",
            "sadness": "Sadness",
            "grief": "Sadness",  # Map grief to Sadness
            "anger": "Anger",
            "fear": "Fear",
            "anxiety": "Anxiety",
            "disgust": "Disgust",
            "dread": "Dread",
            "confusion": "Confusion",
            "boredom": "Boredom",
            "loneliness": "Sadness",  # Map loneliness to Sadness
            "dreamy": "Dreamy",
            "curious": "Dreamy",  # Map curious to Dreamy
            "mystical": "Mystical",
            "spiritual": "Spiritual",
            "whimsical": "Whimsical",
            "creative": "Whimsical",  # Map creative to Whimsical
            "nostalgia": "Nostalgia",
            "contemplative": "Contemplation",
            "wonder": "Wonder",
            "awe": "Awe",
            "connectedness": "Connectedness"
        }
    
    def clear_history(self):
        """Clear the conversation history."""
        self.conversation_history = []
        print_color("Conversation history cleared.", "yellow")
    
    def chat_loop(self):
        """Run the chat loop."""
        print_header()
        
        # Check if Tavily web search is available
        if not self.has_tavily and os.environ.get("TAVILY_API_KEY"):
            print_color("\n⚠️  Warning: Tavily API key is set but web search initialization failed.", "yellow")
            print_color("Web search functionality will be limited. Check your API key or internet connection.", "yellow")
        elif not self.has_tavily:
            print_color("\n⚠️  Notice: Web search functionality is not available (Tavily API key not set).", "yellow")
            print_color("For full functionality, add TAVILY_API_KEY to your .env file.", "yellow")
        else:
            print_color("\n✓ Web search functionality is available using Tavily.", "green")
            print_color("You can ask factual questions that require current information.", "green")
        
        # Warn if OpenAI API key is missing
        if not self.has_valid_api_key:
            print_color("\n⚠️  Warning: OpenAI API key is not set or is invalid.", "red")
            print_color("The art recommendation features will work, but chat responses will be limited.", "red")
            print_color("To enable full functionality, restart the application and provide a valid API key.\n", "red")
        
        # Welcome message
        welcome_message = {
            "role": "assistant",
            "content": "Welcome to the EmpathyArtRAG chat! I'm here to have a conversation with you and recommend artwork that resonates with your emotions. Feel free to share how you're feeling or ask about art directly. You can also ask me factual questions that require current information."
        }
        self.display_message(welcome_message["role"], welcome_message["content"])
        
        while True:
            # Get user input with a nicer prompt
            try:
                print("\n" + "-" * 80)
                user_input = input(f"{COLORS['green']}You: {COLORS['reset']}")
            except (KeyboardInterrupt, EOFError):
                print("\nExiting chat...")
                break
            
            # Process commands
            if user_input.lower() in ["exit", "quit", "q"]:
                print_color("\nEnding conversation. Goodbye!", "yellow")
                break
            elif user_input.lower() in ["clear", "c"]:
                self.clear_history()
                continue
            elif not user_input.strip():
                continue
            
            # Process the user message
            self.process_message(user_input)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Interactive chat interface for EmpathyArtRAG"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-3.5-turbo",
        help="OpenAI model to use (default: gpt-3.5-turbo)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="OpenAI API key (overrides environment variable)"
    )
    
    args = parser.parse_args()
    
    # Set API key from command line if provided
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key
        global api_key
        api_key = args.api_key
    
    try:
        # Initialize and run the chat interface
        chat = ChatInterface(model_name=args.model)
        chat.chat_loop()
    except KeyboardInterrupt:
        print("\nExiting chat...")
    except Exception as e:
        print_color(f"Error: {e}", "red")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 