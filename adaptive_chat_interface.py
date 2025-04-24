#!/usr/bin/env python3
"""
Interactive Chat Interface for the Adaptive RAG System
"""

import os
import sys
import time
import re
from pathlib import Path
import argparse
import textwrap
import uuid  # For generating unique thread IDs
from typing import List, Dict, Any

# Add parent directory to path if necessary (adjust as needed)
# sys.path.insert(0, str(Path(__file__).parent.parent))

# Add src directory to Python path
SRC_DIR = Path(__file__).resolve().parent / 'src'
if SRC_DIR.is_dir():
    sys.path.insert(0, str(SRC_DIR))
    print(f"Added {SRC_DIR} to sys.path")
else:
    print(f"Warning: src directory not found at {SRC_DIR}")

# --- API Key Setup (Adapted from chat_interface.py) ---
try:
    # Assuming fix_api_key handles necessary keys (OpenAI, potentially others)
    from fix_api_key import setup_api_key
    api_keys_set = setup_api_key() # We might need specific keys later
except ImportError:
    print("Warning: fix_api_key module not found. Trying .env...")
    api_keys_set = False

if not api_keys_set:
    try:
        from dotenv import load_dotenv
        env_path = Path(__file__).parent.parent / '.env' # Adjust if .env is elsewhere
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
            print(f"Loaded environment variables from {env_path}")
            # Check for essential keys (adjust if needed)
            if os.environ.get("OPENAI_API_KEY"):
                 api_keys_set = True
        else:
             print(f".env file not found at {env_path}")
    except ImportError:
        print("python-dotenv not installed. Cannot load .env file.")

# Check/Prompt for OpenAI key specifically if still not set
openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    print("\n" + "=" * 80)
    print("⚠️  OpenAI API Key is not set! Chat functionality will likely fail.")
    print("You can set it in one of these ways:")
    print("  1. Create a .env file in the project root with: OPENAI_API_KEY=your_key_here")
    print("  2. Set the OPENAI_API_KEY environment variable")
    print("  3. Enter your API key below (less secure)")
    print("=" * 80)
    user_api_key = input("\nEnter your OpenAI API key (or press Enter to skip): ").strip()
    if user_api_key:
        openai_api_key = user_api_key
        os.environ["OPENAI_API_KEY"] = user_api_key
        api_keys_set = True
    else:
        print("No OpenAI API key provided. Functionality may be limited.")
# --- End API Key Setup ---

# --- Import the Adaptive RAG Chain from the src module ---
# !!! IMPORTANT: Verify 'adaptive_rag' is the correct object name in src/chain.py !!!
try:
    from chain import adaptive_rag
    # If the object has a different name in chain.py (e.g., 'app'), use:
    # from chain import app as adaptive_rag
    print("Successfully imported adaptive_rag from src.chain")
except ImportError as e:
    print(f"Error importing adaptive_rag from src.chain: {e}")
    print("Please ensure 'adaptive_rag' (or the correct object name) is defined in src/chain.py.")
    print("Also check that all dependencies within src modules are installed.")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during import from src.chain: {e}")
    sys.exit(1)

# Import LangChain components
try:
    from langchain_core.runnables import RunnableConfig
except ImportError:
    print("Error: langchain_core not installed. Please install with: pip install langchain-core")
    sys.exit(1)

# --- Terminal Colors and Helpers (Adapted from chat_interface.py) ---
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

def print_color(text, color="reset", end="\n"):
    """Print text with specified color."""
    print(f"{COLORS.get(color, '')}{text}{COLORS['reset']}", end=end)

def print_header():
    """Print the chat interface header."""
    print("\n" + "=" * 80)
    print_color("   ADAPTIVE RAG CHAT INTERFACE", "cyan")
    print("=" * 80)
    print_color("\nType your message to chat with the AI assistant.", "yellow")
    print_color("Type 'exit', 'quit', or 'q' to end the conversation.", "yellow")
    print_color("Type 'clear' to reset the conversation state.", "yellow")
    print("=" * 80 + "\n")

def wrap_text(text, initial_indent="", subsequent_indent="  ", width=80):
    """Wrap text to fit terminal width."""
    # Basic wrapping, can be enhanced later if needed
    wrapper = textwrap.TextWrapper(
        width=width,
        initial_indent=initial_indent,
        subsequent_indent=subsequent_indent,
        replace_whitespace=False,
        drop_whitespace=False
    )
    # Handle potential None or non-string inputs
    if not isinstance(text, str):
        text = str(text)
    return "\n".join(wrapper.wrap(line) for line in text.splitlines() if line)

def display_artwork(artwork_details):
    """Display artwork information and attempt to open the image."""
    if not artwork_details or not isinstance(artwork_details, dict):
        print_color("No artwork details to display.", "yellow")
        return

    print_color("\n" + "╔" + "═" * 78 + "╗", "cyan")
    print_color("║" + " " * 24 + "ARTWORK INFORMATION" + " " * 24 + "║", "cyan")
    print_color("╠" + "═" * 78 + "╣", "cyan")

    # --- Adapt these keys based on your adaptive_rag output ---
    title = artwork_details.get('Title', 'N/A')
    artist = artwork_details.get('Artist Display Name', 'N/A')
    period = artwork_details.get('Object Date', 'N/A')
    image_path = artwork_details.get('image_path', None)
    # Add more fields as needed (Culture, Department, etc.)
    culture = artwork_details.get('Culture', 'N/A')
    department = artwork_details.get('Department', 'N/A')
    # ---

    # Helper to format lines
    def format_line(label, value):
        label_str = f"{label}:".ljust(12)
        value_str = str(value).ljust(62)
        # Truncate long values
        if len(value_str) > 62:
            value_str = value_str[:59] + "..."
        return f"║  {label_str} {value_str} ║"

    print_color(format_line("Title", title), "cyan")
    print_color(format_line("Artist", artist), "cyan")
    print_color(format_line("Period", period), "cyan")
    print_color(format_line("Culture", culture), "cyan")
    print_color(format_line("Department", department), "cyan")
    if image_path:
        print_color(format_line("Image Path", image_path), "cyan")
    else:
        print_color(format_line("Image Path", "Not available"), "yellow")


    print_color("╚" + "═" * 78 + "╝", "cyan")

    # Attempt to open image
    if image_path and os.path.exists(image_path):
        try:
            import subprocess
            import platform

            system = platform.system()
            if system == 'Darwin':  # macOS
                subprocess.run(['open', image_path], check=True, capture_output=True)
                print_color("✓ Attempted to open image.", "green")
            elif system == 'Windows':
                os.startfile(image_path)
                print_color("✓ Attempted to open image.", "green")
            elif system == 'Linux':
                subprocess.run(['xdg-open', image_path], check=True, capture_output=True)
                print_color("✓ Attempted to open image.", "green")
            else:
                print_color(f"⚠ Unable to auto-open image on this system ({system}).", "yellow")
                print_color(f"  Please view manually at: {image_path}", "yellow")
        except FileNotFoundError:
             print_color(f"⚠ Command 'open'/'startfile'/'xdg-open' not found. Cannot open image automatically.", "yellow")
             print_color(f"  Please view manually at: {image_path}", "yellow")
        except subprocess.CalledProcessError as e:
            print_color(f"⚠ Error opening image: {e}", "red")
            print_color(f"  Please view manually at: {image_path}", "yellow")
        except Exception as e:
            print_color(f"⚠ Failed to open image: {e}", "red")
            print_color(f"  Please view manually at: {image_path}", "yellow")
    elif image_path:
        print_color(f"⚠ Image file not found at the specified path: {image_path}", "red")
    else:
        print_color("⚠ No image path provided for this artwork.", "yellow")


# --- ChatInterface Class ---
class ChatInterface:
    """Interactive chat interface using the Adaptive RAG system."""

    def __init__(self):
        """Initialize the chat interface."""
        self.thread_id = str(uuid.uuid4()) # Unique ID for this conversation
        print_color(f"Initializing new conversation (Thread ID: {self.thread_id})", "magenta")
        self.config = RunnableConfig(
            recursion_limit=25, # Increased recursion limit slightly
            configurable={"thread_id": self.thread_id}
        )
        self.terminal_width = 80 # Or use shutil.get_terminal_size().columns
        self.current_artwork = None # To hold artwork details for display

        # Check if adaptive_rag was imported successfully
        if 'adaptive_rag' not in globals():
             print_color("Error: adaptive_rag object not found. Exiting.", "red")
             sys.exit(1)
        self.rag_chain = adaptive_rag

        # Check for necessary API keys based on graph needs (example)
        if not openai_api_key:
             print_color("Warning: OpenAI API key missing, RAG may fail.", "yellow")


    def process_message(self, user_message: str):
        """Process user message using the adaptive_rag chain."""
        inputs = {"question": user_message}
        full_response_text = ""
        self.current_artwork = None # Reset artwork for this turn

        try:
            print_color("...", "blue", end=" ") # Indicate processing
            start_time = time.time()
            # --- Invoke the RAG Chain ---
            result = self.rag_chain.invoke(inputs, self.config)
            # ---
            duration = time.time() - start_time
            print_color(f"({duration:.2f}s)", "magenta") # Show duration


            # --- Extract Response Components (Adapt keys based on your chain's output) ---
            # These keys are based on the notebook examples provided
            generation = result.get("generation", "") # Main text response
            suggestion = result.get("suggestion_message", "") # Potential art offer
            artwork_details = result.get("artwork_details", None) # Artwork data if recommended
            # ---

            # Combine text parts for display
            if generation:
                full_response_text += generation
            if suggestion:
                # Add spacing if both parts exist
                if full_response_text:
                    full_response_text += "\n\n"
                full_response_text += suggestion

            if not full_response_text and not artwork_details:
                 full_response_text = "Sorry, I didn't get a specific response. Could you try rephrasing?"
                 print_color("[Warning] RAG chain returned empty response.", "yellow")

            # Store artwork for display after text
            if artwork_details:
                self.current_artwork = artwork_details

        except Exception as e:
            print_color(f"\nError invoking RAG chain: {e}", "red")
            # Provide a fallback response
            full_response_text = "I apologize, but I encountered an error processing your request."
            # Consider logging the full traceback here for debugging
            # import traceback
            # traceback.print_exc()


        # Display the response
        self.display_response(full_response_text)

        # Display artwork if available for this turn
        if self.current_artwork:
            display_artwork(self.current_artwork)
            self.current_artwork = None # Clear after display


    def display_response(self, text: str):
        """Display the AI's response."""
        print("\n" + "-" * self.terminal_width)
        print_color("AI:", "blue")
        # Use wrap_text for better formatting
        print(wrap_text(text.strip(), initial_indent="  ", subsequent_indent="  ", width=self.terminal_width - 2))
        print("-" * self.terminal_width)


    def clear_history(self):
        """Reset the conversation state by starting a new thread."""
        old_thread_id = self.thread_id
        self.thread_id = str(uuid.uuid4())
        self.config = RunnableConfig(
             recursion_limit=25,
             configurable={"thread_id": self.thread_id}
        )
        print_color(f"Conversation history cleared. Started new thread ({self.thread_id}).", "yellow")
        print_color(f"(Previous thread was: {old_thread_id})", "magenta")


    def chat_loop(self):
        """Run the main chat loop."""
        print_header()

        # Initial welcome / instructions
        welcome_message = "Hello! I'm your empathetic art assistant powered by Adaptive RAG. How are you feeling today, or what can I help you with?"
        self.display_response(welcome_message) # Display initial message

        while True:
            # Get user input
            try:
                print("\n" + "=" * self.terminal_width)
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
                # Display a fresh prompt after clearing
                welcome_clarified = "New conversation started. How can I help you?"
                self.display_response(welcome_clarified)
                continue
            elif not user_input.strip():
                continue # Ignore empty input

            # Process the user message using the RAG chain
            self.process_message(user_input)


# --- Main Execution ---
def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Interactive chat interface for the Adaptive RAG System"
    )
    # Add any command-line arguments if needed (e.g., specific model variant)
    # parser.add_argument("--model-variant", type=str, help="Specify a model variant if applicable")

    args = parser.parse_args()

    # You could potentially pass args to ChatInterface if needed
    # chat = ChatInterface(args=args)
    try:
        chat = ChatInterface()
        chat.chat_loop()
    except KeyboardInterrupt:
        print("\nExiting chat...")
    except Exception as e:
        print_color(f"An unexpected error occurred: {e}", "red")
        # Consider logging traceback here
        # import traceback
        # traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main()) 