"""
Utility functions for the EmpathyArtRAG system.

This module contains various utility functions used throughout the system.
"""

import sys
from typing import Optional

def print_color(text: str, color: Optional[str] = None, end: str = "\n") -> None:
    """
    Print colored text to the terminal.
    
    Args:
        text: The text to print
        color: The color to use (blue, green, yellow, red, purple, cyan)
        end: The string to append at the end (default newline)
    """
    colors = {
        'blue': '\033[94m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'red': '\033[91m',
        'purple': '\033[95m',
        'cyan': '\033[96m',
        'reset': '\033[0m'
    }
    
    if color in colors and sys.stdout.isatty():
        print(f"{colors[color]}{text}{colors['reset']}", end=end)
    else:
        print(text, end=end)

def print_colored(text: str, color: Optional[str] = None, end: str = "\n") -> None:
    """
    Alias for print_color for compatibility with demo.py.
    
    Args:
        text: The text to print
        color: The color to use
        end: The string to append at the end (default newline)
    """
    print_color(text, color, end) 