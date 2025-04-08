"""
EmpathyArtRAG - Emotion-Aware Conversational AI with Artwork Recommendations

This package integrates a conversational AI system with emotion detection
and artwork recommendations based on detected emotions.
"""

__version__ = "0.1.0"
__author__ = "IPEX Project Team"

from .main import process_query
from .workflow import create_workflow_graph
from .models import GraphState
from .config import check_environment

# Check environment 
env_warnings = check_environment()
if env_warnings:
    print("\n".join(env_warnings))
    print("Please complete environment setup. See README.md for details.")

def create_empathy_art_rag():
    """Create the main EmpathyArtRAG system."""
    return create_workflow_graph() 