"""
Data models for the EmpathyArtRAG system.

This file defines the types and data structures used throughout the system.
"""
from typing import Dict, List, Any, Optional, Union, Callable
from typing_extensions import TypedDict  # Use typing_extensions instead of typing for TypedDict
from pydantic import BaseModel, Field
from langchain_core.documents import Document

# Extended graph state definition
class GraphState(TypedDict):
    '''Enhanced graph state with emotion tracking and art recommendation'''
    
    question: str  # User question
    documents: List[Document]  # Retrieved documents
    generation: str  # Generated answer
    conversation_history: List[Dict]  # Conversation history
    detected_emotion: Optional[str]  # Detected emotion if any
    art_recommendation: Optional[Dict]  # Artwork recommendation if any
    offer_art: bool  # Whether to offer art recommendation 

# Type alias for the graph state
GraphStateAlias = Dict[str, Any]

# Node function type
NodeFunction = Callable[[GraphStateAlias], GraphStateAlias]

class ArtRecommendation(TypedDict):
    """Art recommendation information."""
    emotion: str
    artwork: Dict[str, Any]
    formatted_text: str
    similarity: Optional[float]

class WorkflowNode(BaseModel):
    """A node in the workflow graph."""
    id: str
    func: NodeFunction = Field(exclude=True)
    next_nodes: List[str] = Field(default_factory=list)
    
    class Config:
        arbitrary_types_allowed = True

class WorkflowGraph(BaseModel):
    """The workflow graph defining the system's processing pipeline."""
    nodes: Dict[str, WorkflowNode]
    start_node: str
    
    class Config:
        arbitrary_types_allowed = True

class Message(BaseModel):
    """A message in the conversation history."""
    role: str
    content: str

class Conversation(BaseModel):
    """Conversation history and metadata."""
    messages: List[Message] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class UserQuery(BaseModel):
    """User query and related information."""
    query: str
    conversation_id: Optional[str] = None
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class Response(BaseModel):
    """System response to a user query."""
    answer: str
    conversation_id: str
    artwork: Optional[ArtRecommendation] = None
    metadata: Dict[str, Any] = Field(default_factory=dict) 