"""
Main entry point for the EmpathyArtRAG system.

This file provides the main interface for interacting with the system.
"""

import uuid
import json
from typing import Dict, List, Any, Optional

from .models import UserQuery, Response, GraphState, Conversation
from .workflow import create_workflow_graph, execute_workflow

# Create workflow graph once
_workflow_graph = create_workflow_graph()

# Store conversations in memory (in production, use a database)
_conversations: Dict[str, Conversation] = {}

def _get_or_create_conversation(conversation_id: Optional[str] = None) -> tuple[str, Conversation]:
    """
    Get an existing conversation or create a new one.
    
    Args:
        conversation_id: ID of the conversation to retrieve
        
    Returns:
        Tuple of (conversation_id, conversation)
    """
    if conversation_id and conversation_id in _conversations:
        return conversation_id, _conversations[conversation_id]
    
    # Create a new conversation ID if not provided or invalid
    if not conversation_id:
        conversation_id = str(uuid.uuid4())
    
    # Create and store a new conversation
    _conversations[conversation_id] = Conversation(
        messages=[],
        metadata={"created_at": uuid.uuid1().time}
    )
    
    return conversation_id, _conversations[conversation_id]

def process_query(query: str, 
                  conversation_id: Optional[str] = None, 
                  user_id: Optional[str] = None,
                  metadata: Optional[Dict[str, Any]] = None) -> Response:
    """
    Process a user query and generate a response.
    
    Args:
        query: The user's question or message
        conversation_id: Optional ID of an existing conversation
        user_id: Optional ID of the user
        metadata: Optional additional metadata
        
    Returns:
        Response object containing the answer and metadata
    """
    # Get or create conversation
    conversation_id, conversation = _get_or_create_conversation(conversation_id)
    
    # Add user message to conversation
    conversation.messages.append({
        "role": "user",
        "content": query
    })
    
    # Prepare the user query
    user_query = UserQuery(
        query=query,
        conversation_id=conversation_id,
        user_id=user_id,
        metadata=metadata or {}
    )
    
    # Prepare initial state for workflow
    initial_state: GraphState = {
        "query": query,
        "conversation_id": conversation_id,
        "user_id": user_id,
        "conversation_history": conversation.messages,
        "metadata": user_query.metadata,
    }
    
    # Execute the workflow
    final_state = execute_workflow(_workflow_graph, initial_state)
    
    # Extract answer from final state
    answer = final_state.get("answer", "I couldn't generate a response.")
    
    # Check if we have an art recommendation
    art_recommendation = final_state.get("art_recommendation")
    
    # Add assistant message to conversation
    conversation.messages.append({
        "role": "assistant",
        "content": answer
    })
    
    # Create response
    response = Response(
        answer=answer,
        conversation_id=conversation_id,
        artwork=art_recommendation,
        metadata={
            "execution_path": final_state.get("execution_path", []),
            "detected_emotion": final_state.get("detected_emotion"),
            "retrieval_needed": final_state.get("retrieval_needed", False),
            "offer_art": final_state.get("offer_art", False),
        }
    )
    
    return response

# Simple CLI for testing
if __name__ == "__main__":
    print("Welcome to EmpathyArtRAG! Type 'exit' to quit.")
    conversation_id = None
    
    while True:
        # Get user input
        user_input = input("\nUser: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            break
        
        # Process the query
        response = process_query(user_input, conversation_id)
        conversation_id = response.conversation_id
        
        # Print the answer
        print(f"\nAssistant: {response.answer}")
        
        # Print art recommendation if available
        if response.artwork:
            print("\n--- Art Recommendation ---")
            print(response.artwork["formatted_text"]) 