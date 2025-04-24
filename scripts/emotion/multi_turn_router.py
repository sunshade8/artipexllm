"""
Multi-turn router for the ArtiTech LangGraph system.

This module provides router functions to handle multi-turn conversations,
particularly detecting direct art requests in follow-up messages.
"""

from typing import Dict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def multi_turn_router(state: Dict) -> str:
    """
    Router function for multi-turn conversations using an LLM for intent classification.

    This function analyzes the user's message to determine if it's a direct
    art request or a standard query requiring RAG/web search.

    Args:
        state: Current graph state containing user message

    Returns:
        Routing decision: "art_request", "memory", "web_search", or "vectorstore"
    """
    print("==== [LLM-Based Multi-Turn Router] ====")
    question = state.get("question", "")
    
    # --- Check for pending art suggestion response --- 
    offer_art = state.get("offer_art", False) # Was art suggested based on emotion previously?
    processed_suggestion = state.get("processed_suggestion", False) # Was the response already processed?
    
    # If an offer was made in the previous turn and we haven't processed the response yet
    if offer_art and not processed_suggestion:
        # Assume the current question is the user's response to the offer
        print("Pending art suggestion found. Routing to process response.")
        return "process_art_response" 
        # NOTE: We might need to ensure 'question' is correctly populated 
        # in the state for process_art_response to analyze.
        # LangGraph usually handles passing the latest input.
        
    # --- If no pending suggestion, classify current intent --- 
    print("No pending suggestion. Classifying user intent.")
    if not question:
        print("No question found in state, defaulting to standard route.")
        # Fallback to original RAG routing if no question
        try:
            from __main__ import route_question # Avoid top-level import issues
            return route_question(state)
        except ImportError:
             print("Warning: route_question not found. Defaulting to 'web_search'.")
             return "web_search" # Or a safer default

    # Initialize LLM
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

    # Define prompt for intent classification
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an intent classification assistant.\n"
         "Analyze the user's message and classify it into one of the following categories:\n"
         "1. 'art_request': The user is explicitly asking for art recommendations, suggestions, or related artistic help (e.g., \"show me art\", \"recommend a painting for sadness\", \"can you find art about joy?\").\n"
         "2. 'standard_query': The user is asking a general question, seeking information, expressing feelings without asking for art, or making a statement that needs a researched or generated answer.\n\n"
         "Respond ONLY with 'art_request' or 'standard_query'."
        ),
        ("human", f"User message: {question}\n\nClassification:")
    ])

    # Create and invoke the chain
    intent_chain = prompt | llm | StrOutputParser()

    try:
        intent = intent_chain.invoke({}).strip().lower()
        print(f"LLM Router Intent: {intent}")

        if intent == "art_request":
            # Route to the node that handles direct requests
            print("Routing to direct art request handler.")
            return "art_request"
        elif intent == "standard_query":
            # Defer to the original RAG/web search router
            try:
                from __main__ import route_question # Avoid top-level import issues
                standard_route = route_question(state)
                print(f"Routing to standard path: {standard_route}")
                return standard_route
            except ImportError:
                 print("Warning: route_question not found. Defaulting to 'web_search'.")
                 return "web_search" # Or a safer default

        else:
            print(f"Warning: LLM returned unexpected intent '{intent}'. Defaulting to standard route.")
            # Fallback to original RAG routing on unexpected LLM output
            try:
                from __main__ import route_question # Avoid top-level import issues
                return route_question(state)
            except ImportError:
                 print("Warning: route_question not found. Defaulting to 'web_search'.")
                 return "web_search" # Or a safer default

    except Exception as e:
        print(f"Error during LLM intent classification: {e}. Defaulting to standard route.")
        # Fallback to original RAG routing on error
        try:
            from __main__ import route_question # Avoid top-level import issues
            return route_question(state)
        except ImportError:
            print("Warning: route_question not found. Defaulting to 'web_search'.")
            return "web_search" # Or a safer default

# Remove or comment out the old is_follow_up_art_request function if it's no longer needed
# def is_follow_up_art_request(question: str) -> bool: ... 