# src/graph_builder.py
# --- Standard Library Imports ---
# (No standard library imports used directly in this snippet)

# --- Third-party Library Imports ---
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.base import BaseCheckpointSaver

# --- Local Application/Library Specific Imports ---

# Attempt to import the central state definition for the graph
# Assuming GraphState is defined in src.state
# If it's elsewhere, adjust the import
try:
    from .state import GraphState
except ImportError:
    # Fallback or error handling if state is not in src.state
    # This might happen if sys.path isn't set correctly yet when this is imported
    # Or if the file structure is different
    print("Warning: Could not import GraphState from .state. Ensure src is in sys.path and src/state.py exists.")
    # As a placeholder, using dict, but this WILL likely fail later if GraphState is complex
    GraphState = dict

# Attempt to import node functions defined within the src package
# Nodes likely defined in src/graph_nodes.py
try:
    from .graph_nodes import (
        retrieve_documents, web_search, grade_documents,
        generate_answer_vectorstore, generate_answer_websearch,
        rewrite_question, memory_query_handler, decide_to_generate,
        check_hallucination, emotion_detect_grader, generate_art_explanation
    )
except ImportError:
    print("Warning: Could not import standard graph nodes from .graph_nodes.")
    # Define dummy functions or raise error if essential
    def dummy_node(state): return state
    retrieve_documents = web_search = grade_documents = generate_answer_vectorstore = \
    generate_answer_websearch = rewrite_question = memory_query_handler = \
    emotion_detect_grader = generate_art_explanation = dummy_node
    def dummy_condition(state): return "end"
    decide_to_generate = check_hallucination = dummy_condition


# Attempt to import nodes and routers from the scripts/emotion directory.
# This structure suggests 'scripts' might contain experimental or separate modules.
# Note: Importing from a parallel 'scripts' directory can be fragile. Consider moving
# these functions into the 'src' package for better organization if they are core.
# --- Import Nodes/Routers from scripts/emotion (Restored) ---
# These might need sys.path manipulation handled *before* this module is imported.
try:
    from scripts.emotion.multi_turn_router import multi_turn_router
    from scripts.emotion.art_response_handler import process_art_request
    # from scripts.emotion.emotion_detect import emotion_detect_node # Might be redundant if covered by emotion_detect_grader
    from scripts.emotion.art_suggestion import (
        suggest_art_node,
        process_art_suggestion_response,
        art_suggestion_router
    )
    from scripts.emotion.art_recommendation_connector import (
        generate_art_recommendations_node,
        display_art_recommendations_node
        # initialize_art_recommendation_system # Initialization likely happens outside graph build
    )
except ImportError as e:
    print(f"Warning: Could not import functions from scripts.emotion.*: {e}")
    print("Ensure scripts directory is accessible or move these functions to src.")
    # Define dummy functions or raise error if essential
    if 'dummy_router' not in locals(): # Define if not already defined
        def dummy_router(state): return "end"
    if 'dummy_node' not in locals(): # Define if not already defined
        def dummy_node(state): return state
    multi_turn_router = art_suggestion_router = dummy_router
    process_art_request = suggest_art_node = process_art_suggestion_response = \
    generate_art_recommendations_node = display_art_recommendations_node = dummy_node


def create_adaptive_rag_graph(checkpointer: BaseCheckpointSaver):
    """
    Creates and compiles the adaptive RAG LangGraph workflow.

    Args:
        checkpointer: The checkpointer instance (e.g., MemorySaver) to use.

    Returns:
        Compiled LangGraph application.
    """
    # Initialize the state machine graph, defining the structure of our application state
    workflow = StateGraph(GraphState)

    # --- Define Nodes ---
    # Add each function as a node in the graph. Nodes represent units of work.
    workflow.add_node("memory_handler", memory_query_handler)
    workflow.add_node("web_search", web_search)
    workflow.add_node("retrieve_documents", retrieve_documents)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate_answer_vectorstore", generate_answer_vectorstore)
    workflow.add_node("generate_answer_websearch", generate_answer_websearch)
    workflow.add_node("rewrite_question", rewrite_question)
    workflow.add_node("emotion_detect", emotion_detect_grader)
    workflow.add_node("suggest_art", suggest_art_node)
    workflow.add_node("process_art_response", process_art_suggestion_response)
    workflow.add_node("generate_art_recommendations", generate_art_recommendations_node)
    workflow.add_node("display_art_recommendations", display_art_recommendations_node)
    workflow.add_node("process_art_request", process_art_request)
    workflow.add_node("generate_art_explanation", generate_art_explanation)
    # workflow.add_node("select_art_for_negative_emotion", select_art_for_negative_emotion) # Uncomment if needed

    # --- Define Edges and Conditional Edges ---
    # Connect the nodes to define the flow of execution.

    # --- Entry Point ---
    # The START node transitions based on the 'multi_turn_router' decision.
    # This router determines the initial path based on user input or context.
    workflow.add_conditional_edges(
        START,
        multi_turn_router,
        {
            "memory": "memory_handler",
            "web_search": "web_search",
            "vectorstore": "retrieve_documents",
            "art_request": "process_art_request",
            "process_art_response": "process_art_response"
        }
    )

    # --- Memory Handling Branch ---
    # If the router directs to 'memory', handle the memory query and end.
    workflow.add_edge("memory_handler", END)

    # --- Web Search Branch ---
    # If the router directs to 'web_search', perform a search, generate an answer,
    # check emotion, and potentially suggest art or end.
    workflow.add_edge("web_search", "generate_answer_websearch")
    workflow.add_edge("generate_answer_websearch", "emotion_detect")

    # --- Vectorstore Retrieval Branch ---
    # If the router directs to 'vectorstore', retrieve documents, grade them,
    # potentially rewrite the question or generate an answer.
    workflow.add_edge("retrieve_documents", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "rewrite": "rewrite_question",
            "generate": "generate_answer_vectorstore"
        }
    )
    workflow.add_edge("rewrite_question", "retrieve_documents")

    # --- Generation & Hallucination Check Branch (after Vectorstore) ---
    # After generating an answer from the vectorstore, check for hallucinations.
    # Based on the check, either regenerate, proceed to emotion detection, or rewrite the question.
    workflow.add_conditional_edges(
        "generate_answer_vectorstore",
        check_hallucination,
        {
            "hallucination": "generate_answer_vectorstore",
            "relevant": "emotion_detect",
            "not_relevant": "rewrite_question"
        }
    )

    # --- Emotion Detection & Art Suggestion Routing ---
    # After generating a valid answer (from web or vectorstore), detect emotion.
    # Based on emotion, decide whether to show art directly, offer to suggest art, or skip art.
    workflow.add_conditional_edges(
        "emotion_detect",
        art_suggestion_router,
        {
            "show_art": "generate_art_recommendations",
            "suggest_art": "suggest_art",
            "skip_art": END,
        }
    )

    # --- Art Suggestion Offer Branch ---
    # If suggesting art, the flow ends here, awaiting user response ("Yes" or "No").
    # The user's response will re-enter the graph via the START node and be routed
    # by 'multi_turn_router' to 'process_art_response'.
    workflow.add_edge("suggest_art", END)

    # --- Process User Response to Art Suggestion Branch ---
    # Handles the user's "Yes" or "No" to the art suggestion.
    # Routes to showing art if "Yes", skips if "No". The 'suggest_art' path here
    # likely represents an edge case or invalid state and should ideally end.
    workflow.add_conditional_edges(
        "process_art_response",
        art_suggestion_router,
        {
            "show_art": "generate_art_recommendations",
            "skip_art": END,
            "suggest_art": END
        }
    )

    # --- Direct Art Request Branch ---
    # If the initial routing was 'art_request', process it and proceed to generate recommendations.
    workflow.add_edge("process_art_request", "generate_art_recommendations")

    # --- Art Recommendation & Display Branch ---
    # Generate and display art recommendations, followed by an explanation.
    workflow.add_edge("generate_art_recommendations", "display_art_recommendations")
    workflow.add_edge("display_art_recommendations", "generate_art_explanation")
    workflow.add_edge("generate_art_explanation", END)

    # --- Compile the Graph ---
    # Finalize the graph structure and associate the checkpointer for state persistence.
    adaptive_rag_app = workflow.compile(checkpointer=checkpointer)
    print("Adaptive RAG Graph Compiled.")
    return adaptive_rag_app 