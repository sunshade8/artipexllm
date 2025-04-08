"""
Workflow graph definition for the EmpathyArtRAG system.

This file contains the workflow graph that defines how user queries are processed.
"""

from .models import WorkflowNode, WorkflowGraph, GraphState
from .handlers.question_handler import process_question
from .handlers.emotion_handler import detect_emotion, should_offer_art
from .handlers.routing_handler import determine_needs_retrieval
from .handlers.retrieval_handler import retrieve_context
from .generators.answer_generator import generate_answer
from .generators.art_generator import generate_art_recommendation

def create_workflow_graph() -> WorkflowGraph:
    """
    Create the workflow graph for processing user queries.
    
    Returns:
        The workflow graph with all nodes and connections.
    """
    # Define the workflow nodes
    nodes = {
        # Entry point - processes the user's question and initializes state
        "process_question": WorkflowNode(
            id="process_question",
            func=process_question,
            next_nodes=["detect_emotion"]
        ),
        
        # Analyzes the question to detect emotions
        "detect_emotion": WorkflowNode(
            id="detect_emotion",
            func=detect_emotion,
            next_nodes=["determine_needs_retrieval"]
        ),
        
        # Determines if retrieval is needed
        "determine_needs_retrieval": WorkflowNode(
            id="determine_needs_retrieval",
            func=determine_needs_retrieval,
            next_nodes=["retrieve_context", "check_should_offer_art"]
        ),
        
        # Retrieves relevant context from the RAG system
        "retrieve_context": WorkflowNode(
            id="retrieve_context",
            func=retrieve_context,
            next_nodes=["check_should_offer_art"]
        ),
        
        # Determines if art recommendations should be offered
        "check_should_offer_art": WorkflowNode(
            id="check_should_offer_art",
            func=should_offer_art,
            next_nodes=["generate_art_recommendation", "generate_answer"]
        ),
        
        # Generates art recommendations if needed
        "generate_art_recommendation": WorkflowNode(
            id="generate_art_recommendation",
            func=generate_art_recommendation,
            next_nodes=["generate_answer"]
        ),
        
        # Generates the final answer
        "generate_answer": WorkflowNode(
            id="generate_answer",
            func=generate_answer,
            next_nodes=[]
        )
    }
    
    # Create the workflow graph
    workflow_graph = WorkflowGraph(
        nodes=nodes,
        start_node="process_question"
    )
    
    return workflow_graph

def execute_workflow(graph: WorkflowGraph, initial_state: GraphState) -> GraphState:
    """
    Execute the workflow graph from the start node to completion.
    
    Args:
        graph: The workflow graph to execute
        initial_state: The initial state for the workflow
        
    Returns:
        The final state after workflow execution
    """
    # Start with the initial state and the start node
    current_state = initial_state
    current_node_id = graph.start_node
    
    # Execute nodes until we reach a terminal node
    visited_nodes = []
    while current_node_id:
        # Get the current node
        current_node = graph.nodes[current_node_id]
        
        # Log the execution path
        visited_nodes.append(current_node_id)
        print(f"Executing node: {current_node_id}")
        
        # Execute the node function
        try:
            current_state = current_node.func(current_state)
        except Exception as e:
            print(f"Error in node {current_node_id}: {e}")
            # Add error to state and try to continue to next node
            current_state["errors"] = current_state.get("errors", []) + [
                {"node": current_node_id, "error": str(e)}
            ]
        
        # Determine the next node
        next_node_id = None
        if current_node.next_nodes:
            if len(current_node.next_nodes) == 1:
                # If there's only one option, use it
                next_node_id = current_node.next_nodes[0]
            else:
                # If there are multiple options, check for routing_result in state
                routing_key = f"{current_node_id}_routing"
                if routing_key in current_state:
                    route_index = current_state[routing_key]
                    if 0 <= route_index < len(current_node.next_nodes):
                        next_node_id = current_node.next_nodes[route_index]
                
                if next_node_id is None and current_node.next_nodes:
                    # Default to first option if no routing info
                    next_node_id = current_node.next_nodes[0]
        
        # Update the current node
        current_node_id = next_node_id
    
    # Add execution path to final state
    current_state["execution_path"] = visited_nodes
    
    return current_state 