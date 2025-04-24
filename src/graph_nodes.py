# src/graph_nodes.py

from src.state import GraphState
from src.chain import format_docs
from src.tools import web_search_and_format
# from scripts.emotion.emotion_detect import emotion_detect_node # Old import
from src.emotion_nodes import emotion_detect_node # <-- New import

# === Node Functions for the LangGraph Workflow ===

# Note: Many of these functions rely on objects (retrievers, graders, chains)
# being available in the global scope (e.g., initialized in a Jupyter notebook)
# when the graph is compiled and run. This is indicated by `from __main__ import ...`.

# --- Retrieval and Web Search Nodes ---

# Node to retrieve relevant documents from the configured vector store (ensemble_retriever).
def retrieve_documents(state: GraphState):
    print("==== [Node: retrieve_documents] ====")
    question = state["question"]
    # Assuming ensemble_retriever is globally available
    from __main__ import ensemble_retriever 
    documents = ensemble_retriever.invoke(question)
    return {"documents": documents}

# Node to perform a web search using the provided tool and format the results.
def web_search(state: GraphState):
    print("==== [Node: web_search] ====")
    question = state["question"]
    documents = web_search_and_format(question)
    return {"documents": documents}

# --- Grading Nodes ---

# Node to grade the relevance of retrieved documents against the user question.
# Filters out documents deemed irrelevant by the retrieval_grader.
def grade_documents(state: GraphState):
    print("==== [Node: grade_documents] ====")
    question = state["question"]
    documents = state["documents"]
    # Assuming retrieval_grader is globally available
    from __main__ import retrieval_grader 
    
    filtered_docs = []
    for doc in documents:
        score = retrieval_grader.invoke({"question": question, "document": doc.page_content})
        grade = score.binary_score
        if grade == "yes":
            print("✓ Relevant Document Found")
            filtered_docs.append(doc)
        else:
            print("✗ Irrelevant Document Filtered Out")
            # Optionally remove source/page metadata if not relevant
            # doc.metadata = {"relevance": "no"}
            # filtered_docs.append(doc) # Or just skip
    return {"documents": filtered_docs}

# Node to check the generated answer for hallucinations (consistency with documents)
# and relevance to the original question.
# Returns a string indicating the outcome ('relevant', 'not_relevant', 'hallucination').
def check_hallucination(state: GraphState):
    print("==== [Node: check_hallucination] ====")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    # Assuming hallucination_grader and answer_grader are globally available
    from __main__ import hallucination_grader, answer_grader
    
    docs_text = format_docs(documents)
    hall_score = hallucination_grader.invoke({"documents": docs_text, "generation": generation})
    
    if hall_score.binary_score == "yes":
        print("✓ Answer Grounded in Documents")
        answer_score = answer_grader.invoke({"question": question, "generation": generation})
        if answer_score.binary_score == "yes":
            print("✓ Answer Addresses Question")
            return "relevant"
        else:
            print("✗ Answer Does Not Address Question")
            return "not_relevant"
    else:
        print("✗ Hallucination Detected")
        return "hallucination"

# --- Generation Nodes ---

# Node to generate an answer based on the context from retrieved vector store documents.
# It uses the og_chain_modular and includes conversation history.
def generate_answer_vectorstore(state: GraphState):
    print("==== [Node: generate_answer_vectorstore] ====")
    question = state["question"]
    documents = state["documents"]
    conversation_history = state.get("conversation_history", [])
    # Assuming og_chain_modular is globally available
    from __main__ import og_chain_modular

    # Simplified history formatting
    formatted_history = "\n".join([f"User: {turn['question']}\nAssistant: {turn['answer']}" for turn in conversation_history[-3:]])
    formatted_docs = format_docs(documents)
    
    generation = og_chain_modular.invoke({
        "context": formatted_docs,
        "question": question,
        "conversation_history": formatted_history
    })

    # Append to history immediately after generation
    new_history = conversation_history + [{'question': question, 'answer': generation}]

    return {"generation": generation, "conversation_history": new_history}

# Node to generate an answer based on the context from web search results.
# Currently uses the same chain (og_chain_modular) as the vectorstore generation.
def generate_answer_websearch(state: GraphState):
    # This node might be simpler if web search results don't need the complex prompt logic
    # Or it could be identical to generate_answer_vectorstore if using the same chain
    print("==== [Node: generate_answer_websearch] ====")
    question = state["question"]
    documents = state["documents"]
    conversation_history = state.get("conversation_history", [])
    # Assuming og_chain_modular is globally available
    from __main__ import og_chain_modular
    
    formatted_history = "\n".join([f"User: {turn['question']}\nAssistant: {turn['answer']}" for turn in conversation_history[-3:]])
    formatted_docs = format_docs(documents)

    generation = og_chain_modular.invoke({
        "context": formatted_docs,
        "question": question,
        "conversation_history": formatted_history
    })
    
    new_history = conversation_history + [{'question': question, 'answer': generation}]
    return {"generation": generation, "conversation_history": new_history}

# Node to generate a simple concluding remark after art has been shown or suggested.
# Note: This currently overwrites the main 'generation' field in the state.
def generate_art_explanation(state: GraphState):
    """
    Generates a concluding remark after suggesting/showing art.
    (Reverted to original version)
    """
    print("==== [Node: generate_art_explanation] ====")
    
    # Original simple explanation
    explanation = "I hope viewing this artwork provides some comfort or perspective. Let me know if you'd like to explore other options or talk more."

    # This OVERWRITES the main generation. Consider appending or handling differently if needed.
    return {"generation": explanation}

# --- Rewriting Node ---

# Node to rewrite the user's question for potentially better retrieval results.
# Uses the question_rewriter chain.
def rewrite_question(state: GraphState):
    print("==== [Node: rewrite_question] ====")
    question = state["question"]
    # Assuming question_rewriter is globally available
    from __main__ import question_rewriter 
    improved_question = question_rewriter.invoke({"question": question})
    print(f"Original Question: {question}")
    print(f"Rewritten Question: {improved_question}")
    return {"question": improved_question}

# --- Decision Nodes ---

# Node to decide whether to generate an answer or rewrite the question
# based on whether any relevant documents were found after grading.
def decide_to_generate(state: GraphState):
    print("==== [Node: decide_to_generate] ====")
    filtered_docs = state.get("documents", [])
    # Check if the list is actually empty, not just contains the string "documents"
    if not filtered_docs:
        print("Decision: No Relevant Documents -> Rewrite Question")
        return "rewrite"
    else:
        print(f"Decision: {len(filtered_docs)} Relevant Document(s) -> Generate Answer")
        return "generate"

# --- Memory Node ---

# Node to handle specific questions related to conversation history.
# If a memory-related keyword is detected, it provides a summary of recent turns.
# Otherwise, it passes the state through (intended to be routed away from). 
def memory_query_handler(state: GraphState):
    # This node seems designed to intercept specific questions about history
    # and provide a canned response, bypassing the main RAG flow.
    print("==== [Node: memory_query_handler] ====")
    question = state["question"]
    conversation_history = state.get("conversation_history", [])
    
    memory_questions = [
        "what was my question?", "what did i ask again?", "previous question",
        "what i asked before", "what did i ask", "the content of my previous question",
        "our immediate previous conversation", "what content", "what did we talk about",
        "we talked about", "we said / you mentioned", "we mentioned",
        "what we talked about before", "what we talked about", "just now",
        "so far / up to now", "our conversation", "which color", "what color",
        "color", "which topic", "what was mentioned earlier / previously mentioned"
    ]
    is_memory_question = any(mem_q in question.lower() for mem_q in memory_questions)

    if is_memory_question and conversation_history:
        print("Handling memory-related question.")
        # Simple response summarizing last few questions
        prev_interactions = "\n".join([f"- You asked: {turn['question']}" for turn in conversation_history[-3:]])
        generation = f"Recently, we discussed:\n{prev_interactions}"
        # Return the generation directly, effectively ending this branch
        return {"generation": generation}
    else:
        # This case should ideally not be reached if routing is correct,
        # but if it is, returning None or an empty dict might cause issues.
        # It's better if the router handles non-memory questions.
        # For safety, let's return the state unchanged, assuming router handles next step.
        print("Not a memory question or no history, passing through.")
        return {}

# --- Emotion Node ---

# Node wrapper that calls the main emotion detection logic from emotion_nodes.py.
# Prepares the input state and integrates the results back into the graph state.
def emotion_detect_grader(state: GraphState):
    print("==== [Node: emotion_detect_grader] ====")
    question = state["question"]
    generation = state.get("generation", "") # Use generation if available
    conversation_history = state.get("conversation_history", [])
    
    # Prepare state for the imported emotion detection node
    # Ensure all required fields for emotion_detect_node are present
    emotion_input_state = {
        "question": question,
        "generation": generation,
        "conversation_history": conversation_history
        # Add any other fields required by emotion_detect_node if necessary
    }
    
    try:
        result_state = emotion_detect_node(emotion_input_state)
        detected_emotion = result_state.get("detected_emotion")
        emotion_confidence = result_state.get("emotion_confidence", 0.0)
        
        if detected_emotion:
            print(f"✓ Emotion Detected: {detected_emotion} (Confidence: {emotion_confidence:.2f})")
            # Merge results back into the main graph state
            return {
                "detected_emotion": detected_emotion,
                "emotion_confidence": emotion_confidence,
                # Pass through other potential results from emotion_detect_node if needed
                # "emotion_result": result_state.get("emotion_result", {})
            }
        else:
            print("✗ No Emotion Detected")
            return {"detected_emotion": None, "emotion_confidence": 0.0}
    except Exception as e:
        print(f"Error during emotion detection: {e}")
        return {"detected_emotion": None, "emotion_confidence": 0.0} 