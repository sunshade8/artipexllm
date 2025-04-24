from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever

# === Document Formatting ===

# Formats a list of Langchain Document objects into a single string for context.
# Includes metadata (source, page) in an XML-like structure.
# (Matches the format expected by some prompts and used in LangGraph nodes)
def format_docs(documents):
    """Formats documents with content, source, and page metadata in XML-like tags."""
    return "\n\n".join(
        [
            f'<document><content>{doc.page_content}</content><source>{doc.metadata.get("source", "Unknown")}</source><page>{doc.metadata.get("page", -1)+1}</page></document>'
            for doc in documents
        ]
    )

# === RAG Chain Creation and Execution ===

# (DEPRECATED/Example) - Creates a basic RAG chain using LCEL.
# This version might be simpler but less flexible for complex state management needed in LangGraph.
def create_rag_chain(retriever, prompt_template, llm_model_name="gpt-4o", temperature=0):
    """Creates the main RAG chain using LangChain Expression Language (LCEL)."""
    llm = ChatOpenAI(model_name=llm_model_name, temperature=temperature)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()} # Pass query to retriever and then format, pass query directly to question
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return rag_chain

# (DEPRECATED/Example) - Runs the RAG process step-by-step: retrieve, format, invoke LLM, format citations.
# This provides more explicit control but is less integrated than a full LCEL chain or LangGraph.
def run_rag_chain(query, retriever, prompt_template, llm_model_name="gpt-4o", temperature=0, filter_threshold=0.5):
    """Runs the RAG chain: retrieve, filter, invoke LLM, format response."""

    # 1. Retrieve documents (using the retriever's invoke method)
    # Note: The ensemble retriever doesn't directly support filtering threshold in invoke.
    # Filtering logic needs to be applied separately if needed, or integrated differently.
    # For simplicity here, we rely on the retriever's internal ranking.
    # retrieved_docs = retriever.invoke(query)

    # If you still need the explicit filtering step (like in the original notebook):
    # We need the filter function, let's assume it's imported or defined here.
    # from .retriever import filter_relevant_documents # Assuming it's in retriever.py
    # retrieved_docs = retriever.get_relevant_documents(query) # Deprecated but used in original code
    retrieved_docs = retriever.invoke(query) # Use invoke for retriever
    # filtered_docs = filter_relevant_documents(query, retrieved_docs, threshold=filter_threshold)
    # For now, let's use the retrieved docs directly as the ensemble retriever already ranks them.
    filtered_docs = retrieved_docs # Using retrieved docs directly
    print(f"Retrieved {len(filtered_docs)} documents for query.")

    if not filtered_docs:
        return "Sorry, I couldn't find relevant information to answer your question.", []

    # 2. Format context
    context_text = format_docs(filtered_docs)

    # 3. Create LLM and Prompt
    llm = ChatOpenAI(model_name=llm_model_name, temperature=temperature)

    # 4. Format the final prompt
    final_prompt_str = prompt_template.format(context=context_text, question=query)

    # 5. Invoke LLM
    response = llm.invoke([HumanMessage(content=final_prompt_str)])
    ai_message_content = response.content

    # 6. Format citations
    citations = [
        f"[Source: {doc.metadata.get('source', 'Unknown Document')}, Page {doc.metadata.get('page', 'N/A')}]"
        for doc in filtered_docs
    ]
    unique_citations = sorted(list(set(citations))) # Keep unique citations

    # 7. Combine response and citations
    final_response = f"{ai_message_content}\n\nSources:\n" + "\n".join(unique_citations)

    return final_response, filtered_docs # Return response and the docs used


# === LCEL RAG Chain (More Advanced Example) ===

# Creates a RAG chain using LCEL that handles retrieval, formatting, generation,
# and citation formatting in a single, expressive pipeline.
# Passes retrieved documents along for citation formatting.
def create_lcel_rag_chain(retriever, prompt_template, llm_model_name="gpt-4o", temperature=0):
    """Creates the main RAG chain using LangChain Expression Language (LCEL)."""
    llm = ChatOpenAI(model_name=llm_model_name, temperature=temperature)

    # Define the sequence of operations
    # 1. Retrieve documents based on the input query
    # 2. Format the retrieved documents into a single string context
    # 3. Prepare the input for the prompt (context and question)
    # 4. Apply the prompt template
    # 5. Pass the formatted prompt to the LLM
    # 6. Parse the LLM output to a string
    lcel_chain = (
        {
            "context": itemgetter("query") | retriever | format_docs,
            "question": itemgetter("query"),
            "retrieved_docs": itemgetter("query") | retriever # Pass retriever results for citation
        }
        | RunnablePassthrough.assign( # Pass retrieved docs along for citation
            ai_response = (
                {
                    "context": itemgetter("context"),
                    "question": itemgetter("question")
                }
                | prompt_template
                | llm
                | StrOutputParser()
            )
        )
        | (lambda x: format_response_with_citations(x['ai_response'], x['retrieved_docs'])) # Final formatting step
    )
    return lcel_chain

# Helper function used within the LCEL chain to format the final response
# by appending source citations from the retrieved documents.
def format_response_with_citations(ai_response, retrieved_docs):
    citations = [
        f"[Source: {doc.metadata.get('source', 'Unknown Document')}, Page {doc.metadata.get('page', 'N/A')}]"
        for doc in retrieved_docs
    ]
    unique_citations = sorted(list(set(citations)))
    return f"{ai_response}\n\nSources:\n" + "\n".join(unique_citations)


# Example usage comments for the LCEL chain
# query = "Your query here"
# rag_chain = create_lcel_rag_chain(ensemble_retriever, prompt_template)
# final_result = rag_chain.invoke({"query": query})
# print(final_result)

# === Helper Functions for RAG/LangGraph ===

# Helper function primarily for testing or manual invocation outside the graph.
# Retrieves documents for a question, formats them, and prepares the input dict.
def prepare_rag_input(question: str, retriever: BaseRetriever, conversation_history: str = "") -> dict:
    """Retrieves documents, formats them, and prepares the input dictionary for the RAG chain."""
    retrieved_docs = retriever.invoke(question)
    # Use the renamed format_docs function
    formatted_context = format_docs(retrieved_docs)
    print(f"Prepared context for query: '{question}' using {len(retrieved_docs)} documents.")
    return {
        "context": formatted_context,
        "question": question,
        "conversation_history": conversation_history
    }

# Factory function to create the core generation chain used within LangGraph nodes.
# This is a simple chain: takes context/question via prompt, sends to LLM, parses output.
def create_og_chain(llm: ChatOpenAI, prompt: PromptTemplate):
    """Creates the simple RAG chain: prompt | llm | parser."""
    return (
        prompt
        | llm
        | StrOutputParser()
    ) 