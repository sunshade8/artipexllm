from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI # Assuming this is the LLM type

# Import the specific prompt needed
from src.prompts import rewrite_prompt

# Factory function to create the LangChain Runnable for rewriting questions.
# This chain uses a specific prompt (rewrite_prompt) and the provided LLM.
def create_question_rewriter(llm: ChatOpenAI):
    """Creates the question rewriting chain.

    Args:
        llm: The language model instance (e.g., ChatOpenAI).

    Returns:
        The configured question rewriter chain.
    """
    question_rewriter = rewrite_prompt | llm | StrOutputParser()
    return question_rewriter 