from typing import Literal, Dict
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI # Assuming ChatOpenAI is the type for llm
# from langchain_core.output_parsers import StrOutputParser # Only needed by removed routers

# Model to structure the output of the routing LLM call.
# It forces the LLM to choose between 'vectorstore' and 'web_search'.
class RouteQuery(BaseModel):
    """Routes the user's query to the most relevant data source."""
    datasource: Literal["vectorstore", "web_search"] = Field(
        ...,
        description="Routes the user's query to either a web search or a vector store, depending on relevance.",
    )

# Factory function to create the LangChain Runnable for routing questions.
def create_question_router(llm: ChatOpenAI):
    """Creates the question routing chain.

    Args:
        llm: The language model instance (e.g., ChatOpenAI).

    Returns:
        The configured question router chain.
    """
    # Configure the LLM to directly output structured data (RouteQuery model).
    structured_llm_router = llm.with_structured_output(RouteQuery)

    # System prompt guiding the LLM on how to make the routing decision.
    router_system_prompt = """
    You are an expert in routing user queries to either a vector store or a web search.
    The vector store contains various PDF documents related to topics such as color psychology, art therapy, and formal elements of art.
    For queries related to these topics, use the vector store; for all other queries, use web search.
    """

    # Use the system prompt and the user's question as input to the router.
    route_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", router_system_prompt),
            ("human", "{question}"),
        ]
    )

    question_router = route_prompt | structured_llm_router
    return question_router

# --- Removed Routers Below ---
# def route_question(state: Dict) -> str: ...
# def multi_turn_router(state: Dict) -> str: ...
# def art_suggestion_router(state: Dict) -> str: ... 