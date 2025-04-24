from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI # Assuming ChatOpenAI is the type for llm

# === Document Relevance Grading ===

# Pydantic model for the output of the document relevance grader.
class GradeDocuments(BaseModel):
    """A binary score to verify the relevance of a retrieved document: 'yes' if the document is relevant to the query, 'no' if it is not."""
    binary_score: str = Field(
        description="Evaluate whether the document is relevant to the question with a binary score: 'yes' if it is relevant, 'no' if it is not."
    )

# Factory function to create the LangChain Runnable for grading document relevance.
# This chain uses the LLM to determine if a retrieved document is relevant to the user's question.
def create_retrieval_grader(llm: ChatOpenAI):
    """Creates the document retrieval grader chain."""
    structured_llm_grader = llm.with_structured_output(GradeDocuments)
    grader_system_prompt = """
    You are a grader responsible for evaluating whether a retrieved document is relevant to the user's question.
    If the document contains keywords or semantic meaning related to the user's query, mark it as relevant.
    This does not need to be a strict test—the goal is simply to filter out clearly irrelevant results.
    Provide a binary score: 'yes' if the document is relevant, or 'no' if it is not.
    """
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", grader_system_prompt),
            ("human", "Retrieved_document: \n\n {document} \n\n User Quetsion: {question}"),
        ]
    )
    retrieval_grader = grade_prompt | structured_llm_grader
    return retrieval_grader

# === Hallucination Grading ===

# Pydantic model for the output of the hallucination grader.
# Checks if the generated answer is grounded in the provided documents.
class GradeHallucinations(BaseModel):
    """A binary score indicating the presence of hallucination in the generated answer:
    'yes' if hallucination is present, 'no' if it is not."""
    binary_score: str = Field(
        description="Indicate whether the answer is grounded in the provided facts with a 'yes' or 'no'." # Modified description slightly for clarity
    )

# Factory function to create the LangChain Runnable for grading hallucinations.
# This chain uses the LLM to check if the generated answer is supported by the provided documents.
def create_hallucination_grader(llm: ChatOpenAI):
    """Creates the hallucination grader chain."""
    structured_llm_hallucination = llm.with_structured_output(GradeHallucinations)
    hallucination_system_prompt = """
    You are a grader responsible for evaluating whether an LLM-generated answer is grounded in or supported by the retrieved set of facts.
    Provide a binary score: 'yes' if the answer is grounded in/supported by the facts, or 'no' if it is not.
    """
    hallucination_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", hallucination_system_prompt),
            ("human", "Fact set: \n\n {documents} \n\n LLM-generated answer: {generation}"),
        ]
    )
    hallucination_grader = hallucination_prompt | structured_llm_hallucination
    return hallucination_grader

# === Answer Relevance Grading ===

# Pydantic model for the output of the answer relevance grader.
# Checks if the generated answer actually addresses the user's question.
class GradeAnswer(BaseModel):
    """A binary score to evaluate the appropriateness of the answer to the question."""
    binary_score: str = Field(
        description="Indicate whether the answer addresses the question with a 'yes' or 'no'." # Corrected typo from 'ndicate'
    )

# Factory function to create the LangChain Runnable for grading answer relevance.
# This chain uses the LLM to determine if the generated answer is a relevant response to the question.
def create_answer_grader(llm: ChatOpenAI):
    """Creates the answer relevance grader chain."""
    structured_llm_answer = llm.with_structured_output(GradeAnswer)
    answer_system_prompt = """
    You are a grader responsible for evaluating whether the answer resolves or addresses the question.
    Provide a binary score: 'yes' if the answer resolves the question, or 'no' if it does not.
    """
    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", answer_system_prompt),
            # Corrected to use English for consistency, assuming LLM works better this way
            ("human", "User question: \n\n {question} \n\n LLM generated answer: {generation}"), 
        ]
    )
    answer_grader = answer_prompt | structured_llm_answer
    return answer_grader 