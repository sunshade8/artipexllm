from ..models import GraphState
import os
from langchain_core.documents import Document

# Tavily API Key가 설정되어 있는지 확인
tavily_api_key = os.environ.get("TAVILY_API_KEY")
use_tavily = tavily_api_key is not None and tavily_api_key != "your-tavily-api-key-here"

# 웹 검색 도구 생성 (API 키가 있는 경우에만)
if use_tavily:
    from langchain_community.tools.tavily_search import TavilySearchResults
    web_search_tool = TavilySearchResults(max_results=3)

# 웹 검색 결과를 문서 형식으로 변환하는 함수
def web_search_and_format(query):
    if use_tavily:
        search_results = web_search_tool.invoke({"query": query})
        documents = []
        for result in search_results:
            # 웹 검색 결과를 Document 형식으로 변환
            doc = Document(
                page_content=result.get("content", ""),
                metadata={"source": result.get("url", "")}
            )
            documents.append(doc)
        return documents
    else:
        # API 키가 없는 경우 더미 응답 반환
        return [Document(
            page_content="Web search is not available because Tavily API key is not configured properly. Please set a valid TAVILY_API_KEY in your .env file.",
            metadata={"source": "system"}
        )]

# workflow/graph.py에서 사용하는 함수
def web_search(state: GraphState) -> GraphState:
    """웹 검색을 실행하고 결과를 반환합니다."""
    question = state["question"]
    documents = web_search_and_format(question)
    return {"documents": documents}

