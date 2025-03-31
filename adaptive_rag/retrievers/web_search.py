from ..models import GraphState

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document

# 웹 검색 도구 생성
web_search_tool = TavilySearchResults(max_results=3)

# 웹 검색 결과를 문서 형식으로 변환하는 함수
def web_search_and_format(query):
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

