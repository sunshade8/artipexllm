from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document

# --- Tool Initialization ---
# Attempt to initialize the Tavily web search tool.
# Requires the TAVILY_API_KEY environment variable to be set.
try:
    # max_results specifies the number of search results Tavily returns
    web_search_tool = TavilySearchResults(max_results=3) 
    # You can optionally test the tool upon initialization
    # test_result = web_search_tool.invoke({"query": "Test query"})
    # print("Tavily search tool initialized successfully.")
except Exception as e:
    print(f"Error initializing TavilySearchResults: {e}")
    print("Please ensure TAVILY_API_KEY is set in your environment variables.")
    web_search_tool = None # Set to None if initialization fails

# --- Web Search Function ---
# Wrapper function that uses the initialized Tavily tool to perform a search
# and formats the results into a list of Langchain Document objects.
def web_search_and_format(query: str) -> list[Document]:
    """Performs a web search using the initialized tool and formats the results.

    Args:
        query: The search query string.

    Returns:
        A list of Langchain Document objects containing search results.
    """
    # Check if the tool was initialized successfully.
    if web_search_tool is None:
        print("Web search tool is not available.")
        return [] # Return empty list if tool failed to initialize

    try:
        search_results = web_search_tool.invoke({"query": query})
    except Exception as e:
        print(f"Error during Tavily search invocation: {e}")
        return [] # Return empty list on error

    # Format the raw search results (expected to be a list of dicts)
    # into Langchain Document objects for consistent processing downstream.
    documents = []
    if isinstance(search_results, list):
        for result in search_results:
            # Ensure result is a dictionary before accessing keys
            if isinstance(result, dict):
                doc = Document(
                    page_content=result.get("content", ""),
                    metadata={
                        "source": result.get("url", ""),
                        "title": result.get("title", "") # Optionally add title if available
                        }
                )
                documents.append(doc)
            else:
                 print(f"Warning: Unexpected item format in search results: {result}")
    else:
         print(f"Warning: Unexpected search result format: {type(search_results)}")
         
    return documents 