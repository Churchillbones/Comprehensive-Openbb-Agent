"""
Web Searcher Tool

General web search capability using DuckDuckGo.
Migrated from: comprehensive_agent/processors/web_search.py
"""

from typing import Dict, Any
import logging
from comprehensive_agent.processors.web_search import WebSearchProcessor

logger = logging.getLogger(__name__)

# Create global processor instance
_web_searcher = WebSearchProcessor()


async def search_web(query: str, max_results: int = 5) -> Dict[str, Any]:
    """
    Perform web search using DuckDuckGo

    Args:
        query: Search query string
        max_results: Maximum number of results (default: 5)

    Returns:
        Dict with search results:
            - status: "success" or "error"
            - results: List of search results
            - query: Original query
            - timestamp: Search timestamp
    """
    try:
        result = await _web_searcher.search_web(query, max_results=max_results)

        if "error" in result:
            return {
                "status": "error",
                "error": result["error"],
                "results": [],
                "query": query
            }

        return {
            "status": "success",
            **result
        }

    except Exception as e:
        logger.error(f"Web search failed: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "results": [],
            "query": query
        }


async def detect_web_search_request(query: str) -> bool:
    """
    Detect if query is a web search request

    Args:
        query: User query

    Returns:
        True if query appears to be a web search request
    """
    search_indicators = [
        "search", "find", "look up", "google", "what is", "who is",
        "latest", "recent", "news about"
    ]

    query_lower = query.lower()
    return any(indicator in query_lower for indicator in search_indicators)
