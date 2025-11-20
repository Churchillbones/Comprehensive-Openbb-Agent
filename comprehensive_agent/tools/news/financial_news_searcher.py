"""
Financial News Searcher Tool

Financial-specific news search with sentiment analysis.
Migrated from: comprehensive_agent/processors/financial_web_search.py
"""

from typing import Dict, Any, List
import logging
from datetime import datetime

try:
    from duckduckgo_search import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

logger = logging.getLogger(__name__)


async def search_financial_news(
    query: str,
    max_results: int = 10,
    analyze_sentiment: bool = True
) -> Dict[str, Any]:
    """
    Search for financial news and optionally analyze sentiment

    Args:
        query: Search query (e.g., "AAPL stock news")
        max_results: Maximum number of results (default: 10)
        analyze_sentiment: Whether to perform sentiment analysis (default: True)

    Returns:
        Dict with financial news results:
            - status: "success" or "error"
            - results: List of news articles with sentiment
            - query: Original query
            - average_sentiment: Average sentiment score if analyzed
    """
    if not DDGS_AVAILABLE:
        return {
            "status": "error",
            "error": "DuckDuckGo search not available. Install: pip install duckduckgo-search",
            "results": []
        }

    try:
        logger.info(f"Searching financial news for: {query}")

        # Add financial context to query if not already present
        financial_terms = ["stock", "market", "finance", "trading", "earnings"]
        if not any(term in query.lower() for term in financial_terms):
            query = f"{query} stock market news"

        # Perform search
        results = []
        with DDGS() as ddgs:
            search_results = list(ddgs.text(
                query,
                max_results=max_results,
                region="us-en",
                safesearch="off"
            ))

            for result in search_results:
                article = {
                    "title": result.get("title", ""),
                    "url": result.get("href", ""),
                    "snippet": result.get("body", ""),
                    "source": result.get("href", "").split("/")[2] if result.get("href") else "Unknown"
                }

                # Analyze sentiment if requested and TextBlob available
                if analyze_sentiment and TEXTBLOB_AVAILABLE:
                    text_for_analysis = f"{article['title']}. {article['snippet']}"
                    blob = TextBlob(text_for_analysis)

                    article["sentiment"] = {
                        "polarity": round(blob.sentiment.polarity, 3),
                        "subjectivity": round(blob.sentiment.subjectivity, 3),
                        "label": _get_sentiment_label(blob.sentiment.polarity)
                    }

                results.append(article)

        # Calculate average sentiment if analyzed
        average_sentiment = None
        if analyze_sentiment and TEXTBLOB_AVAILABLE and results:
            sentiments = [r.get("sentiment", {}).get("polarity", 0) for r in results]
            average_sentiment = round(sum(sentiments) / len(sentiments), 3)

        logger.info(f"Found {len(results)} financial news articles")

        return {
            "status": "success",
            "results": results,
            "query": query,
            "count": len(results),
            "average_sentiment": average_sentiment,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Financial news search failed: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "results": [],
            "query": query
        }


def _get_sentiment_label(polarity: float) -> str:
    """Convert polarity score to sentiment label"""
    if polarity > 0.1:
        return "positive"
    elif polarity < -0.1:
        return "negative"
    else:
        return "neutral"
