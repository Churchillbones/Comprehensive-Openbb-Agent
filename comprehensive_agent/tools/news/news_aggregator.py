"""
News Aggregator Tool

Aggregates news from multiple sources, deduplicates, and extracts key themes.
NEW tool created for News & Sentiment Agent.
"""

from typing import Dict, Any, List, Optional, Set
import logging
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)


async def aggregate_news(
    news_sources: List[Dict[str, Any]],
    deduplicate: bool = True,
    sort_by: str = "relevance",
    max_results: Optional[int] = None,
    extract_themes: bool = True
) -> Dict[str, Any]:
    """
    Aggregate news from multiple sources

    Args:
        news_sources: List of news data from different sources
            Each source should be a dict with:
                - source_name: Name of the source
                - results: List of news items
                - timestamp: When data was fetched
        deduplicate: Whether to remove duplicate articles
        sort_by: How to sort results ("relevance", "date", "source")
        max_results: Maximum number of results to return
        extract_themes: Whether to extract common themes/topics

    Returns:
        Dict with aggregated news:
            - status: "success" or "error"
            - total_articles: Total number of articles after aggregation
            - sources: Number of sources aggregated
            - articles: Aggregated and sorted articles
            - themes: Extracted themes (if extract_themes=True)
            - metadata: Aggregation metadata
    """
    try:
        if not news_sources:
            return {
                "status": "error",
                "error": "No news sources provided",
                "total_articles": 0,
                "articles": []
            }

        # Aggregate all articles
        all_articles = []
        source_counts = defaultdict(int)
        seen_titles: Set[str] = set()

        for source in news_sources:
            source_name = source.get("source_name", "unknown")
            results = source.get("results", [])

            if isinstance(results, list):
                for article in results:
                    # Normalize article structure
                    normalized = _normalize_article(article, source_name)

                    # Deduplicate by title if enabled
                    if deduplicate:
                        title_lower = normalized.get("title", "").lower().strip()
                        if title_lower and title_lower in seen_titles:
                            continue
                        seen_titles.add(title_lower)

                    all_articles.append(normalized)
                    source_counts[source_name] += 1

        logger.info(f"Aggregated {len(all_articles)} articles from {len(source_counts)} sources")

        # Sort articles
        if sort_by == "date":
            all_articles = _sort_by_date(all_articles)
        elif sort_by == "source":
            all_articles = sorted(all_articles, key=lambda x: x.get("source", ""))
        elif sort_by == "relevance":
            all_articles = _sort_by_relevance(all_articles)
        # Default: keep original order

        # Limit results if specified
        if max_results and max_results > 0:
            all_articles = all_articles[:max_results]

        # Extract themes if enabled
        themes = []
        if extract_themes and all_articles:
            themes = _extract_themes(all_articles)

        # Build response
        result = {
            "status": "success",
            "total_articles": len(all_articles),
            "sources": len(source_counts),
            "articles": all_articles,
            "source_breakdown": dict(source_counts),
            "metadata": {
                "deduplicated": deduplicate,
                "sorted_by": sort_by,
                "timestamp": datetime.now().isoformat()
            }
        }

        if extract_themes:
            result["themes"] = themes
            result["metadata"]["themes_extracted"] = len(themes)

        return result

    except Exception as e:
        logger.error(f"News aggregation failed: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "total_articles": 0,
            "articles": []
        }


def _normalize_article(article: Any, source_name: str) -> Dict[str, Any]:
    """Normalize article from various formats to standard structure"""
    normalized = {
        "source": source_name,
        "fetched_at": datetime.now().isoformat()
    }

    if isinstance(article, dict):
        # Map common fields
        field_mapping = {
            "title": ["title", "headline", "name"],
            "url": ["url", "link", "href"],
            "snippet": ["snippet", "description", "summary", "body", "content"],
            "date": ["date", "published", "published_at", "timestamp", "time"],
            "author": ["author", "source", "publisher"],
            "sentiment": ["sentiment", "sentiment_score", "polarity"]
        }

        for target_field, possible_fields in field_mapping.items():
            for field in possible_fields:
                if field in article and article[field]:
                    normalized[target_field] = article[field]
                    break

    elif isinstance(article, str):
        # Simple text article
        normalized["title"] = article[:100] if len(article) > 100 else article
        normalized["snippet"] = article

    else:
        # Convert to string
        text = str(article)
        normalized["title"] = text[:100] if len(text) > 100 else text

    return normalized


def _sort_by_date(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort articles by date (most recent first)"""
    def get_date_sort_key(article: Dict) -> float:
        # Try to parse date
        date_str = article.get("date", "")
        if not date_str:
            # Use fetched_at as fallback
            date_str = article.get("fetched_at", "")

        if date_str:
            try:
                from dateutil import parser
                dt = parser.parse(str(date_str))
                return dt.timestamp()
            except:
                pass

        # Return 0 for unparseable dates (will be sorted last)
        return 0.0

    return sorted(articles, key=get_date_sort_key, reverse=True)


def _sort_by_relevance(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort articles by relevance (simple scoring)"""
    def calculate_relevance_score(article: Dict) -> float:
        score = 0.0

        # Has sentiment score = more relevant
        if "sentiment" in article:
            score += 2.0

        # Has URL = more credible
        if "url" in article:
            score += 1.0

        # Has snippet = more informative
        if "snippet" in article and len(article.get("snippet", "")) > 100:
            score += 1.5

        # Has author = more credible
        if "author" in article:
            score += 0.5

        # Recent date = more relevant
        if "date" in article:
            score += 1.0

        # Longer title = potentially more detailed
        title_len = len(article.get("title", ""))
        if title_len > 50:
            score += 0.5

        return score

    return sorted(articles, key=calculate_relevance_score, reverse=True)


def _extract_themes(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract common themes from articles"""
    themes = []

    try:
        # Collect all words from titles and snippets
        word_counts = defaultdict(int)
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'should', 'could', 'may', 'might', 'can', 'this', 'that',
            'these', 'those', 'it', 'its', 'they', 'them', 'their'
        }

        for article in articles:
            text = ""
            if "title" in article:
                text += " " + article["title"]
            if "snippet" in article:
                text += " " + article["snippet"]

            # Extract words
            words = text.lower().split()
            for word in words:
                # Clean word
                word = ''.join(c for c in word if c.isalnum())
                # Filter short words and stop words
                if len(word) > 3 and word not in stop_words:
                    word_counts[word] += 1

        # Get top themes (words appearing in multiple articles)
        min_occurrences = max(2, len(articles) // 10)  # At least 10% of articles
        top_themes = [
            {"theme": word, "count": count}
            for word, count in word_counts.items()
            if count >= min_occurrences
        ]

        # Sort by count
        top_themes = sorted(top_themes, key=lambda x: x["count"], reverse=True)

        # Take top 10
        themes = top_themes[:10]

        logger.debug(f"Extracted {len(themes)} themes from {len(articles)} articles")

    except Exception as e:
        logger.warning(f"Theme extraction failed: {e}")

    return themes


async def merge_with_sentiment(
    aggregated_news: Dict[str, Any],
    sentiment_results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Merge aggregated news with sentiment analysis results

    Args:
        aggregated_news: Results from aggregate_news()
        sentiment_results: Results from sentiment_analyzer.analyze_sentiment()

    Returns:
        Merged results with sentiment added to each article
    """
    try:
        if aggregated_news.get("status") != "success":
            return aggregated_news

        # Get overall sentiment
        overall_sentiment = sentiment_results.get("sentiment_score", 0)
        sentiment_label = sentiment_results.get("sentiment_label", "neutral")

        # Add to metadata
        aggregated_news["metadata"]["overall_sentiment"] = overall_sentiment
        aggregated_news["metadata"]["sentiment_label"] = sentiment_label

        # Add sentiment to individual articles if available
        if "details" in sentiment_results and sentiment_results["details"]:
            articles = aggregated_news.get("articles", [])
            sentiment_details = sentiment_results["details"]

            # Match articles with sentiment (by index)
            for i, article in enumerate(articles):
                if i < len(sentiment_details):
                    detail = sentiment_details[i]
                    if "polarity" in detail:
                        article["sentiment_polarity"] = detail["polarity"]
                    if "subjectivity" in detail:
                        article["sentiment_subjectivity"] = detail["subjectivity"]

        return aggregated_news

    except Exception as e:
        logger.error(f"Merging with sentiment failed: {e}")
        return aggregated_news
