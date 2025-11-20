"""
Sentiment Analyzer Tool

Analyzes sentiment in text data using TextBlob and keyword analysis.
Enhanced version of ml_widget_bridge._analyze_news_data with more advanced features.
"""

from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


async def analyze_sentiment(
    text_data: Any,
    use_textblob: bool = True,
    include_subjectivity: bool = True
) -> Dict[str, Any]:
    """
    Analyze sentiment in text data

    Args:
        text_data: Text to analyze (can be string, list of strings, or dict with text fields)
        use_textblob: Whether to use TextBlob for sentiment (requires textblob package)
        include_subjectivity: Whether to include subjectivity scores

    Returns:
        Dict with sentiment analysis:
            - status: "success" or "error"
            - sentiment_score: Overall sentiment (-1 to 1, negative to positive)
            - sentiment_label: "positive", "negative", or "neutral"
            - polarity: TextBlob polarity score (if use_textblob=True)
            - subjectivity: TextBlob subjectivity score (if enabled)
            - keyword_sentiment: Keyword-based sentiment
            - analyzed_items: Number of items analyzed
            - details: Per-item sentiment scores if multiple items
    """
    try:
        # Extract text items from input
        text_items = _extract_text_items(text_data)

        if not text_items:
            return {
                "status": "error",
                "error": "No text data found to analyze",
                "sentiment_score": 0.0,
                "sentiment_label": "neutral"
            }

        results = {
            "status": "success",
            "analyzed_items": len(text_items),
            "details": []
        }

        # Analyze each text item
        all_polarities = []
        all_subjectivities = []
        all_keyword_scores = []

        for text in text_items:
            item_result = {}

            # Keyword-based sentiment (always calculated)
            keyword_score = _analyze_keywords(text)
            all_keyword_scores.append(keyword_score)
            item_result["keyword_sentiment"] = keyword_score

            # TextBlob sentiment (if available)
            if use_textblob:
                try:
                    from textblob import TextBlob
                    blob = TextBlob(text)
                    polarity = blob.sentiment.polarity  # -1 to 1
                    subjectivity = blob.sentiment.subjectivity  # 0 to 1

                    all_polarities.append(polarity)
                    if include_subjectivity:
                        all_subjectivities.append(subjectivity)

                    item_result["polarity"] = round(polarity, 4)
                    if include_subjectivity:
                        item_result["subjectivity"] = round(subjectivity, 4)

                except ImportError:
                    logger.warning("TextBlob not available, using keyword-based sentiment only")
                    use_textblob = False
                except Exception as e:
                    logger.warning(f"TextBlob analysis failed: {e}")

            results["details"].append(item_result)

        # Calculate aggregate scores
        if all_polarities:
            import numpy as np
            avg_polarity = float(np.mean(all_polarities))
            results["polarity"] = round(avg_polarity, 4)
            results["sentiment_score"] = avg_polarity

            if include_subjectivity and all_subjectivities:
                results["subjectivity"] = round(float(np.mean(all_subjectivities)), 4)
        else:
            # Use keyword-based sentiment as fallback
            import numpy as np
            avg_keyword = float(np.mean(all_keyword_scores))
            # Normalize keyword score to -1 to 1 range
            results["sentiment_score"] = max(-1.0, min(1.0, avg_keyword / 2.0))

        # Calculate average keyword sentiment
        import numpy as np
        results["keyword_sentiment"] = round(float(np.mean(all_keyword_scores)), 4)

        # Determine sentiment label
        sentiment_score = results["sentiment_score"]
        if sentiment_score >= 0.1:
            results["sentiment_label"] = "positive"
            results["emoji"] = "📈"
        elif sentiment_score <= -0.1:
            results["sentiment_label"] = "negative"
            results["emoji"] = "📉"
        else:
            results["sentiment_label"] = "neutral"
            results["emoji"] = "➡️"

        # Generate insights
        results["insights"] = _generate_sentiment_insights(results)

        logger.info(
            f"Sentiment analysis complete: {len(text_items)} items, "
            f"score={sentiment_score:.2f}, label={results['sentiment_label']}"
        )

        return results

    except Exception as e:
        logger.error(f"Sentiment analysis failed: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "sentiment_score": 0.0,
            "sentiment_label": "neutral"
        }


def _extract_text_items(text_data: Any) -> List[str]:
    """Extract text items from various data formats"""
    items = []

    if isinstance(text_data, str):
        # Single text string
        items.append(text_data)

    elif isinstance(text_data, list):
        # List of items
        for item in text_data:
            if isinstance(item, str):
                items.append(item)
            elif isinstance(item, dict):
                # Extract text from dict
                for key in ["text", "content", "title", "description", "summary", "body"]:
                    if key in item and isinstance(item[key], str):
                        items.append(item[key])
                        break
            else:
                # Convert to string
                items.append(str(item))

    elif isinstance(text_data, dict):
        # Dictionary with text fields
        for key in ["text", "content", "title", "description", "summary", "body", "message"]:
            if key in text_data and isinstance(text_data[key], str):
                items.append(text_data[key])

        # If no text fields found, check for nested data
        if not items:
            if "data" in text_data and isinstance(text_data["data"], (list, dict)):
                return _extract_text_items(text_data["data"])
            elif "results" in text_data and isinstance(text_data["results"], list):
                return _extract_text_items(text_data["results"])

    else:
        # Convert to string as fallback
        items.append(str(text_data))

    return [item for item in items if item and len(item.strip()) > 0]


def _analyze_keywords(text: str) -> float:
    """
    Analyze sentiment using keyword matching

    Returns:
        Sentiment score (positive values = positive sentiment, negative = negative)
    """
    text_lower = text.lower()

    # Enhanced keyword lists from ml_widget_bridge
    positive_keywords = [
        'surge', 'gain', 'up', 'growth', 'profit', 'beat', 'strong', 'success',
        'bullish', 'rally', 'soar', 'jump', 'rise', 'advance', 'outperform',
        'boost', 'improve', 'positive', 'optimistic', 'upgrade', 'buy', 'exceed'
    ]

    negative_keywords = [
        'fall', 'drop', 'down', 'loss', 'miss', 'weak', 'decline', 'concern',
        'bearish', 'plunge', 'tumble', 'slump', 'crash', 'underperform',
        'cut', 'worsen', 'negative', 'pessimistic', 'downgrade', 'sell', 'fail'
    ]

    # Count occurrences
    pos_count = sum(1 for kw in positive_keywords if kw in text_lower)
    neg_count = sum(1 for kw in negative_keywords if kw in text_lower)

    # Calculate score
    score = pos_count - neg_count

    return float(score)


def _generate_sentiment_insights(results: Dict[str, Any]) -> List[str]:
    """Generate human-readable insights from sentiment analysis"""
    insights = []

    sentiment_score = results.get("sentiment_score", 0)
    sentiment_label = results.get("sentiment_label", "neutral")
    analyzed_items = results.get("analyzed_items", 0)

    # Overall sentiment insight
    if sentiment_label == "positive":
        if sentiment_score > 0.5:
            insights.append(f"{results.get('emoji', '📈')} Strongly positive sentiment detected")
        else:
            insights.append(f"{results.get('emoji', '📈')} Positive sentiment detected")
    elif sentiment_label == "negative":
        if sentiment_score < -0.5:
            insights.append(f"{results.get('emoji', '📉')} Strongly negative sentiment detected")
        else:
            insights.append(f"{results.get('emoji', '📉')} Negative sentiment detected")
    else:
        insights.append(f"{results.get('emoji', '➡️')} Sentiment is neutral or mixed")

    # Subjectivity insight
    if "subjectivity" in results:
        subjectivity = results["subjectivity"]
        if subjectivity > 0.7:
            insights.append("💭 Content is highly subjective/opinion-based")
        elif subjectivity < 0.3:
            insights.append("📊 Content is mostly factual/objective")

    # Volume insight
    if analyzed_items > 10:
        insights.append(f"📢 Analysis based on {analyzed_items} text items")

    # Mixed sentiment insight
    if "details" in results and len(results["details"]) > 3:
        polarities = [d.get("polarity", 0) for d in results["details"] if "polarity" in d]
        if polarities:
            import numpy as np
            std = np.std(polarities)
            if std > 0.3:
                insights.append("⚖️ Sentiment varies significantly across items - mixed signals")

    return insights
