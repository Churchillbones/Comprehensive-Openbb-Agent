"""
News & Sentiment Tools

Tools for the News & Sentiment Agent:
1. Web Searcher - General web search
2. Financial News Searcher - Financial-specific news search
3. Citation Generator - Generate source citations
4. PDF Processor - Extract text from PDFs
5. Alert Generator - Generate market alerts
6. Sentiment Analyzer - Analyze text sentiment
7. News Aggregator - Aggregate news from multiple sources
"""

from .web_searcher import search_web
from .financial_news_searcher import search_financial_news
from .citation_generator import generate_citations
from .pdf_processor import process_pdf
from .alert_generator import generate_alert
from .sentiment_analyzer import analyze_sentiment
from .news_aggregator import aggregate_news

__all__ = [
    "search_web",
    "search_financial_news",
    "generate_citations",
    "process_pdf",
    "generate_alert",
    "analyze_sentiment",
    "aggregate_news"
]
