"""
News & Sentiment Agent for OpenBB Comprehensive Agent

This agent specializes in news aggregation, sentiment analysis, and information retrieval:
- Web search (general and financial)
- Sentiment analysis
- PDF document processing
- Citation generation
- Market alerts

Capabilities:
- News aggregation and filtering
- Sentiment scoring
- Source citation
- Document extraction
- Market event alerts
"""

from typing import Dict, List, Any, Optional
import logging
from comprehensive_agent.agents.base_agent import BaseAgent, AgentState
from comprehensive_agent.config import AGENT_METADATA

logger = logging.getLogger(__name__)


class NewsSentimentAgent(BaseAgent):
    """
    Agent specialized in news aggregation and sentiment analysis

    Tools:
    1. Web Searcher - General web search capability
    2. Financial News Searcher - Financial-specific news search
    3. Citation Generator - Generate proper citations
    4. PDF Processor - Extract text from PDF documents
    5. Alert Generator - Create market alerts
    6. Sentiment Analyzer - Analyze sentiment from text
    7. News Aggregator - Aggregate news from multiple sources
    """

    def __init__(self):
        """Initialize the News & Sentiment Agent"""
        metadata = AGENT_METADATA.get("NewsSentimentAgent", {})

        super().__init__(
            name=metadata.get("name", "News & Sentiment Agent"),
            description=metadata.get("description", "News aggregation and sentiment analysis"),
            capabilities=metadata.get("capabilities", [
                "web_search",
                "financial_news_search",
                "sentiment_analysis",
                "news_aggregation",
                "citation_generation",
                "pdf_processing",
                "market_alerts"
            ]),
            priority=metadata.get("priority", 2)
        )

        self._register_tools()
        logger.info(f"Initialized {self.name} with {len(self.tools)} tools")

    def _register_tools(self):
        """Register all tools for news and sentiment processing"""
        # Import tools locally to avoid circular dependencies
        from comprehensive_agent.tools.news import (
            search_web,
            search_financial_news,
            generate_citations,
            process_pdf,
            generate_alert,
            analyze_sentiment,
            aggregate_news
        )

        # Register each tool
        self.register_tool("search_web", search_web)
        self.register_tool("search_financial_news", search_financial_news)
        self.register_tool("generate_citations", generate_citations)
        self.register_tool("process_pdf", process_pdf)
        self.register_tool("generate_alert", generate_alert)
        self.register_tool("analyze_sentiment", analyze_sentiment)
        self.register_tool("aggregate_news", aggregate_news)

        logger.debug(f"Registered {len(self.tools)} tools: {list(self.tools.keys())}")

    def can_handle(self, intent: str, context: Dict[str, Any]) -> float:
        """
        Determine if this agent can handle the given intent

        Args:
            intent: The classified intent type
            context: Request context with query, files, etc.

        Returns:
            Confidence score (0.0 to 1.0)
        """
        confidence = 0.0

        # High confidence for news_sentiment intent
        if intent == "news_sentiment":
            confidence = 0.95

        # Check query for news-related keywords
        query = context.get("query", "").lower()
        news_keywords = [
            "news", "article", "headline", "search", "sentiment",
            "latest", "recent", "announcement", "press release"
        ]

        if any(keyword in query for keyword in news_keywords):
            confidence = max(confidence, 0.85)

        # Check for uploaded PDF files
        uploaded_files = context.get("uploaded_files", [])
        if any(f.get("file_type") == "pdf" for f in uploaded_files):
            confidence = max(confidence, 0.8)

        # Check for web search request
        if context.get("web_search_requested") or "search" in query:
            confidence = max(confidence, 0.75)

        # Check for sentiment-related keywords
        sentiment_keywords = ["sentiment", "opinion", "feeling", "positive", "negative"]
        if any(keyword in query for keyword in sentiment_keywords):
            confidence = max(confidence, 0.8)

        logger.debug(f"{self.name} confidence for intent '{intent}': {confidence}")
        return confidence

    async def process(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process news and sentiment request using registered tools

        Args:
            query: User query string
            context: Request context with additional data

        Returns:
            Dict containing:
                - status: "success" or "error"
                - data: News articles, sentiment scores, etc.
                - insights: Generated insights
                - metadata: Processing metadata
        """
        try:
            self.set_state(AgentState.PROCESSING)
            tools_used = []
            processed_data = {}
            insights = []

            # Step 1: Check for web search requests
            if context.get("web_search_requested") or "search" in query.lower():
                logger.info("Processing web search request")

                # Determine if financial news or general search
                is_financial = any(word in query.lower() for word in [
                    "stock", "market", "earnings", "financial", "trading", "price"
                ])

                if is_financial:
                    # Use financial news searcher
                    news_results = await self.tools["search_financial_news"](query)
                    tools_used.append("search_financial_news")
                else:
                    # Use general web searcher
                    search_results = await self.tools["search_web"](query)
                    tools_used.append("search_web")
                    news_results = search_results

                processed_data["search_results"] = news_results

                # Aggregate news if multiple results
                if news_results.get("status") == "success":
                    aggregated = await self.tools["aggregate_news"](news_results.get("results", []))
                    tools_used.append("aggregate_news")
                    processed_data["aggregated_news"] = aggregated

                    insights.append(
                        f"Found {len(news_results.get('results', []))} news articles"
                    )

            # Step 2: Process PDF files if uploaded
            uploaded_files = context.get("uploaded_files", [])
            pdf_files = [f for f in uploaded_files if f.get("file_type") == "pdf"]

            if pdf_files:
                logger.info(f"Processing {len(pdf_files)} PDF files")

                pdf_content = []
                for pdf_file in pdf_files:
                    result = await self.tools["process_pdf"](pdf_file.get("content"))
                    tools_used.append("process_pdf")
                    pdf_content.append(result)

                processed_data["pdf_content"] = pdf_content
                insights.append(f"Extracted text from {len(pdf_files)} PDF document(s)")

            # Step 3: Perform sentiment analysis on collected text
            text_for_sentiment = []

            # Collect text from news results
            if "search_results" in processed_data:
                results = processed_data["search_results"].get("results", [])
                for article in results:
                    if isinstance(article, dict):
                        title = article.get("title", "")
                        snippet = article.get("snippet", "")
                        text_for_sentiment.append(f"{title}. {snippet}")

            # Collect text from PDFs
            if "pdf_content" in processed_data:
                for pdf_result in processed_data["pdf_content"]:
                    if pdf_result.get("status") == "success":
                        text_for_sentiment.append(pdf_result.get("text", ""))

            # Analyze sentiment if we have text
            if text_for_sentiment:
                logger.info("Analyzing sentiment")

                combined_text = " ".join(text_for_sentiment)
                sentiment_result = await self.tools["analyze_sentiment"](combined_text)
                tools_used.append("analyze_sentiment")

                processed_data["sentiment"] = sentiment_result

                if sentiment_result.get("status") == "success":
                    polarity = sentiment_result.get("polarity", 0)
                    sentiment_label = sentiment_result.get("label", "neutral")

                    insights.append(
                        f"Overall sentiment: {sentiment_label} (polarity: {polarity:.2f})"
                    )

            # Step 4: Generate citations if we have sources
            sources = []
            if "search_results" in processed_data:
                results = processed_data["search_results"].get("results", [])
                for article in results[:5]:  # Top 5 sources
                    if isinstance(article, dict):
                        sources.append({
                            "title": article.get("title"),
                            "url": article.get("url"),
                            "source": article.get("source")
                        })

            if sources:
                citations = await self.tools["generate_citations"](sources)
                tools_used.append("generate_citations")
                processed_data["citations"] = citations

            # Step 5: Generate alert if conditions met
            # (This would be based on sentiment thresholds, key events, etc.)
            if processed_data.get("sentiment"):
                sentiment = processed_data["sentiment"]
                polarity = sentiment.get("polarity", 0)

                # Generate alert for highly positive or negative sentiment
                if abs(polarity) > 0.5:
                    alert = await self.tools["generate_alert"](
                        f"High {'positive' if polarity > 0 else 'negative'} sentiment detected",
                        {"polarity": polarity, "query": query}
                    )
                    tools_used.append("generate_alert")
                    processed_data["alert"] = alert

            # Update metrics
            self.update_metrics(success=True)
            self.set_state(AgentState.READY)

            return {
                "status": "success",
                "data": processed_data,
                "insights": insights,
                "metadata": {
                    "agent": self.name,
                    "tools_used": tools_used,
                    "data_sources": list(processed_data.keys())
                }
            }

        except Exception as e:
            logger.error(f"Error in {self.name}: {e}", exc_info=True)
            self.update_metrics(success=False)
            self.set_state(AgentState.ERROR)

            return {
                "status": "error",
                "error": str(e),
                "metadata": {
                    "agent": self.name,
                    "error_type": type(e).__name__
                }
            }

    def __repr__(self) -> str:
        """String representation of the agent"""
        return f"<NewsSentimentAgent state={self.state.value} tools={len(self.tools)}>"
