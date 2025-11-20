"""
Market Data Agent for OpenBB Comprehensive Agent

This agent specializes in retrieving and processing market data from:
- OpenBB workspace widgets
- External APIs
- Real-time data streams

Capabilities:
- Widget data extraction and parsing
- API data fetching and normalization
- Data quality validation
- OHLCV data processing
- Multi-source data integration
"""

from typing import Dict, List, Any, Optional
import logging
from comprehensive_agent.agents.base_agent import BaseAgent, AgentState
from comprehensive_agent.config import AGENT_METADATA

logger = logging.getLogger(__name__)


class MarketDataAgent(BaseAgent):
    """
    Agent specialized in market data retrieval and processing

    Tools:
    1. Widget Processor - Extract data from OpenBB widgets
    2. Widget Intelligence - Analyze widget context and type
    3. API Data Fetcher - Fetch data from external APIs
    4. API Data Processor - Advanced normalization and merging
    5. Data Validator - Validate data quality and integrity
    6. OHLCV Extractor - Extract OHLCV (Open/High/Low/Close/Volume) data
    7. Stream Manager - Manage real-time data streams
    """

    def __init__(self):
        """Initialize the Market Data Agent"""
        metadata = AGENT_METADATA.get("MarketDataAgent", {})

        super().__init__(
            name=metadata.get("name", "Market Data Agent"),
            description=metadata.get("description", "Market data retrieval and processing"),
            capabilities=metadata.get("capabilities", [
                "widget_data_extraction",
                "price_data_retrieval",
                "data_validation",
                "real_time_data",
                "historical_data",
                "ohlcv_processing",
                "api_data_integration"
            ]),
            priority=metadata.get("priority", 1)
        )

        self._register_tools()
        logger.info(f"Initialized {self.name} with {len(self.tools)} tools")

    def _register_tools(self):
        """Register all tools for market data processing"""
        # Import tools locally to avoid circular dependencies
        from comprehensive_agent.tools.market_data import (
            process_widget_data,
            analyze_widget_intelligence,
            fetch_api_data,
            process_api_data_advanced,
            validate_data,
            extract_ohlcv,
            manage_stream
        )

        # Register each tool
        self.register_tool("process_widget_data", process_widget_data)
        self.register_tool("analyze_widget_intelligence", analyze_widget_intelligence)
        self.register_tool("fetch_api_data", fetch_api_data)
        self.register_tool("process_api_data_advanced", process_api_data_advanced)
        self.register_tool("validate_data", validate_data)
        self.register_tool("extract_ohlcv", extract_ohlcv)
        self.register_tool("manage_stream", manage_stream)

        logger.debug(f"Registered {len(self.tools)} tools: {list(self.tools.keys())}")

    def can_handle(self, intent: str, context: Dict[str, Any]) -> float:
        """
        Determine if this agent can handle the given intent

        Args:
            intent: The classified intent type
            context: Request context with query, widget_data, etc.

        Returns:
            Confidence score (0.0 to 1.0)
        """
        confidence = 0.0

        # High confidence for market_data intent
        if intent == "market_data":
            confidence = 0.95

        # Check context for widget data
        if context.get("widget_data"):
            confidence = max(confidence, 0.9)

        # Check for price-related queries in the query string
        query = context.get("query", "").lower()
        price_keywords = ["price", "quote", "stock", "ticker", "market data", "ohlc", "candle"]

        if any(keyword in query for keyword in price_keywords):
            confidence = max(confidence, 0.8)

        # Check for symbols in context
        if context.get("symbols") or context.get("symbol"):
            confidence = max(confidence, 0.75)

        # API data requests
        if context.get("api_data") or "api" in query:
            confidence = max(confidence, 0.7)

        logger.debug(f"{self.name} confidence for intent '{intent}': {confidence}")
        return confidence

    async def process(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process market data request using registered tools

        Args:
            query: User query string
            context: Request context with additional data

        Returns:
            Dict containing:
                - status: "success" or "error"
                - data: Processed market data
                - insights: Generated insights
                - metadata: Processing metadata
        """
        try:
            self.set_state(AgentState.PROCESSING)
            tools_used = []
            processed_data = {}
            insights = []

            # Step 1: Check for widget data
            widget_data = context.get("widget_data")
            if widget_data:
                logger.info("Processing widget data")

                # Analyze widget intelligence
                widget_analysis = await self.tools["analyze_widget_intelligence"](widget_data)
                tools_used.append("analyze_widget_intelligence")

                # Process widget data
                widget_processed = await self.tools["process_widget_data"](widget_data)
                tools_used.append("process_widget_data")

                processed_data["widget"] = widget_processed
                processed_data["widget_analysis"] = widget_analysis

                # Extract OHLCV if applicable
                if widget_analysis.get("has_price_data"):
                    ohlcv_data = await self.tools["extract_ohlcv"](widget_processed)
                    tools_used.append("extract_ohlcv")
                    processed_data["ohlcv"] = ohlcv_data

                    insights.append(
                        f"Extracted OHLCV data with {len(ohlcv_data.get('data', []))} periods"
                    )

            # Step 2: Check for API data requests
            if context.get("api_endpoint") or context.get("api_data"):
                logger.info("Processing API data request")

                api_data = await self.tools["fetch_api_data"](
                    endpoint=context.get("api_endpoint"),
                    params=context.get("api_params", {})
                )
                tools_used.append("fetch_api_data")

                # Advanced processing if needed
                if api_data.get("status") == "success":
                    processed_api = await self.tools["process_api_data_advanced"](api_data)
                    tools_used.append("process_api_data_advanced")
                    processed_data["api"] = processed_api

            # Step 3: Validate all processed data
            if processed_data:
                validation = await self.tools["validate_data"](processed_data)
                tools_used.append("validate_data")

                if validation.get("status") == "valid":
                    insights.append("Data validation passed")
                else:
                    insights.append(f"Data validation issues: {validation.get('issues', [])}")

            # Step 4: Generate summary insights
            if processed_data.get("widget_analysis"):
                widget_type = processed_data["widget_analysis"].get("widget_type", "unknown")
                insights.append(f"Processed {widget_type} widget data")

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
        return f"<MarketDataAgent state={self.state.value} tools={len(self.tools)}>"
