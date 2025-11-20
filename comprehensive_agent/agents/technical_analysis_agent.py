"""
Technical Analysis Agent

Performs technical analysis with charts, indicators, and pattern recognition.
Part of Phase 2b-1 in the multi-agent orchestration architecture.
"""

from typing import Dict, Any, List, Optional
import logging
from comprehensive_agent.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class TechnicalAnalysisAgent(BaseAgent):
    """
    Specialized agent for technical analysis

    Capabilities:
    - Chart generation (line, candlestick, OHLC, bar)
    - Technical indicator calculation (RSI, MACD, Bollinger Bands, etc.)
    - Trend detection and analysis
    - Support/resistance level identification
    - Volume analysis
    - Pattern recognition
    - Interactive chart features
    """

    def __init__(self):
        super().__init__(
            name="Technical Analysis Agent",
            description="Perform technical analysis with charts, indicators, and pattern recognition",
            capabilities=[
                "chart_generation",
                "technical_indicators",
                "trend_analysis",
                "support_resistance",
                "volume_analysis",
                "pattern_recognition",
                "interactive_charts"
            ],
            priority=2
        )
        self._register_tools()

    def _register_tools(self):
        """Register all 7 technical analysis tools"""
        from comprehensive_agent.tools.technical_analysis import (
            generate_chart,
            generate_financial_chart,
            build_interactive_chart,
            calculate_technical_indicator,
            detect_trend,
            find_support_resistance,
            analyze_volume
        )

        self.register_tool("generate_chart", generate_chart)
        self.register_tool("generate_financial_chart", generate_financial_chart)
        self.register_tool("build_interactive_chart", build_interactive_chart)
        self.register_tool("calculate_technical_indicator", calculate_technical_indicator)
        self.register_tool("detect_trend", detect_trend)
        self.register_tool("find_support_resistance", find_support_resistance)
        self.register_tool("analyze_volume", analyze_volume)

        logger.info(f"Technical Analysis Agent initialized with {len(self.tools)} tools")

    def can_handle(self, intent: str, context: Dict[str, Any]) -> float:
        """
        Determine if this agent can handle the request

        Args:
            intent: The classified intent type
            context: Request context with widget_data, query, etc.

        Returns:
            Confidence score (0.0 to 1.0)
        """
        confidence = 0.0

        # Check intent type
        if intent == "technical_analysis":
            confidence = 0.95

        # Check for chart-related keywords
        query = context.get("query", "").lower()
        chart_keywords = [
            "chart", "plot", "graph", "candlestick", "ohlc", "visualize",
            "show", "display", "draw"
        ]
        if any(keyword in query for keyword in chart_keywords):
            confidence = max(confidence, 0.85)

        # Check for technical indicator keywords
        indicator_keywords = [
            "rsi", "macd", "bollinger", "moving average", "ma", "ema", "sma",
            "indicator", "technical", "momentum", "oscillator", "stochastic"
        ]
        if any(keyword in query for keyword in indicator_keywords):
            confidence = max(confidence, 0.90)

        # Check for trend keywords
        trend_keywords = ["trend", "uptrend", "downtrend", "reversal", "breakout", "breakdown"]
        if any(keyword in query for keyword in trend_keywords):
            confidence = max(confidence, 0.85)

        # Check for support/resistance keywords
        level_keywords = ["support", "resistance", "level", "barrier", "floor", "ceiling"]
        if any(keyword in query for keyword in level_keywords):
            confidence = max(confidence, 0.80)

        # Check for volume keywords
        volume_keywords = ["volume", "vwap", "volume profile", "volume spike"]
        if any(keyword in query for keyword in volume_keywords):
            confidence = max(confidence, 0.80)

        # Check for OHLCV data in context
        if context.get("has_ohlcv_data"):
            confidence = max(confidence, 0.70)

        # Check for price data
        if context.get("has_price_data"):
            confidence = max(confidence, 0.65)

        # Widget type checks
        widget_type = context.get("widget_type", "")
        if widget_type in ["price_chart", "technical_indicator", "candlestick"]:
            confidence = max(confidence, 0.80)

        return confidence

    async def process(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process technical analysis request

        Args:
            query: User query
            context: Request context

        Returns:
            Processing results with charts, indicators, and insights
        """
        try:
            logger.info(f"Technical Analysis Agent processing: {query[:100]}")

            results = {
                "agent": self.name,
                "query": query,
                "status": "success",
                "charts": [],
                "indicators": {},
                "analysis": {},
                "insights": []
            }

            # Extract price data from context
            price_data = context.get("price_data")
            ohlcv_data = context.get("ohlcv_data")
            widget_data = context.get("widget_data")

            # Determine what analysis to perform based on query
            query_lower = query.lower()

            # 1. Chart generation (if requested or needed)
            if any(kw in query_lower for kw in ["chart", "plot", "graph", "show", "visualize"]):
                if "candlestick" in query_lower or "candle" in query_lower:
                    chart_result = await self.tools["generate_financial_chart"](
                        data=ohlcv_data or price_data,
                        chart_type="candlestick"
                    )
                    if chart_result.get("status") == "success":
                        results["charts"].append(chart_result)

                elif "ohlc" in query_lower:
                    chart_result = await self.tools["generate_financial_chart"](
                        data=ohlcv_data or price_data,
                        chart_type="ohlc"
                    )
                    if chart_result.get("status") == "success":
                        results["charts"].append(chart_result)

                else:
                    # Default line chart
                    chart_result = await self.tools["generate_chart"](
                        data=price_data or ohlcv_data,
                        chart_type="line"
                    )
                    if chart_result.get("status") == "success":
                        results["charts"].append(chart_result)

            # 2. Technical indicators (if requested)
            indicator_requests = []
            if "rsi" in query_lower:
                indicator_requests.append("rsi")
            if "macd" in query_lower:
                indicator_requests.append("macd")
            if "bollinger" in query_lower:
                indicator_requests.append("bollinger_bands")
            if "moving average" in query_lower or "ma" in query_lower:
                indicator_requests.append("sma")
            if "ema" in query_lower:
                indicator_requests.append("ema")

            for indicator in indicator_requests:
                indicator_result = await self.tools["calculate_technical_indicator"](
                    data=price_data or ohlcv_data,
                    indicator_type=indicator
                )
                if indicator_result.get("status") == "success":
                    results["indicators"][indicator] = indicator_result

            # 3. Trend analysis (always perform if we have price data)
            if price_data or ohlcv_data:
                trend_result = await self.tools["detect_trend"](
                    data=price_data or ohlcv_data
                )
                if trend_result.get("status") == "success":
                    results["analysis"]["trend"] = trend_result
                    results["insights"].extend(trend_result.get("insights", []))

            # 4. Support/Resistance (if requested or useful)
            if any(kw in query_lower for kw in ["support", "resistance", "level", "breakout"]):
                sr_result = await self.tools["find_support_resistance"](
                    data=price_data or ohlcv_data
                )
                if sr_result.get("status") == "success":
                    results["analysis"]["support_resistance"] = sr_result
                    results["insights"].extend(sr_result.get("insights", []))

            # 5. Volume analysis (if requested or if volume data available)
            if "volume" in query_lower or ohlcv_data:
                volume_result = await self.tools["analyze_volume"](
                    data=ohlcv_data or price_data
                )
                if volume_result.get("status") == "success":
                    results["analysis"]["volume"] = volume_result
                    results["insights"].extend(volume_result.get("insights", []))

            # Generate summary
            results["summary"] = self._generate_summary(results)

            logger.info(f"Technical Analysis complete: {len(results['charts'])} charts, "
                       f"{len(results['indicators'])} indicators, {len(results['insights'])} insights")

            return results

        except Exception as e:
            logger.error(f"Technical Analysis Agent error: {e}", exc_info=True)
            return {
                "agent": self.name,
                "status": "error",
                "error": str(e),
                "charts": [],
                "indicators": {},
                "insights": []
            }

    def _generate_summary(self, results: Dict[str, Any]) -> str:
        """Generate a summary of the technical analysis"""
        parts = []

        if results.get("charts"):
            parts.append(f"Generated {len(results['charts'])} chart(s)")

        if results.get("indicators"):
            parts.append(f"Calculated {len(results['indicators'])} indicator(s)")

        if results.get("analysis", {}).get("trend"):
            trend_data = results["analysis"]["trend"]
            if "trend_direction" in trend_data:
                parts.append(f"Trend: {trend_data['trend_direction']}")

        if results.get("insights"):
            parts.append(f"{len(results['insights'])} key insights")

        return " | ".join(parts) if parts else "Technical analysis completed"
