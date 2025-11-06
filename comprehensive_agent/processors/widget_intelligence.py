"""
Widget Intelligence Processor

Automatically detects widget types, extracts context, and provides
smart analysis suggestions based on the widget data.
"""

from typing import Any, Dict, List, Optional, Tuple
import json
import logging
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class WidgetContext:
    """Extracted context from a widget."""
    widget_id: str
    widget_name: str
    detected_type: str
    symbols: List[str]
    timeframe: Optional[str]
    data_points: int
    metrics: List[str]
    category: Optional[str]
    analysis_suggestions: List[str]


class WidgetIntelligence:
    """
    Intelligent widget analysis and context extraction.

    This processor automatically:
    - Detects widget types (price, financials, news, economic, etc.)
    - Extracts key information (symbols, timeframes, metrics)
    - Suggests relevant analyses
    - Identifies relationships between widgets
    """

    # Widget type patterns
    WIDGET_TYPE_PATTERNS = {
        "price_chart": ["price", "historical", "chart", "ohlc", "candlestick", "stock_price"],
        "financial_statement": ["income", "balance", "cash_flow", "financials", "earnings"],
        "news": ["news", "article", "headline", "press"],
        "economic_indicator": ["gdp", "inflation", "unemployment", "cpi", "economic", "fed"],
        "technical_indicator": ["rsi", "macd", "bollinger", "moving_average", "technical"],
        "fundamental_ratio": ["pe", "pb", "roe", "debt", "ratio", "valuation"],
        "portfolio": ["portfolio", "holdings", "allocation", "position"],
        "sector": ["sector", "industry", "etf"],
        "options": ["option", "call", "put", "strike", "expiry"],
        "crypto": ["crypto", "bitcoin", "ethereum", "btc", "eth"],
    }

    def __init__(self):
        self.widget_cache: Dict[str, WidgetContext] = {}

    async def analyze_widget(self, widget: Any, widget_data: Any = None) -> WidgetContext:
        """
        Analyze a widget and extract context.

        Args:
            widget: The widget object from OpenBB
            widget_data: Optional widget data content

        Returns:
            WidgetContext with extracted information
        """
        widget_id = getattr(widget, 'uuid', str(widget))
        widget_name = getattr(widget, 'name', 'Unknown Widget')

        # Check cache
        if widget_id in self.widget_cache:
            return self.widget_cache[widget_id]

        # Detect widget type
        detected_type = self._detect_widget_type(widget, widget_data)

        # Extract symbols
        symbols = self._extract_symbols(widget, widget_data)

        # Detect timeframe
        timeframe = self._detect_timeframe(widget, widget_data)

        # Count data points
        data_points = self._count_data_points(widget_data)

        # Extract metrics
        metrics = self._extract_metrics(widget, widget_data)

        # Get category
        category = getattr(widget, 'category', None)

        # Generate analysis suggestions
        suggestions = self._generate_suggestions(detected_type, symbols, metrics, timeframe)

        context = WidgetContext(
            widget_id=widget_id,
            widget_name=widget_name,
            detected_type=detected_type,
            symbols=symbols,
            timeframe=timeframe,
            data_points=data_points,
            metrics=metrics,
            category=category,
            analysis_suggestions=suggestions,
        )

        # Cache the context
        self.widget_cache[widget_id] = context

        logger.info(f"Analyzed widget '{widget_name}': type={detected_type}, symbols={symbols}")

        return context

    def _detect_widget_type(self, widget: Any, widget_data: Any) -> str:
        """Detect the type of widget based on metadata and data."""
        widget_name = getattr(widget, 'name', '').lower()
        widget_id = getattr(widget, 'widget_id', '').lower()
        description = getattr(widget, 'description', '').lower()

        # Combine text for pattern matching
        text = f"{widget_name} {widget_id} {description}"

        # Check against patterns
        type_scores = {}
        for widget_type, patterns in self.WIDGET_TYPE_PATTERNS.items():
            score = sum(1 for pattern in patterns if pattern in text)
            if score > 0:
                type_scores[widget_type] = score

        if type_scores:
            return max(type_scores, key=type_scores.get)

        # Check data structure for hints
        if widget_data:
            data_str = str(widget_data).lower()
            if any(term in data_str for term in ["open", "high", "low", "close"]):
                return "price_chart"
            elif any(term in data_str for term in ["revenue", "assets", "liabilities"]):
                return "financial_statement"

        return "generic"

    def _extract_symbols(self, widget: Any, widget_data: Any) -> List[str]:
        """Extract stock symbols or tickers from widget."""
        symbols = []

        # Check widget params
        if hasattr(widget, 'params'):
            for param in widget.params:
                if hasattr(param, 'name') and param.name.lower() in ['symbol', 'ticker', 'symbols']:
                    value = getattr(param, 'current_value', None)
                    if value:
                        if isinstance(value, list):
                            symbols.extend(value)
                        else:
                            symbols.append(str(value))

        # Check widget name for symbols (e.g., "AAPL Price Chart")
        widget_name = getattr(widget, 'name', '')
        # Simple heuristic: uppercase words 2-5 chars might be symbols
        words = widget_name.split()
        for word in words:
            if word.isupper() and 2 <= len(word) <= 5:
                symbols.append(word)

        return list(set(symbols))  # Remove duplicates

    def _detect_timeframe(self, widget: Any, widget_data: Any) -> Optional[str]:
        """Detect the timeframe of the data."""
        # Check params
        if hasattr(widget, 'params'):
            for param in widget.params:
                param_name = getattr(param, 'name', '').lower()
                if param_name in ['period', 'timeframe', 'interval', 'range']:
                    return str(getattr(param, 'current_value', None))

        # Check widget name
        widget_name = getattr(widget, 'name', '').lower()
        timeframe_keywords = ['1d', '1w', '1m', '3m', '6m', '1y', '5y', 'ytd', 'daily', 'weekly', 'monthly']
        for keyword in timeframe_keywords:
            if keyword in widget_name:
                return keyword

        return None

    def _count_data_points(self, widget_data: Any) -> int:
        """Count the number of data points in widget data."""
        if not widget_data:
            return 0

        try:
            if isinstance(widget_data, list):
                total = 0
                for item in widget_data:
                    if hasattr(item, 'items'):
                        total += len(item.items)
                    else:
                        total += 1
                return total
            elif hasattr(widget_data, 'items'):
                return len(widget_data.items)
        except Exception as e:
            logger.warning(f"Error counting data points: {e}")

        return 0

    def _extract_metrics(self, widget: Any, widget_data: Any) -> List[str]:
        """Extract available metrics from widget."""
        metrics = []

        # Check widget params for metric types
        if hasattr(widget, 'params'):
            for param in widget.params:
                param_name = getattr(param, 'name', '').lower()
                if param_name in ['metric', 'metrics', 'indicator', 'field']:
                    value = getattr(param, 'current_value', None)
                    if value:
                        if isinstance(value, list):
                            metrics.extend(value)
                        else:
                            metrics.append(str(value))

        # Parse widget name for common metrics
        widget_name = getattr(widget, 'name', '').lower()
        common_metrics = ['price', 'volume', 'revenue', 'earnings', 'pe', 'pb', 'roe', 'debt', 'rsi', 'macd']
        for metric in common_metrics:
            if metric in widget_name:
                metrics.append(metric)

        return list(set(metrics))

    def _generate_suggestions(
        self,
        widget_type: str,
        symbols: List[str],
        metrics: List[str],
        timeframe: Optional[str]
    ) -> List[str]:
        """Generate analysis suggestions based on widget context."""
        suggestions = []

        # Type-specific suggestions
        if widget_type == "price_chart":
            suggestions.extend([
                "Calculate returns and volatility metrics",
                "Apply technical indicators (RSI, MACD, Moving Averages)",
                "Identify support and resistance levels",
                "Analyze volume patterns",
            ])

        elif widget_type == "financial_statement":
            suggestions.extend([
                "Calculate financial ratios (P/E, ROE, Debt/Equity)",
                "Analyze growth trends over time",
                "Compare metrics against industry peers",
                "Identify unusual items or anomalies",
            ])

        elif widget_type == "news":
            suggestions.extend([
                "Perform sentiment analysis on headlines",
                "Correlate news sentiment with price movements",
                "Identify key events and their market impact",
                "Track news volume as a volatility indicator",
            ])

        elif widget_type == "economic_indicator":
            suggestions.extend([
                "Correlate with stock market performance",
                "Analyze impact on specific sectors",
                "Track changes over time",
                "Compare against historical averages",
            ])

        # Symbol-specific suggestions
        if symbols:
            suggestions.append(f"Compare {', '.join(symbols[:3])} against each other")
            suggestions.append(f"Fetch latest news for {', '.join(symbols[:2])}")

        # Timeframe-specific suggestions
        if timeframe:
            if any(term in str(timeframe).lower() for term in ['1d', 'daily', 'intraday']):
                suggestions.append("Analyze intraday trading patterns")
            elif any(term in str(timeframe).lower() for term in ['1y', '5y', 'long']):
                suggestions.append("Perform long-term trend analysis")

        return suggestions[:5]  # Limit to top 5 suggestions

    async def analyze_dashboard(self, widgets: List[Any]) -> Dict[str, Any]:
        """
        Analyze all widgets on the dashboard to understand user focus.

        Returns:
            Dashboard profile with insights about user's interests
        """
        contexts = []
        for widget in widgets:
            try:
                context = await self.analyze_widget(widget)
                contexts.append(context)
            except Exception as e:
                logger.warning(f"Error analyzing widget: {e}")
                continue

        # Aggregate insights
        all_symbols = []
        all_types = []
        all_categories = []

        for ctx in contexts:
            all_symbols.extend(ctx.symbols)
            all_types.append(ctx.detected_type)
            if ctx.category:
                all_categories.append(ctx.category)

        # Count frequencies
        symbol_freq = {}
        for symbol in all_symbols:
            symbol_freq[symbol] = symbol_freq.get(symbol, 0) + 1

        type_freq = {}
        for wtype in all_types:
            type_freq[wtype] = type_freq.get(wtype, 0) + 1

        # Determine focus
        primary_focus = max(type_freq, key=type_freq.get) if type_freq else "general"
        top_symbols = sorted(symbol_freq.items(), key=lambda x: x[1], reverse=True)[:5]

        dashboard_profile = {
            "widget_count": len(contexts),
            "primary_focus": primary_focus,
            "focus_distribution": type_freq,
            "top_symbols": [s[0] for s in top_symbols],
            "symbol_frequency": dict(top_symbols),
            "categories": list(set(all_categories)),
            "analysis_style": self._infer_analysis_style(type_freq),
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(f"Dashboard analysis: focus={primary_focus}, symbols={top_symbols[:3]}")

        return dashboard_profile

    def _infer_analysis_style(self, type_freq: Dict[str, int]) -> str:
        """Infer the user's preferred analysis style."""
        technical_count = type_freq.get("technical_indicator", 0) + type_freq.get("price_chart", 0)
        fundamental_count = type_freq.get("financial_statement", 0) + type_freq.get("fundamental_ratio", 0)

        if technical_count > fundamental_count * 2:
            return "technical"
        elif fundamental_count > technical_count * 2:
            return "fundamental"
        else:
            return "balanced"

    async def suggest_related_widgets(self, widget: Any) -> List[str]:
        """Suggest complementary widgets based on current widget."""
        context = await self.analyze_widget(widget)
        suggestions = []

        widget_type = context.detected_type
        symbols = context.symbols

        # Type-based suggestions
        if widget_type == "price_chart":
            suggestions.extend([
                f"Add Technical Indicators for {symbols[0] if symbols else 'this asset'}",
                f"Add Volume Analysis widget",
                f"Add News widget to correlate with price movements",
            ])

        elif widget_type == "financial_statement":
            suggestions.extend([
                f"Add Price Chart to see market valuation",
                f"Add Peer Comparison widget",
                f"Add Financial Ratios dashboard",
            ])

        elif widget_type == "news":
            suggestions.extend([
                f"Add Price Chart to correlate news with price",
                f"Add Sentiment Analysis widget",
                f"Add Economic Calendar widget",
            ])

        return suggestions[:3]

    def get_widget_summary(self, context: WidgetContext) -> str:
        """Generate a human-readable summary of widget context."""
        summary_parts = [f"**{context.widget_name}**"]

        if context.symbols:
            summary_parts.append(f"Tracking: {', '.join(context.symbols)}")

        if context.timeframe:
            summary_parts.append(f"Timeframe: {context.timeframe}")

        if context.metrics:
            summary_parts.append(f"Metrics: {', '.join(context.metrics)}")

        summary_parts.append(f"Data points: {context.data_points}")

        return " | ".join(summary_parts)


# Global instance
widget_intelligence = WidgetIntelligence()
