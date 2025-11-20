"""
Widget Intelligence Tool

Automatically detects widget types, extracts context, and provides smart analysis
suggestions based on OpenBB workspace widget data.

Migrated from: comprehensive_agent/processors/widget_intelligence.py
"""

from typing import Any, Dict, List, Optional
import logging
from comprehensive_agent.processors.widget_intelligence import (
    WidgetIntelligence,
    WidgetContext,
    widget_intelligence as _widget_intelligence_instance
)

logger = logging.getLogger(__name__)


async def analyze_widget_intelligence(widget_data: Any, widget: Any = None) -> Dict[str, Any]:
    """
    Analyze widget and extract intelligent context

    Args:
        widget_data: Widget data content
        widget: Optional widget object from OpenBB

    Returns:
        Dict with widget analysis:
            - widget_type: Detected widget type
            - symbols: Extracted symbols/tickers
            - timeframe: Detected timeframe
            - data_points: Number of data points
            - metrics: Extracted metrics
            - suggestions: Analysis suggestions
            - has_price_data: Whether widget contains price data
    """
    try:
        # Use global widget intelligence instance
        intelligence = _widget_intelligence_instance

        # If we have a widget object, analyze it
        if widget:
            context = await intelligence.analyze_widget(widget, widget_data)
        else:
            # Create a simple mock widget object from data
            mock_widget = type('Widget', (), {
                'uuid': str(hash(str(widget_data))),
                'name': 'Widget',
                'params': []
            })()
            context = await intelligence.analyze_widget(mock_widget, widget_data)

        # Convert WidgetContext to dict
        result = {
            "widget_type": context.detected_type,
            "symbols": context.symbols,
            "timeframe": context.timeframe,
            "data_points": context.data_points,
            "metrics": context.metrics,
            "category": context.category,
            "suggestions": context.analysis_suggestions,
            "has_price_data": context.detected_type in ["price_chart", "technical_indicator"],
            "summary": intelligence.get_widget_summary(context)
        }

        logger.debug(f"Widget analysis: type={result['widget_type']}, symbols={result['symbols']}")

        return result

    except Exception as e:
        logger.error(f"Widget intelligence analysis failed: {e}", exc_info=True)
        return {
            "widget_type": "unknown",
            "symbols": [],
            "timeframe": None,
            "data_points": 0,
            "metrics": [],
            "category": None,
            "suggestions": [],
            "has_price_data": False,
            "error": str(e)
        }


async def analyze_dashboard(widgets: List[Any]) -> Dict[str, Any]:
    """
    Analyze all widgets on dashboard to understand user focus

    Args:
        widgets: List of widget objects

    Returns:
        Dashboard profile with insights about user interests
    """
    try:
        intelligence = _widget_intelligence_instance
        return await intelligence.analyze_dashboard(widgets)

    except Exception as e:
        logger.error(f"Dashboard analysis failed: {e}", exc_info=True)
        return {
            "widget_count": 0,
            "primary_focus": "unknown",
            "error": str(e)
        }


async def suggest_related_widgets(widget: Any) -> List[str]:
    """
    Suggest complementary widgets based on current widget

    Args:
        widget: Widget object

    Returns:
        List of widget suggestions
    """
    try:
        intelligence = _widget_intelligence_instance
        return await intelligence.suggest_related_widgets(widget)

    except Exception as e:
        logger.error(f"Widget suggestion failed: {e}", exc_info=True)
        return []
