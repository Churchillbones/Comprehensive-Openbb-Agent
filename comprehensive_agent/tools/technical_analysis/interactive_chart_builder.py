"""
Interactive Chart Builder Tool

Add annotations, overlays, and technical indicators to charts.
Migrated from: comprehensive_agent/visualizations/interactive_charts.py
"""

from typing import Dict, Any, List, Optional
import logging
from comprehensive_agent.visualizations.interactive_charts import (
    add_technical_indicators,
    add_annotations,
    create_chart_grid
)

logger = logging.getLogger(__name__)


async def build_interactive_chart(
    base_chart: Dict[str, Any],
    action: str = "add_indicators",
    **kwargs
) -> Dict[str, Any]:
    """
    Build interactive chart with overlays and annotations

    Args:
        base_chart: Base chart artifact to enhance
        action: Action to perform:
            - "add_indicators": Add technical indicator overlays
            - "add_annotations": Add text/shape annotations
            - "create_grid": Combine multiple charts into grid
        **kwargs: Action-specific parameters

    Returns:
        Dict with enhanced chart:
            - status: "success" or "error"
            - chart: Enhanced chart artifact
            - action: Action performed
    """
    try:
        if action == "add_indicators":
            # Add technical indicator overlays
            indicators = kwargs.get("indicators", [])
            if not indicators:
                return {
                    "status": "error",
                    "error": "No indicators provided"
                }

            enhanced_chart = add_technical_indicators(base_chart, indicators)

            return {
                "status": "success",
                "chart": enhanced_chart,
                "action": "add_indicators",
                "indicators_added": len(indicators)
            }

        elif action == "add_annotations":
            # Add annotations
            annotations = kwargs.get("annotations", [])
            if not annotations:
                return {
                    "status": "error",
                    "error": "No annotations provided"
                }

            enhanced_chart = add_annotations(base_chart, annotations)

            return {
                "status": "success",
                "chart": enhanced_chart,
                "action": "add_annotations",
                "annotations_added": len(annotations)
            }

        elif action == "create_grid":
            # Create chart grid
            charts = kwargs.get("charts", [base_chart])
            layout = kwargs.get("layout", "2x2")

            grid_chart = create_chart_grid(charts, layout)

            return {
                "status": "success",
                "chart": grid_chart,
                "action": "create_grid",
                "layout": layout,
                "charts_in_grid": len(charts)
            }

        else:
            return {
                "status": "error",
                "error": f"Unknown action: {action}"
            }

    except Exception as e:
        logger.error(f"Interactive chart building failed: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "action": action
        }


async def add_indicator_overlay(
    chart_artifact: Dict[str, Any],
    indicator_name: str,
    indicator_data: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Add a single technical indicator overlay to a chart

    Args:
        chart_artifact: Base chart artifact
        indicator_name: Name of indicator (e.g., "RSI 14", "MACD")
        indicator_data: Indicator data points [{"x": ..., "y": ...}, ...]

    Returns:
        Enhanced chart artifact
    """
    indicator_config = {
        "type": "line",
        "name": indicator_name,
        "data": indicator_data,
        "yAxisIndex": 1  # Secondary axis for indicators
    }

    return await build_interactive_chart(
        chart_artifact,
        action="add_indicators",
        indicators=[indicator_config]
    )


async def add_event_annotation(
    chart_artifact: Dict[str, Any],
    events: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Add event annotations to a chart

    Args:
        chart_artifact: Base chart artifact
        events: List of events [{"x": "2024-01-15", "text": "Earnings", "yPosition": 0.95}, ...]

    Returns:
        Enhanced chart artifact
    """
    return await build_interactive_chart(
        chart_artifact,
        action="add_annotations",
        annotations=events
    )
