"""
Financial Chart Generator Tool

Specialized financial charts (candlestick, OHLC, heatmaps, waterfall).
Migrated from: comprehensive_agent/visualizations/financial_charts.py
"""

from typing import Dict, Any, Optional, List, Sequence
import logging
from comprehensive_agent.visualizations.financial_charts import (
    candlestick_chart,
    correlation_heatmap,
    treemap_chart,
    waterfall_chart,
    dual_axis_chart
)

logger = logging.getLogger(__name__)


async def generate_financial_chart(
    data: Sequence[dict],
    chart_type: str = "candlestick",
    **kwargs
) -> Dict[str, Any]:
    """
    Generate specialized financial charts

    Args:
        data: Financial data (OHLC for candlestick, correlation matrix for heatmap, etc.)
        chart_type: Type of chart ("candlestick", "ohlc", "heatmap", "treemap", "waterfall", "dual_axis")
        **kwargs: Additional parameters specific to chart type

    Returns:
        Dict with chart artifact:
            - status: "success" or "error"
            - chart: Chart artifact for OpenBB
            - chart_type: Type of chart generated
    """
    try:
        if not data:
            return {
                "status": "error",
                "error": "No data provided for chart generation"
            }

        chart_artifact = None

        if chart_type == "candlestick":
            # Candlestick chart for OHLC data
            chart_artifact = await candlestick_chart(
                data,
                time_key=kwargs.get("time_key", "date"),
                open_key=kwargs.get("open_key", "open"),
                high_key=kwargs.get("high_key", "high"),
                low_key=kwargs.get("low_key", "low"),
                close_key=kwargs.get("close_key", "close"),
                hide_rangeslider=kwargs.get("hide_rangeslider", True),
                colors=kwargs.get("colors"),
                name=kwargs.get("name"),
                description=kwargs.get("description")
            )

        elif chart_type == "heatmap":
            # Correlation heatmap
            chart_artifact = await correlation_heatmap(
                data,
                x_key=kwargs.get("x_key"),
                y_key=kwargs.get("y_key"),
                name=kwargs.get("name"),
                description=kwargs.get("description")
            )

        elif chart_type == "treemap":
            # Treemap chart for hierarchical data
            chart_artifact = await treemap_chart(
                data,
                label_key=kwargs.get("label_key", "label"),
                parent_key=kwargs.get("parent_key", "parent"),
                value_key=kwargs.get("value_key", "value"),
                name=kwargs.get("name"),
                description=kwargs.get("description")
            )

        elif chart_type == "waterfall":
            # Waterfall chart for breakdown data
            chart_artifact = await waterfall_chart(
                data,
                x_key=kwargs.get("x_key", "label"),
                y_key=kwargs.get("y_key", "value"),
                name=kwargs.get("name"),
                description=kwargs.get("description")
            )

        elif chart_type == "dual_axis":
            # Dual-axis chart
            secondary_data = kwargs.get("secondary_data", [])
            if not secondary_data:
                return {
                    "status": "error",
                    "error": "secondary_data required for dual_axis chart"
                }

            chart_artifact = await dual_axis_chart(
                primary_data=data,
                secondary_data=secondary_data,
                x_key=kwargs.get("x_key", "date"),
                primary_y_key=kwargs.get("primary_y_key", "value"),
                secondary_y_key=kwargs.get("secondary_y_key", "value"),
                name=kwargs.get("name"),
                description=kwargs.get("description")
            )

        else:
            return {
                "status": "error",
                "error": f"Unknown chart type: {chart_type}"
            }

        if chart_artifact:
            return {
                "status": "success",
                "chart": chart_artifact,
                "chart_type": chart_type
            }

        return {
            "status": "error",
            "error": f"Failed to generate {chart_type} chart"
        }

    except Exception as e:
        logger.error(f"Financial chart generation failed: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "chart_type": chart_type
        }
