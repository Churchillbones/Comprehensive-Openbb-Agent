"""
Chart Generator Tool

General chart generation for technical analysis.
Migrated from: comprehensive_agent/visualizations/charts.py
"""

from typing import Dict, Any, List, Optional
import logging
from comprehensive_agent.visualizations.charts import (
    generate_charts,
    create_chart,
    detect_chart_types,
    parse_chart_data
)

logger = logging.getLogger(__name__)


async def generate_chart(
    data: Any,
    chart_type: Optional[str] = None,
    x_key: Optional[str] = None,
    y_keys: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Generate chart for technical analysis

    Args:
        data: Data to chart (can be widget data, list of dicts, or processed data)
        chart_type: Chart type ("line", "bar", "scatter", "pie") - auto-detects if not specified
        x_key: X-axis key (optional for auto-detection)
        y_keys: Y-axis keys (optional for auto-detection)

    Returns:
        Dict with chart artifact:
            - status: "success" or "error"
            - chart: Chart artifact for OpenBB
            - chart_type: Type of chart generated
            - config: Chart configuration used
    """
    try:
        # Handle widget data format
        if isinstance(data, list) and data and hasattr(data[0], 'items'):
            # This is widget data - use existing generate_charts function
            charts = await generate_charts(data)
            if charts:
                return {
                    "status": "success",
                    "chart": charts[0],
                    "chart_type": "auto",
                    "count": len(charts)
                }

        # Handle raw data format
        if isinstance(data, list) and data and isinstance(data[0], dict):
            # Auto-detect chart configuration
            if not chart_type:
                configs = detect_chart_types(data)
                if configs:
                    config = configs[0]
                else:
                    # Default config
                    keys = list(data[0].keys())
                    config = {
                        "type": "line",
                        "x_key": keys[0] if keys else "x",
                        "y_keys": keys[1:2] if len(keys) > 1 else ["y"],
                        "name": "Chart",
                        "description": "Generated chart"
                    }
            else:
                # Use provided parameters
                config = {
                    "type": chart_type,
                    "x_key": x_key or list(data[0].keys())[0],
                    "y_keys": y_keys or [list(data[0].keys())[1]] if len(data[0].keys()) > 1 else ["value"],
                    "name": f"{chart_type.title()} Chart",
                    "description": f"{chart_type} chart visualization"
                }

            # Create chart
            chart_artifact = await create_chart(data, config)

            if chart_artifact:
                return {
                    "status": "success",
                    "chart": chart_artifact,
                    "chart_type": config["type"],
                    "config": config
                }

        return {
            "status": "error",
            "error": "Could not generate chart from provided data"
        }

    except Exception as e:
        logger.error(f"Chart generation failed: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e)
        }
