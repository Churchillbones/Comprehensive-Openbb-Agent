"""
Financial Table Generator Tool

Create financial tables for visualization.
Migrated from: comprehensive_agent/visualizations/tables.py
"""

from typing import Dict, Any, List, Optional
import logging
from comprehensive_agent.visualizations.tables import (
    generate_tables,
    create_table,
    parse_table_data
)

logger = logging.getLogger(__name__)


async def generate_financial_table(
    data: Any,
    table_type: str = "auto",
    name: Optional[str] = None,
    description: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate financial table for visualization

    Args:
        data: Data for table (list of dicts, widget data, or dict)
        table_type: Type of table ("metrics_summary", "financial_statement", "auto")
        name: Optional table name
        description: Optional table description

    Returns:
        Dict with table artifact:
            - status: "success" or "error"
            - table: Table artifact for OpenBB
            - table_type: Type of table generated
    """
    try:
        # Handle widget data format
        if isinstance(data, list) and data and hasattr(data[0], 'items'):
            # This is widget data
            tables = await generate_tables(data)
            if tables:
                return {
                    "status": "success",
                    "table": tables[0],
                    "table_type": table_type,
                    "count": len(tables)
                }

        # Handle list of dicts
        if isinstance(data, list) and data and isinstance(data[0], dict):
            table_artifact = await create_table(data)

            if table_artifact:
                # Override name/description if provided
                if name:
                    table_artifact["name"] = name
                if description:
                    table_artifact["description"] = description

                return {
                    "status": "success",
                    "table": table_artifact,
                    "table_type": table_type
                }

        # Handle single dict
        if isinstance(data, dict):
            table_data = [data]
            table_artifact = await create_table(table_data)

            if table_artifact:
                return {
                    "status": "success",
                    "table": table_artifact,
                    "table_type": table_type
                }

        return {
            "status": "error",
            "error": "Could not generate table from provided data"
        }

    except Exception as e:
        logger.error(f"Table generation failed: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e)
        }
