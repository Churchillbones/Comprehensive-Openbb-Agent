"""
Spreadsheet Processor Tool

Process Excel/CSV files with financial data.
Migrated from: comprehensive_agent/processors/spreadsheet.py
"""

from typing import Dict, Any, Optional, Union, List
import logging
from comprehensive_agent.processors.spreadsheet import process_spreadsheet_data

logger = logging.getLogger(__name__)


async def process_spreadsheet(
    widget_data: Optional[Any] = None,
    file_data: Optional[bytes] = None,
    filename: Optional[str] = None
) -> Dict[str, Any]:
    """
    Process spreadsheet data from widgets or files

    Args:
        widget_data: Widget data containing spreadsheet
        file_data: Raw file bytes (Excel/CSV)
        filename: Original filename

    Returns:
        Dict with processed spreadsheet:
            - status: "success" or "error"
            - sheets: List of processed sheets
            - summary: Data summary with row/column counts
            - metadata: File metadata
    """
    try:
        if widget_data:
            # Process from widget
            result = await process_spreadsheet_data(widget_data)

            if result and result.get("sheets"):
                return {
                    "status": "success",
                    **result
                }
            else:
                return {
                    "status": "error",
                    "error": "No spreadsheet data found in widget"
                }

        elif file_data:
            # Process from file bytes
            # For now, pass to advanced processor
            # Could add simple CSV parsing here
            return {
                "status": "pending",
                "message": "Use advanced_spreadsheet_processor for file uploads",
                "filename": filename
            }

        else:
            return {
                "status": "error",
                "error": "No widget_data or file_data provided"
            }

    except Exception as e:
        logger.error(f"Spreadsheet processing failed: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e)
        }
