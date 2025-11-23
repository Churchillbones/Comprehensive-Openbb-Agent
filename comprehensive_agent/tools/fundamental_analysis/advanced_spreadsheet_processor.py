"""
Advanced Spreadsheet Processor Tool

Handle complex spreadsheets with financial statement detection.
Migrated from: comprehensive_agent/processors/spreadsheet_processor.py
"""

from typing import Dict, Any, Optional
import logging
from comprehensive_agent.processors.spreadsheet_processor import SpreadsheetProcessor

logger = logging.getLogger(__name__)

# Global processor instance
_processor = SpreadsheetProcessor()


async def process_advanced_spreadsheet(
    file_data: bytes,
    filename: str,
    detect_financials: bool = True
) -> Dict[str, Any]:
    """
    Process complex spreadsheets with financial statement detection

    Args:
        file_data: Raw file bytes (Excel)
        filename: Original filename
        detect_financials: Whether to detect financial statements

    Returns:
        Dict with processed data:
            - status: "success" or "error"
            - sheets: Dict of processed sheets
            - financial_statements: Detected financial statements
            - metadata: Processing metadata
    """
    try:
        # Use SpreadsheetProcessor for advanced parsing
        result = await _processor.parse_excel_file(file_data)

        if result and result.get("sheets"):
            return {
                "status": "success",
                "data": result,
                "sheets": result.get("sheets", {}),
                "financial_statements": result.get("financial_statements", {}),
                "metadata": result.get("metadata", {}),
                "filename": filename
            }
        else:
            return {
                "status": "error",
                "error": "Failed to parse spreadsheet",
                "filename": filename
            }

    except Exception as e:
        logger.error(f"Advanced spreadsheet processing failed: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "filename": filename
        }
