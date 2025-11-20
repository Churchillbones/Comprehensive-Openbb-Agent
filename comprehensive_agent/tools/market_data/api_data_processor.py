"""
API Data Processor Tool (Advanced)

Advanced normalization and merging of multi-source API data.
Migrated from: comprehensive_agent/processors/api_data_processor.py
"""

from typing import Dict, Any, List
import logging
from comprehensive_agent.processors.api_data_processor import APIDataProcessor

logger = logging.getLogger(__name__)

# Create global processor instance
_processor = APIDataProcessor()


async def process_api_data_advanced(
    api_data: Any,
    normalize: bool = True,
    merge_sources: bool = False
) -> Dict[str, Any]:
    """
    Advanced processing of API data with normalization and merging

    Args:
        api_data: Raw API data
        normalize: Whether to normalize the data
        merge_sources: Whether to merge data from multiple sources

    Returns:
        Dict with processed data:
            - status: "success" or "error"
            - data: Processed and normalized data
            - metadata: Processing metadata
    """
    try:
        if normalize:
            normalized = await _processor.normalize_data(api_data)
        else:
            normalized = api_data

        result = {
            "status": "success",
            "data": normalized,
            "metadata": {
                "normalized": normalize,
                "merged": merge_sources
            }
        }

        return result

    except Exception as e:
        logger.error(f"Advanced API processing failed: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "data": None
        }


async def merge_data_sources(data_sources: List[Any]) -> Dict[str, Any]:
    """
    Merge data from multiple sources

    Args:
        data_sources: List of data from different sources

    Returns:
        Merged data dictionary
    """
    try:
        merged = await _processor.merge_sources(data_sources)
        return {
            "status": "success",
            "data": merged,
            "metadata": {"source_count": len(data_sources)}
        }

    except Exception as e:
        logger.error(f"Data merging failed: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e)
        }
