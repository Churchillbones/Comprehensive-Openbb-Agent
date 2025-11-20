"""
API Data Fetcher Tool

Fetches and prepares data from external APIs.
Migrated from: comprehensive_agent/processors/api_data.py
"""

from typing import Dict, Any, Optional
import logging
from comprehensive_agent.processors.api_data import process_api_data, prepare_api_data_for_visualization

logger = logging.getLogger(__name__)


async def fetch_api_data(
    endpoint: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
    api_data: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Fetch and process API data

    Args:
        endpoint: API endpoint URL (if fetching new data)
        params: API request parameters
        api_data: Pre-fetched API data to process

    Returns:
        Dict with processed API data:
            - status: "success" or "error"
            - data: Processed data
            - metadata: API response metadata
    """
    try:
        if api_data:
            # Process existing API data
            processed = await process_api_data(api_data)
            return {
                "status": "success",
                "data": processed,
                "metadata": {"source": "provided_data"}
            }

        elif endpoint:
            # Fetch from API endpoint (simplified for now)
            logger.info(f"Fetching data from endpoint: {endpoint}")
            # TODO: Implement actual API fetching when needed
            return {
                "status": "pending",
                "message": "API fetching not yet implemented",
                "endpoint": endpoint,
                "params": params
            }

        else:
            return {
                "status": "error",
                "error": "No endpoint or data provided"
            }

    except Exception as e:
        logger.error(f"API data fetch failed: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e)
        }


async def prepare_for_visualization(api_data: Any) -> Dict[str, Any]:
    """
    Prepare API data for visualization

    Args:
        api_data: Raw API data

    Returns:
        Data formatted for visualization
    """
    try:
        viz_data = await prepare_api_data_for_visualization(api_data)
        return {
            "status": "success",
            "data": viz_data
        }

    except Exception as e:
        logger.error(f"Visualization preparation failed: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e)
        }
