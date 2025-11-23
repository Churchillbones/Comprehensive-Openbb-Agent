"""
Widget Processor Tool

Extracts and processes data from OpenBB workspace widgets.
Migrated from: comprehensive_agent/processors/widgets.py
"""

from typing import List, Optional, Dict, Any, Union
from comprehensive_agent.processors.error_handler import ErrorHandler, DataProcessingError, error_boundary
from comprehensive_agent.processors.data_validator import DataValidator


@error_boundary(DataProcessingError)
async def process_widget_data(widget_data: Union[List[Any], Any]) -> Dict[str, Any]:
    """
    Process widget data and extract content

    Args:
        widget_data: Widget data from OpenBB workspace (list or single widget)

    Returns:
        Dict with processed data and metadata
    """
    if not widget_data:
        return {
            "status": "empty",
            "data": "",
            "metadata": {}
        }

    try:
        # Validate widget structure
        is_valid, message = DataValidator.validate_widget_structure(widget_data)
        if not is_valid:
            await ErrorHandler.log_error_with_context(
                DataProcessingError(f"Widget validation failed: {message}"),
                widget_data
            )
            return {
                "status": "error",
                "error": f"Data processing issue: {message}",
                "data": ""
            }

        # Process widget data
        context_str = "Use the following data to answer the question:\n\n"
        result_str = "--- Widget Data ---\n"

        if isinstance(widget_data, list):
            for content in widget_data:
                result_str += await _process_single_widget_content(content)
        else:
            result_str += await _process_single_widget_content(widget_data)

        context_str += result_str

        # Get metadata
        metadata = await get_widget_metadata(widget_data if isinstance(widget_data, list) else [widget_data])

        return {
            "status": "success",
            "data": context_str,
            "raw_data": result_str,
            "metadata": metadata
        }

    except Exception as e:
        error_info = await ErrorHandler.handle_data_error(e, {"data_type": type(widget_data).__name__})
        return {
            "status": "error",
            "error": f"Widget processing failed: {error_info['message']}",
            "data": ""
        }


async def _process_single_widget_content(content: Any) -> str:
    """Process individual widget content with error handling"""
    result = ""

    try:
        if hasattr(content, 'items'):
            for item in content.items:
                item_content = getattr(item, 'content', '')
                if item_content:
                    result += f"{item_content}\n------\n"
        elif hasattr(content, 'content'):
            content_data = getattr(content, 'content', '')
            if content_data:
                result += f"{content_data}\n------\n"
        elif isinstance(content, dict):
            for item in content.get("items", []):
                item_content = item.get('content', '')
                if item_content:
                    result += f"{item_content}\n------\n"
    except Exception as e:
        result += f"Content processing error: {str(e)}\n------\n"

    return result


async def extract_data_for_visualization(widget_data: List[Dict[str, Any]]) -> List[dict]:
    """
    Extract data suitable for visualization from widgets

    Args:
        widget_data: List of widget data dictionaries

    Returns:
        List of data items ready for visualization
    """
    visualization_data = []

    for result in widget_data:
        for item in result.get("items", []):
            try:
                import json
                content = item.get("content", "")
                if content and content.strip().startswith(('[', '{')):
                    data = json.loads(content)
                    if isinstance(data, list):
                        visualization_data.extend(data)
                    elif isinstance(data, dict):
                        visualization_data.append(data)
            except (json.JSONDecodeError, KeyError):
                continue

    return visualization_data


async def get_widget_metadata(widget_data: List[Dict[str, Any]]) -> dict:
    """
    Extract metadata from widget data

    Args:
        widget_data: List of widget data dictionaries

    Returns:
        Dict with widget metadata
    """
    metadata = {
        "widget_count": len(widget_data),
        "total_items": sum(len(result.get("items", [])) for result in widget_data),
        "data_types": set(),
        "has_numerical_data": False,
        "has_time_series": False
    }

    for result in widget_data:
        for item in result.get("items", []):
            content_type = item.get("content_type")
            if content_type:
                metadata["data_types"].add(content_type)

            content = item.get("content", "")
            if content:
                content_lower = content.lower()
                if any(num_indicator in content_lower for num_indicator in ['price', 'value', 'amount', 'volume']):
                    metadata["has_numerical_data"] = True
                if any(time_indicator in content_lower for time_indicator in ['date', 'time', 'timestamp']):
                    metadata["has_time_series"] = True

    metadata["data_types"] = list(metadata["data_types"])
    return metadata
