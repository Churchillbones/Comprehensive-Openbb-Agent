"""
Stream Manager Tool

Manages real-time data streams for market data.
NEW tool created for Market Data Agent.
"""

from typing import Dict, Any, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


# Simple in-memory stream cache
_active_streams: Dict[str, Dict[str, Any]] = {}


async def manage_stream(
    action: str = "status",
    stream_id: Optional[str] = None,
    symbol: Optional[str] = None,
    data: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Manage real-time data streams

    Args:
        action: Action to perform ("start", "stop", "status", "update")
        stream_id: Unique identifier for the stream
        symbol: Symbol/ticker to stream
        data: Data to update in stream

    Returns:
        Dict with stream status:
            - status: "success" or "error"
            - stream_id: Stream identifier
            - action: Action performed
            - data: Stream data or status info
    """
    try:
        if action == "start":
            return await _start_stream(stream_id or f"stream_{symbol}_{datetime.now().timestamp()}", symbol)

        elif action == "stop":
            return await _stop_stream(stream_id)

        elif action == "status":
            return await _get_stream_status(stream_id)

        elif action == "update":
            return await _update_stream(stream_id, data)

        elif action == "list":
            return await _list_streams()

        else:
            return {
                "status": "error",
                "error": f"Unknown action: {action}",
                "valid_actions": ["start", "stop", "status", "update", "list"]
            }

    except Exception as e:
        logger.error(f"Stream management failed: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "action": action
        }


async def _start_stream(stream_id: str, symbol: Optional[str]) -> Dict[str, Any]:
    """Start a new data stream"""
    if stream_id in _active_streams:
        return {
            "status": "error",
            "error": f"Stream {stream_id} already exists",
            "stream_id": stream_id
        }

    stream = {
        "id": stream_id,
        "symbol": symbol,
        "started_at": datetime.now().isoformat(),
        "last_update": None,
        "update_count": 0,
        "latest_data": None,
        "is_active": True
    }

    _active_streams[stream_id] = stream

    logger.info(f"Started stream: {stream_id} for symbol: {symbol}")

    return {
        "status": "success",
        "action": "start",
        "stream_id": stream_id,
        "data": stream
    }


async def _stop_stream(stream_id: Optional[str]) -> Dict[str, Any]:
    """Stop an active stream"""
    if not stream_id:
        return {
            "status": "error",
            "error": "stream_id required for stop action"
        }

    if stream_id not in _active_streams:
        return {
            "status": "error",
            "error": f"Stream {stream_id} not found",
            "stream_id": stream_id
        }

    stream = _active_streams[stream_id]
    stream["is_active"] = False
    stream["stopped_at"] = datetime.now().isoformat()

    # Remove from active streams
    del _active_streams[stream_id]

    logger.info(f"Stopped stream: {stream_id}")

    return {
        "status": "success",
        "action": "stop",
        "stream_id": stream_id,
        "data": stream
    }


async def _get_stream_status(stream_id: Optional[str]) -> Dict[str, Any]:
    """Get status of a stream"""
    if stream_id:
        if stream_id in _active_streams:
            return {
                "status": "success",
                "action": "status",
                "stream_id": stream_id,
                "data": _active_streams[stream_id]
            }
        else:
            return {
                "status": "error",
                "error": f"Stream {stream_id} not found",
                "stream_id": stream_id
            }
    else:
        # Return all streams status
        return await _list_streams()


async def _update_stream(stream_id: Optional[str], data: Any) -> Dict[str, Any]:
    """Update stream with new data"""
    if not stream_id:
        return {
            "status": "error",
            "error": "stream_id required for update action"
        }

    if stream_id not in _active_streams:
        return {
            "status": "error",
            "error": f"Stream {stream_id} not found",
            "stream_id": stream_id
        }

    stream = _active_streams[stream_id]
    stream["last_update"] = datetime.now().isoformat()
    stream["update_count"] += 1
    stream["latest_data"] = data

    logger.debug(f"Updated stream {stream_id}: update #{stream['update_count']}")

    return {
        "status": "success",
        "action": "update",
        "stream_id": stream_id,
        "data": stream
    }


async def _list_streams() -> Dict[str, Any]:
    """List all active streams"""
    return {
        "status": "success",
        "action": "list",
        "active_streams": len(_active_streams),
        "streams": list(_active_streams.keys()),
        "data": {k: v for k, v in _active_streams.items()}
    }


async def get_latest_quote(stream_id: str) -> Dict[str, Any]:
    """
    Get latest quote from a stream

    Args:
        stream_id: Stream identifier

    Returns:
        Latest data from stream
    """
    try:
        if stream_id not in _active_streams:
            return {
                "status": "error",
                "error": f"Stream {stream_id} not found"
            }

        stream = _active_streams[stream_id]
        return {
            "status": "success",
            "stream_id": stream_id,
            "symbol": stream.get("symbol"),
            "data": stream.get("latest_data"),
            "last_update": stream.get("last_update")
        }

    except Exception as e:
        logger.error(f"Failed to get latest quote: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e)
        }
