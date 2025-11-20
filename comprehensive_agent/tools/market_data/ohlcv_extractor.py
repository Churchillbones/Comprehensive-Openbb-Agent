"""
OHLCV Extractor Tool

Extracts OHLCV (Open, High, Low, Close, Volume) data from market data sources.
NEW tool created for Market Data Agent.
"""

from typing import Dict, Any, List, Optional
import logging
import json

logger = logging.getLogger(__name__)


async def extract_ohlcv(data: Any) -> Dict[str, Any]:
    """
    Extract OHLCV data from various data formats

    Args:
        data: Market data (can be widget data, API response, or raw data)

    Returns:
        Dict with OHLCV data:
            - status: "success" or "error"
            - data: List of OHLCV records
            - symbols: List of symbols found
            - period_count: Number of periods
            - metadata: Additional metadata
    """
    try:
        ohlcv_records = []
        symbols = set()

        # Handle different data formats
        if isinstance(data, dict):
            # Check if it's processed widget data
            if "data" in data:
                raw_data = data["data"]
            else:
                raw_data = data

            # Try to extract OHLCV from dict
            ohlcv_records, symbols = await _extract_from_dict(raw_data)

        elif isinstance(data, list):
            # Process list of data
            ohlcv_records, symbols = await _extract_from_list(data)

        elif isinstance(data, str):
            # Try to parse JSON string
            try:
                parsed = json.loads(data)
                ohlcv_records, symbols = await _extract_from_dict(parsed)
            except json.JSONDecodeError:
                # Try to parse as text
                ohlcv_records, symbols = await _extract_from_text(data)

        # Calculate returns if we have close prices
        if ohlcv_records:
            ohlcv_records = await _calculate_returns(ohlcv_records)

        logger.info(f"Extracted {len(ohlcv_records)} OHLCV records for {len(symbols)} symbols")

        return {
            "status": "success" if ohlcv_records else "no_data",
            "data": ohlcv_records,
            "symbols": list(symbols),
            "period_count": len(ohlcv_records),
            "metadata": {
                "has_volume": any("volume" in str(r).lower() for r in ohlcv_records),
                "has_dates": any("date" in str(r).lower() or "time" in str(r).lower() for r in ohlcv_records)
            }
        }

    except Exception as e:
        logger.error(f"OHLCV extraction failed: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "data": [],
            "symbols": [],
            "period_count": 0
        }


async def _extract_from_dict(data: Dict) -> tuple[List[Dict], set]:
    """Extract OHLCV from dictionary data"""
    records = []
    symbols = set()

    # Check for common OHLCV keys
    ohlcv_keys = ["open", "high", "low", "close", "volume"]

    # If dict has these keys directly
    if any(key in data for key in ohlcv_keys):
        record = {k: v for k, v in data.items() if k in ohlcv_keys}
        if record:
            records.append(record)
            if "symbol" in data:
                symbols.add(data["symbol"])

    # Check for nested data
    elif "results" in data:
        records, symbols = await _extract_from_list(data["results"])

    elif "data" in data and isinstance(data["data"], list):
        records, symbols = await _extract_from_list(data["data"])

    return records, symbols


async def _extract_from_list(data: List) -> tuple[List[Dict], set]:
    """Extract OHLCV from list data"""
    records = []
    symbols = set()

    for item in data:
        if isinstance(item, dict):
            # Check if this item has OHLCV data
            ohlcv_keys = ["open", "high", "low", "close"]
            if any(key in item for key in ohlcv_keys):
                record = {}

                # Extract OHLCV fields
                for key in ["open", "high", "low", "close", "volume", "date", "timestamp", "symbol"]:
                    if key in item:
                        record[key] = item[key]

                if record:
                    records.append(record)

                    if "symbol" in item:
                        symbols.add(item["symbol"])

    return records, symbols


async def _extract_from_text(text: str) -> tuple[List[Dict], set]:
    """Extract OHLCV from text data"""
    records = []
    symbols = set()

    # Try to find OHLCV patterns in text
    lines = text.split('\n')

    for line in lines:
        line_lower = line.lower()

        # Look for price-like patterns
        if any(keyword in line_lower for keyword in ["open", "high", "low", "close"]):
            # This is a simplified extraction
            # In production, would use regex patterns
            record = {"text": line.strip()}
            records.append(record)

    return records, symbols


async def _calculate_returns(records: List[Dict]) -> List[Dict]:
    """Calculate returns from OHLCV data"""
    try:
        # Sort by date if available
        if records and "date" in records[0]:
            records = sorted(records, key=lambda x: x.get("date", ""))

        # Calculate returns
        for i in range(1, len(records)):
            prev_close = records[i-1].get("close")
            curr_close = records[i].get("close")

            if prev_close and curr_close:
                try:
                    prev_close = float(prev_close)
                    curr_close = float(curr_close)
                    returns = (curr_close - prev_close) / prev_close
                    records[i]["returns"] = round(returns, 6)
                except (ValueError, ZeroDivisionError):
                    pass

        return records

    except Exception as e:
        logger.warning(f"Return calculation failed: {e}")
        return records
