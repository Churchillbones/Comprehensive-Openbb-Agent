"""
Market Data Tools

Tools for the Market Data Agent:
1. Widget Processor - Extract and process widget data
2. Widget Intelligence - Analyze widget context and type
3. API Data Fetcher - Fetch data from external APIs
4. API Data Processor - Advanced data normalization
5. Data Validator - Validate data quality
6. OHLCV Extractor - Extract OHLCV data
7. Stream Manager - Manage real-time data streams
"""

from .widget_processor import process_widget_data
from .widget_intelligence import analyze_widget_intelligence
from .api_data_fetcher import fetch_api_data
from .api_data_processor import process_api_data_advanced
from .data_validator import validate_data
from .ohlcv_extractor import extract_ohlcv
from .stream_manager import manage_stream

__all__ = [
    "process_widget_data",
    "analyze_widget_intelligence",
    "fetch_api_data",
    "process_api_data_advanced",
    "validate_data",
    "extract_ohlcv",
    "manage_stream"
]
