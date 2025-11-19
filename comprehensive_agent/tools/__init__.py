"""
Tools module for OpenBB Comprehensive Agent

This module contains all tools organized by agent domain:
- market_data: Widget processing, API data fetching, data validation
- technical: Chart generation, indicator calculation, pattern detection
- fundamental: Spreadsheet processing, financial metrics, valuation models
- risk: Volatility analysis, VaR calculation, correlation engine
- portfolio: Portfolio analysis, optimization, allocation
- economic: Forecasting, feature engineering, regression models
- news: Web search, sentiment analysis, PDF processing, citations
"""

__all__ = [
    "market_data",
    "technical",
    "fundamental",
    "risk",
    "portfolio",
    "economic",
    "news"
]