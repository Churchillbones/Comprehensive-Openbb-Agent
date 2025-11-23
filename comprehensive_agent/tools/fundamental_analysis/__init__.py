"""
Fundamental Analysis Tools

Tools for the Fundamental Analysis Agent:
1. Spreadsheet Processor - Process Excel/CSV files
2. Advanced Spreadsheet Processor - Handle complex spreadsheets
3. Financial Metrics Calculator - Calculate P/E, ROE, ROA, etc.
4. Valuation Model Engine - DCF, comparable company analysis
5. Growth Analyzer - Revenue, earnings, CAGR analysis
6. Profitability Analyzer - Margin and profitability analysis
7. Financial Table Generator - Create financial tables
"""

from .spreadsheet_processor import process_spreadsheet
from .advanced_spreadsheet_processor import process_advanced_spreadsheet
from .financial_metrics_calculator import calculate_financial_metrics
from .valuation_model_engine import run_valuation_model
from .growth_analyzer import analyze_growth
from .profitability_analyzer import analyze_profitability
from .financial_table_generator import generate_financial_table

__all__ = [
    "process_spreadsheet",
    "process_advanced_spreadsheet",
    "calculate_financial_metrics",
    "run_valuation_model",
    "analyze_growth",
    "analyze_profitability",
    "generate_financial_table"
]
