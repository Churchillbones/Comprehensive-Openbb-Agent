"""
Data Validator Tool

Validates data quality and integrity for market data.
Wrapper for: comprehensive_agent/processors/data_validator.py
"""

from typing import Dict, Any
import logging
from comprehensive_agent.processors.data_validator import DataValidator as _DataValidator

logger = logging.getLogger(__name__)


async def validate_data(data: Any) -> Dict[str, Any]:
    """
    Validate market data quality and integrity

    Args:
        data: Data to validate (can be dict, list, or other structure)

    Returns:
        Dict with validation results:
            - status: "valid" or "invalid"
            - issues: List of validation issues found
            - data_quality_score: Score from 0-100
            - suggestions: Suggestions for fixing issues
    """
    try:
        issues = []
        suggestions = []

        # Validate basic structure
        if not data:
            return {
                "status": "invalid",
                "issues": ["Data is empty or None"],
                "data_quality_score": 0,
                "suggestions": ["Provide non-empty data"]
            }

        # Validate widget structure if applicable
        if isinstance(data, (list, dict)):
            if isinstance(data, dict) and "widget" in str(type(data)).lower():
                is_valid, message = _DataValidator.validate_widget_structure(data)
                if not is_valid:
                    issues.append(f"Widget validation: {message}")
                    suggestions.append("Check widget data format")

        # Check for common data issues
        data_str = str(data).lower()

        # Check for null/none values
        if "none" in data_str or "null" in data_str:
            issues.append("Contains null/None values")
            suggestions.append("Handle missing data appropriately")

        # Check for error indicators
        if "error" in data_str:
            issues.append("Data contains error indicators")
            suggestions.append("Review error messages in data")

        # Calculate quality score
        max_issues = 10
        issue_count = len(issues)
        quality_score = max(0, 100 - (issue_count / max_issues * 100))

        # Determine status
        status = "valid" if quality_score >= 70 else "invalid"

        logger.debug(f"Data validation: status={status}, quality={quality_score}%, issues={len(issues)}")

        return {
            "status": status,
            "issues": issues,
            "data_quality_score": round(quality_score, 2),
            "suggestions": suggestions if issues else ["Data looks good!"]
        }

    except Exception as e:
        logger.error(f"Data validation failed: {e}", exc_info=True)
        return {
            "status": "error",
            "issues": [f"Validation error: {str(e)}"],
            "data_quality_score": 0,
            "suggestions": ["Check data format and structure"]
        }
