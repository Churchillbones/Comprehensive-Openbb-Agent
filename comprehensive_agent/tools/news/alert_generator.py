"""
Alert Generator Tool

Generate market alerts based on conditions.
Migrated from: comprehensive_agent/utils/alerting.py
"""

from typing import Dict, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


async def generate_alert(
    message: str,
    context: Optional[Dict[str, Any]] = None,
    severity: str = "info"
) -> Dict[str, Any]:
    """
    Generate a market alert

    Args:
        message: Alert message
        context: Optional context data (e.g., sentiment, symbols, etc.)
        severity: Alert severity ("info", "warning", "critical")

    Returns:
        Dict with alert information:
            - status: "success" or "error"
            - alert: Alert details
            - timestamp: Alert generation time
    """
    try:
        alert = {
            "message": message,
            "severity": severity,
            "timestamp": datetime.now().isoformat(),
            "context": context or {},
            "alert_id": f"alert_{int(datetime.now().timestamp())}"
        }

        logger.info(f"Generated {severity} alert: {message}")

        # In a real system, this would:
        # - Send email/SMS notifications
        # - Post to webhook
        # - Store in database
        # - Trigger automated actions

        return {
            "status": "success",
            "alert": alert,
            "timestamp": alert["timestamp"],
            "sent": False,  # Would be True after actual notification sent
            "channels": []  # Would list notification channels used
        }

    except Exception as e:
        logger.error(f"Alert generation failed: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "alert": None
        }


async def check_alert_conditions(
    data: Dict[str, Any],
    conditions: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Check if data meets alert conditions

    Args:
        data: Data to check (e.g., sentiment scores, price changes)
        conditions: Alert conditions (e.g., {"sentiment_threshold": 0.5})

    Returns:
        Dict with condition check results
    """
    try:
        triggered = []

        # Check sentiment threshold
        if "sentiment" in data and "sentiment_threshold" in conditions:
            sentiment = data.get("sentiment", {}).get("polarity", 0)
            threshold = conditions["sentiment_threshold"]

            if abs(sentiment) > threshold:
                triggered.append({
                    "condition": "sentiment_threshold",
                    "value": sentiment,
                    "threshold": threshold,
                    "message": f"Sentiment {sentiment:.2f} exceeds threshold {threshold}"
                })

        # Check price change threshold
        if "price_change" in data and "price_threshold" in conditions:
            price_change = abs(data.get("price_change", 0))
            threshold = conditions["price_threshold"]

            if price_change > threshold:
                triggered.append({
                    "condition": "price_threshold",
                    "value": price_change,
                    "threshold": threshold,
                    "message": f"Price change {price_change:.2f}% exceeds threshold {threshold}%"
                })

        return {
            "status": "success",
            "triggered": len(triggered) > 0,
            "conditions_met": triggered,
            "count": len(triggered)
        }

    except Exception as e:
        logger.error(f"Condition check failed: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "triggered": False
        }
