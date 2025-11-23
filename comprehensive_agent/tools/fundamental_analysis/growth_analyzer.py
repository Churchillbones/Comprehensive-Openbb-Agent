"""
Growth Analyzer Tool

Analyze revenue, earnings, and growth trends (YoY, CAGR).
Extracted from: comprehensive_agent/core/ml_widget_bridge.py
NEW tool created for Fundamental Analysis Agent.
"""

from typing import Dict, Any, List, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)


async def analyze_growth(
    financial_data: Any,
    periods: int = 5
) -> Dict[str, Any]:
    """
    Analyze growth trends in financial data

    Args:
        financial_data: Historical financial data (multi-period)
        periods: Number of periods to analyze

    Returns:
        Dict with growth analysis:
            - status: "success" or "error"
            - revenue_growth: Revenue growth metrics
            - earnings_growth: Earnings growth metrics
            - cagr: Compound Annual Growth Rate for key metrics
            - growth_trend: "accelerating", "decelerating", or "stable"
            - insights: Human-readable insights
    """
    try:
        # Extract time series financial metrics
        metrics = _extract_time_series_metrics(financial_data, periods)

        if not metrics:
            return {
                "status": "error",
                "error": "Could not extract time series financial data"
            }

        result = {
            "status": "success",
            "revenue_growth": {},
            "earnings_growth": {},
            "cagr": {},
            "growth_trend": "stable",
            "insights": []
        }

        # Analyze revenue growth
        if "revenue" in metrics and len(metrics["revenue"]) >= 2:
            result["revenue_growth"] = _analyze_metric_growth(metrics["revenue"], "Revenue")

        # Analyze earnings growth
        if "net_income" in metrics and len(metrics["net_income"]) >= 2:
            result["earnings_growth"] = _analyze_metric_growth(metrics["net_income"], "Earnings")

        # Calculate CAGR for multi-year data
        if len(metrics.get("revenue", [])) >= 3:
            for metric_name, values in metrics.items():
                if len(values) >= 3:
                    cagr = _calculate_cagr(values)
                    if cagr is not None:
                        result["cagr"][metric_name] = cagr

        # Determine overall growth trend
        result["growth_trend"] = _determine_growth_trend(result)

        # Generate insights
        result["insights"] = _generate_growth_insights(result)

        logger.debug(f"Growth analysis complete: {len(result['insights'])} insights generated")

        return result

    except Exception as e:
        logger.error(f"Growth analysis failed: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e)
        }


def _extract_time_series_metrics(data: Any, periods: int) -> Dict[str, List[float]]:
    """Extract time series of financial metrics"""
    metrics = {}

    # Handle different data formats
    if isinstance(data, dict):
        # Check if data has historical periods
        if "periods" in data or "historical" in data:
            period_data = data.get("periods") or data.get("historical", [])

            if isinstance(period_data, list):
                # Extract metrics from each period
                metric_names = ["revenue", "net_income", "gross_profit", "operating_income", "ebitda"]

                for metric_name in metric_names:
                    values = []
                    for period in period_data[:periods]:
                        if isinstance(period, dict) and metric_name in period:
                            try:
                                values.append(float(period[metric_name]))
                            except (ValueError, TypeError):
                                pass

                    if values:
                        metrics[metric_name] = values

        # Or check if metrics are already lists
        else:
            for key, value in data.items():
                if isinstance(value, list) and all(isinstance(x, (int, float)) for x in value):
                    metrics[key] = [float(x) for x in value[:periods]]

    elif isinstance(data, list):
        # Assume list of period dicts
        metric_names = ["revenue", "net_income", "gross_profit", "operating_income"]

        for metric_name in metric_names:
            values = []
            for period in data[:periods]:
                if isinstance(period, dict) and metric_name in period:
                    try:
                        values.append(float(period[metric_name]))
                    except (ValueError, TypeError):
                        pass

            if values:
                metrics[metric_name] = values

    return metrics


def _analyze_metric_growth(values: List[float], metric_name: str) -> Dict[str, Any]:
    """Analyze growth for a single metric"""
    if len(values) < 2:
        return {}

    # Year-over-year growth (most recent period)
    yoy_growth = ((values[-1] / values[-2]) - 1) * 100 if values[-2] != 0 else None

    # Average growth rate
    growth_rates = []
    for i in range(1, len(values)):
        if values[i-1] != 0:
            rate = ((values[i] / values[i-1]) - 1) * 100
            growth_rates.append(rate)

    avg_growth = np.mean(growth_rates) if growth_rates else None

    # Growth volatility
    growth_volatility = np.std(growth_rates) if len(growth_rates) > 1 else None

    # Growth trend (accelerating/decelerating)
    trend = "stable"
    if len(growth_rates) >= 3:
        recent_avg = np.mean(growth_rates[-2:])
        earlier_avg = np.mean(growth_rates[:-2])

        if recent_avg > earlier_avg + 5:
            trend = "accelerating"
        elif recent_avg < earlier_avg - 5:
            trend = "decelerating"

    return {
        "metric": metric_name,
        "current_value": round(values[-1], 2),
        "previous_value": round(values[-2], 2) if len(values) >= 2 else None,
        "yoy_growth_pct": round(yoy_growth, 2) if yoy_growth is not None else None,
        "avg_growth_pct": round(avg_growth, 2) if avg_growth is not None else None,
        "volatility": round(growth_volatility, 2) if growth_volatility is not None else None,
        "trend": trend,
        "periods_analyzed": len(values)
    }


def _calculate_cagr(values: List[float]) -> Optional[float]:
    """Calculate Compound Annual Growth Rate"""
    if len(values) < 2 or values[0] == 0:
        return None

    n = len(values) - 1  # Number of years
    beginning = values[0]
    ending = values[-1]

    if beginning <= 0:
        return None

    cagr = ((ending / beginning) ** (1 / n) - 1) * 100
    return round(cagr, 2)


def _determine_growth_trend(result: Dict[str, Any]) -> str:
    """Determine overall growth trend"""
    trends = []

    if "revenue_growth" in result and result["revenue_growth"].get("trend"):
        trends.append(result["revenue_growth"]["trend"])

    if "earnings_growth" in result and result["earnings_growth"].get("trend"):
        trends.append(result["earnings_growth"]["trend"])

    if not trends:
        return "stable"

    # If majority are accelerating/decelerating
    if trends.count("accelerating") > len(trends) / 2:
        return "accelerating"
    elif trends.count("decelerating") > len(trends) / 2:
        return "decelerating"

    return "stable"


def _generate_growth_insights(result: Dict[str, Any]) -> List[str]:
    """Generate insights from growth analysis"""
    insights = []

    # Revenue growth insights
    if "revenue_growth" in result and result["revenue_growth"]:
        rev_growth = result["revenue_growth"]
        yoy = rev_growth.get("yoy_growth_pct")

        if yoy is not None:
            if yoy > 20:
                insights.append(f"📈 Strong revenue growth: {yoy:.1f}% YoY")
            elif yoy > 10:
                insights.append(f"📈 Solid revenue growth: {yoy:.1f}% YoY")
            elif yoy < 0:
                insights.append(f"📉 Revenue declining: {yoy:.1f}% YoY")
            else:
                insights.append(f"Revenue growth: {yoy:.1f}% YoY")

        if rev_growth.get("trend") == "accelerating":
            insights.append("🚀 Revenue growth accelerating")
        elif rev_growth.get("trend") == "decelerating":
            insights.append("⚠️ Revenue growth decelerating")

    # Earnings growth insights
    if "earnings_growth" in result and result["earnings_growth"]:
        earn_growth = result["earnings_growth"]
        yoy = earn_growth.get("yoy_growth_pct")

        if yoy is not None:
            if yoy > 25:
                insights.append(f"💰 Exceptional earnings growth: {yoy:.1f}% YoY")
            elif yoy < 0:
                insights.append(f"⚠️ Earnings declining: {yoy:.1f}% YoY")

    # CAGR insights
    if "cagr" in result and result["cagr"]:
        if "revenue" in result["cagr"]:
            cagr = result["cagr"]["revenue"]
            insights.append(f"📊 Revenue CAGR: {cagr:.1f}%")

        if "net_income" in result["cagr"]:
            cagr = result["cagr"]["net_income"]
            insights.append(f"💵 Earnings CAGR: {cagr:.1f}%")

    # Overall trend
    trend = result.get("growth_trend", "stable")
    if trend == "accelerating":
        insights.append("🎯 Overall growth trend is accelerating")
    elif trend == "decelerating":
        insights.append("⚠️ Overall growth trend is decelerating")

    if not insights:
        insights.append("Growth analysis completed")

    return insights
