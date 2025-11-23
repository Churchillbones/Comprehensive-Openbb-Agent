"""
Profitability Analyzer Tool

Analyze margins, profitability trends, and ROIC.
Extracted from: comprehensive_agent/core/ml_widget_bridge.py
NEW tool created for Fundamental Analysis Agent.
"""

from typing import Dict, Any, List, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)


async def analyze_profitability(
    financial_data: Any
) -> Dict[str, Any]:
    """
    Analyze profitability metrics and trends

    Args:
        financial_data: Financial statement data (single or multi-period)

    Returns:
        Dict with profitability analysis:
            - status: "success" or "error"
            - margins: Margin analysis (gross, operating, net)
            - returns: Return metrics (ROE, ROA, ROIC)
            - trends: Profitability trends over time
            - insights: Human-readable insights
    """
    try:
        # Extract financial metrics
        metrics = _extract_profitability_metrics(financial_data)

        if not metrics:
            return {
                "status": "error",
                "error": "Could not extract profitability metrics"
            }

        result = {
            "status": "success",
            "margins": {},
            "returns": {},
            "trends": {},
            "insights": []
        }

        # Calculate margins
        result["margins"] = _calculate_margins(metrics)

        # Calculate return metrics
        result["returns"] = _calculate_returns(metrics)

        # Analyze trends if multi-period data
        if _has_multi_period_data(metrics):
            result["trends"] = _analyze_profitability_trends(metrics)

        # Generate insights
        result["insights"] = _generate_profitability_insights(result)

        logger.debug(f"Profitability analysis complete: {len(result['insights'])} insights")

        return result

    except Exception as e:
        logger.error(f"Profitability analysis failed: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e)
        }


def _extract_profitability_metrics(data: Any) -> Dict[str, Any]:
    """Extract profitability-related metrics from data"""
    metrics = {}

    if isinstance(data, dict):
        # Single period data
        field_mappings = {
            "revenue": ["revenue", "sales", "total_revenue"],
            "gross_profit": ["gross_profit", "gross_income"],
            "operating_income": ["operating_income", "ebit", "operating_profit"],
            "net_income": ["net_income", "net_profit", "profit"],
            "total_assets": ["total_assets", "assets"],
            "total_equity": ["total_equity", "shareholders_equity", "equity"],
            "invested_capital": ["invested_capital"],
            "total_debt": ["total_debt", "debt"]
        }

        for metric_name, possible_keys in field_mappings.items():
            for key in possible_keys:
                if key in data:
                    try:
                        value = data[key]
                        # Check if it's a list (multi-period)
                        if isinstance(value, list):
                            metrics[metric_name] = [float(x) for x in value]
                        else:
                            metrics[metric_name] = float(value)
                        break
                    except (ValueError, TypeError):
                        pass

    return metrics


def _has_multi_period_data(metrics: Dict[str, Any]) -> bool:
    """Check if metrics contain multi-period data"""
    for value in metrics.values():
        if isinstance(value, list) and len(value) > 1:
            return True
    return False


def _calculate_margins(metrics: Dict[str, Any]) -> Dict[str, float]:
    """Calculate profit margins"""
    margins = {}

    # Get single values or latest values
    def get_value(key):
        val = metrics.get(key)
        if isinstance(val, list):
            return val[-1] if val else None
        return val

    revenue = get_value("revenue")
    gross_profit = get_value("gross_profit")
    operating_income = get_value("operating_income")
    net_income = get_value("net_income")

    if revenue and revenue != 0:
        # Gross Margin
        if gross_profit is not None:
            margins["gross_margin"] = round((gross_profit / revenue) * 100, 2)

        # Operating Margin
        if operating_income is not None:
            margins["operating_margin"] = round((operating_income / revenue) * 100, 2)

        # Net Margin
        if net_income is not None:
            margins["net_margin"] = round((net_income / revenue) * 100, 2)

    return margins


def _calculate_returns(metrics: Dict[str, Any]) -> Dict[str, float]:
    """Calculate return metrics"""
    returns = {}

    # Get single values or latest values
    def get_value(key):
        val = metrics.get(key)
        if isinstance(val, list):
            return val[-1] if val else None
        return val

    net_income = get_value("net_income")
    operating_income = get_value("operating_income")
    total_assets = get_value("total_assets")
    total_equity = get_value("total_equity")
    total_debt = get_value("total_debt")

    # ROE (Return on Equity)
    if net_income is not None and total_equity and total_equity != 0:
        returns["roe"] = round((net_income / total_equity) * 100, 2)

    # ROA (Return on Assets)
    if net_income is not None and total_assets and total_assets != 0:
        returns["roa"] = round((net_income / total_assets) * 100, 2)

    # ROIC (Return on Invested Capital)
    if operating_income is not None:
        invested_capital = get_value("invested_capital")

        if not invested_capital and total_equity and total_debt:
            invested_capital = total_equity + total_debt

        if invested_capital and invested_capital != 0:
            returns["roic"] = round((operating_income / invested_capital) * 100, 2)

    return returns


def _analyze_profitability_trends(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze profitability trends over time"""
    trends = {}

    # Analyze margin trends
    margin_metrics = ["gross_profit", "operating_income", "net_income"]

    for metric in margin_metrics:
        if metric in metrics and isinstance(metrics[metric], list):
            values = metrics[metric]
            revenue_values = metrics.get("revenue", [])

            if len(values) >= 2 and len(revenue_values) >= 2:
                # Calculate margins for each period
                margin_series = []
                for i in range(min(len(values), len(revenue_values))):
                    if revenue_values[i] != 0:
                        margin = (values[i] / revenue_values[i]) * 100
                        margin_series.append(margin)

                if len(margin_series) >= 2:
                    # Determine trend
                    recent_avg = np.mean(margin_series[-2:])
                    earlier_avg = np.mean(margin_series[:-2]) if len(margin_series) > 2 else margin_series[0]

                    trend_direction = "stable"
                    if recent_avg > earlier_avg + 1:
                        trend_direction = "improving"
                    elif recent_avg < earlier_avg - 1:
                        trend_direction = "declining"

                    metric_name = metric.replace("_", " ").title()
                    trends[f"{metric}_margin_trend"] = {
                        "direction": trend_direction,
                        "current": round(margin_series[-1], 2),
                        "previous": round(margin_series[-2], 2),
                        "change_pct": round(margin_series[-1] - margin_series[-2], 2)
                    }

    return trends


def _generate_profitability_insights(result: Dict[str, Any]) -> List[str]:
    """Generate insights from profitability analysis"""
    insights = []

    margins = result.get("margins", {})
    returns = result.get("returns", {})
    trends = result.get("trends", {})

    # Margin insights
    if "gross_margin" in margins:
        gm = margins["gross_margin"]
        if gm > 60:
            insights.append(f"💰 Excellent gross margin: {gm:.1f}%")
        elif gm < 30:
            insights.append(f"⚠️ Low gross margin: {gm:.1f}%")

    if "operating_margin" in margins:
        om = margins["operating_margin"]
        if om > 20:
            insights.append(f"💪 Strong operating margin: {om:.1f}%")
        elif om < 5:
            insights.append(f"📉 Weak operating margin: {om:.1f}%")

    if "net_margin" in margins:
        nm = margins["net_margin"]
        if nm > 15:
            insights.append(f"✅ Healthy net margin: {nm:.1f}%")
        elif nm < 5:
            insights.append(f"⚠️ Thin net margin: {nm:.1f}%")

    # Return metrics insights
    if "roe" in returns:
        roe = returns["roe"]
        if roe > 20:
            insights.append(f"🎯 Exceptional ROE: {roe:.1f}%")
        elif roe < 10:
            insights.append(f"📊 ROE below expectations: {roe:.1f}%")

    if "roic" in returns:
        roic = returns["roic"]
        if roic > 15:
            insights.append(f"💎 Superior capital efficiency (ROIC: {roic:.1f}%)")

    # Trend insights
    for metric_key, trend_data in trends.items():
        if "direction" in trend_data:
            direction = trend_data["direction"]
            if direction == "improving":
                metric_name = metric_key.replace("_margin_trend", "").replace("_", " ").title()
                insights.append(f"📈 {metric_name} margins improving")
            elif direction == "declining":
                metric_name = metric_key.replace("_margin_trend", "").replace("_", " ").title()
                insights.append(f"📉 {metric_name} margins declining")

    if not insights:
        insights.append("Profitability metrics analyzed")

    return insights
