"""
Financial Metrics Calculator Tool

Calculate key financial ratios and metrics (P/E, ROE, ROA, etc.).
NEW tool created for Fundamental Analysis Agent.
"""

from typing import Dict, Any, List, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)


async def calculate_financial_metrics(
    financial_data: Any,
    current_price: Optional[float] = None
) -> Dict[str, Any]:
    """
    Calculate financial metrics and ratios

    Args:
        financial_data: Financial statement data (dict or structured data)
        current_price: Current stock price (for market-based ratios)

    Returns:
        Dict with calculated metrics:
            - status: "success" or "error"
            - ratios: Dict of calculated ratios
            - profitability: Profitability metrics
            - liquidity: Liquidity ratios
            - leverage: Leverage/solvency ratios
            - valuation: Valuation ratios (if price provided)
            - insights: Human-readable insights
    """
    try:
        if not financial_data:
            return {
                "status": "error",
                "error": "No financial data provided"
            }

        # Extract financial statement items
        metrics = _extract_financial_items(financial_data)

        if not metrics:
            return {
                "status": "error",
                "error": "Could not extract financial statement items"
            }

        result = {
            "status": "success",
            "ratios": {},
            "profitability": {},
            "liquidity": {},
            "leverage": {},
            "valuation": {},
            "insights": []
        }

        # Calculate profitability ratios
        result["profitability"] = _calculate_profitability_ratios(metrics)

        # Calculate liquidity ratios
        result["liquidity"] = _calculate_liquidity_ratios(metrics)

        # Calculate leverage ratios
        result["leverage"] = _calculate_leverage_ratios(metrics)

        # Calculate valuation ratios (if price available)
        if current_price and metrics.get("shares_outstanding"):
            result["valuation"] = _calculate_valuation_ratios(metrics, current_price)

        # Combine all ratios
        result["ratios"] = {
            **result["profitability"],
            **result["liquidity"],
            **result["leverage"],
            **result["valuation"]
        }

        # Generate insights
        result["insights"] = _generate_metrics_insights(result)

        logger.debug(f"Calculated {len(result['ratios'])} financial metrics")

        return result

    except Exception as e:
        logger.error(f"Financial metrics calculation failed: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e)
        }


def _extract_financial_items(data: Any) -> Dict[str, float]:
    """Extract financial statement line items from data"""
    metrics = {}

    # Handle different data formats
    if isinstance(data, dict):
        # Direct mapping of common financial items
        field_mappings = {
            # Income statement
            "revenue": ["revenue", "sales", "total_revenue", "net_sales"],
            "gross_profit": ["gross_profit", "gross_income"],
            "operating_income": ["operating_income", "ebit", "operating_profit"],
            "net_income": ["net_income", "net_profit", "net_earnings", "profit"],
            "ebitda": ["ebitda"],
            "cost_of_revenue": ["cost_of_revenue", "cogs", "cost_of_goods_sold"],
            "operating_expenses": ["operating_expenses", "opex"],

            # Balance sheet
            "total_assets": ["total_assets", "assets"],
            "current_assets": ["current_assets"],
            "total_liabilities": ["total_liabilities", "liabilities"],
            "current_liabilities": ["current_liabilities"],
            "total_equity": ["total_equity", "shareholders_equity", "equity"],
            "total_debt": ["total_debt", "debt"],
            "long_term_debt": ["long_term_debt", "lt_debt"],
            "cash": ["cash", "cash_and_equivalents"],
            "inventory": ["inventory"],
            "accounts_receivable": ["accounts_receivable", "receivables"],

            # Other
            "shares_outstanding": ["shares_outstanding", "shares", "outstanding_shares"]
        }

        for metric_name, possible_keys in field_mappings.items():
            for key in possible_keys:
                if key in data:
                    try:
                        metrics[metric_name] = float(data[key])
                        break
                    except (ValueError, TypeError):
                        pass

    return metrics


def _calculate_profitability_ratios(metrics: Dict[str, float]) -> Dict[str, float]:
    """Calculate profitability ratios"""
    ratios = {}

    # Gross Margin
    if "gross_profit" in metrics and "revenue" in metrics and metrics["revenue"] != 0:
        ratios["gross_margin"] = round((metrics["gross_profit"] / metrics["revenue"]) * 100, 2)

    # Operating Margin
    if "operating_income" in metrics and "revenue" in metrics and metrics["revenue"] != 0:
        ratios["operating_margin"] = round((metrics["operating_income"] / metrics["revenue"]) * 100, 2)

    # Net Margin
    if "net_income" in metrics and "revenue" in metrics and metrics["revenue"] != 0:
        ratios["net_margin"] = round((metrics["net_income"] / metrics["revenue"]) * 100, 2)

    # ROE (Return on Equity)
    if "net_income" in metrics and "total_equity" in metrics and metrics["total_equity"] != 0:
        ratios["roe"] = round((metrics["net_income"] / metrics["total_equity"]) * 100, 2)

    # ROA (Return on Assets)
    if "net_income" in metrics and "total_assets" in metrics and metrics["total_assets"] != 0:
        ratios["roa"] = round((metrics["net_income"] / metrics["total_assets"]) * 100, 2)

    # ROIC (Return on Invested Capital)
    if "operating_income" in metrics and "total_equity" in metrics and "total_debt" in metrics:
        invested_capital = metrics["total_equity"] + metrics["total_debt"]
        if invested_capital != 0:
            ratios["roic"] = round((metrics["operating_income"] / invested_capital) * 100, 2)

    return ratios


def _calculate_liquidity_ratios(metrics: Dict[str, float]) -> Dict[str, float]:
    """Calculate liquidity ratios"""
    ratios = {}

    # Current Ratio
    if "current_assets" in metrics and "current_liabilities" in metrics and metrics["current_liabilities"] != 0:
        ratios["current_ratio"] = round(metrics["current_assets"] / metrics["current_liabilities"], 2)

    # Quick Ratio (assuming inventory exists)
    if all(k in metrics for k in ["current_assets", "inventory", "current_liabilities"]) and metrics["current_liabilities"] != 0:
        quick_assets = metrics["current_assets"] - metrics["inventory"]
        ratios["quick_ratio"] = round(quick_assets / metrics["current_liabilities"], 2)

    # Cash Ratio
    if "cash" in metrics and "current_liabilities" in metrics and metrics["current_liabilities"] != 0:
        ratios["cash_ratio"] = round(metrics["cash"] / metrics["current_liabilities"], 2)

    return ratios


def _calculate_leverage_ratios(metrics: Dict[str, float]) -> Dict[str, float]:
    """Calculate leverage/solvency ratios"""
    ratios = {}

    # Debt to Equity
    if "total_debt" in metrics and "total_equity" in metrics and metrics["total_equity"] != 0:
        ratios["debt_to_equity"] = round(metrics["total_debt"] / metrics["total_equity"], 2)

    # Debt to Assets
    if "total_debt" in metrics and "total_assets" in metrics and metrics["total_assets"] != 0:
        ratios["debt_to_assets"] = round(metrics["total_debt"] / metrics["total_assets"], 2)

    # Equity Multiplier
    if "total_assets" in metrics and "total_equity" in metrics and metrics["total_equity"] != 0:
        ratios["equity_multiplier"] = round(metrics["total_assets"] / metrics["total_equity"], 2)

    return ratios


def _calculate_valuation_ratios(metrics: Dict[str, float], price: float) -> Dict[str, float]:
    """Calculate valuation ratios"""
    ratios = {}

    shares = metrics.get("shares_outstanding", 0)
    if shares == 0:
        return ratios

    market_cap = price * shares

    # P/E Ratio
    if "net_income" in metrics and metrics["net_income"] != 0:
        eps = metrics["net_income"] / shares
        ratios["pe_ratio"] = round(price / eps, 2)
        ratios["eps"] = round(eps, 2)

    # P/B Ratio
    if "total_equity" in metrics and metrics["total_equity"] != 0:
        book_value_per_share = metrics["total_equity"] / shares
        ratios["pb_ratio"] = round(price / book_value_per_share, 2)
        ratios["book_value_per_share"] = round(book_value_per_share, 2)

    # P/S Ratio
    if "revenue" in metrics and metrics["revenue"] != 0:
        revenue_per_share = metrics["revenue"] / shares
        ratios["ps_ratio"] = round(price / revenue_per_share, 2)

    # EV/EBITDA
    if "ebitda" in metrics and "total_debt" in metrics and "cash" in metrics and metrics["ebitda"] != 0:
        enterprise_value = market_cap + metrics["total_debt"] - metrics["cash"]
        ratios["ev_ebitda"] = round(enterprise_value / metrics["ebitda"], 2)
        ratios["enterprise_value"] = round(enterprise_value, 2)

    return ratios


def _generate_metrics_insights(result: Dict[str, Any]) -> List[str]:
    """Generate insights from calculated metrics"""
    insights = []

    profitability = result.get("profitability", {})
    liquidity = result.get("liquidity", {})
    leverage = result.get("leverage", {})
    valuation = result.get("valuation", {})

    # Profitability insights
    if "net_margin" in profitability:
        margin = profitability["net_margin"]
        if margin > 20:
            insights.append(f"💰 Strong net profit margin: {margin:.1f}%")
        elif margin < 5:
            insights.append(f"⚠️ Low net profit margin: {margin:.1f}%")

    if "roe" in profitability:
        roe = profitability["roe"]
        if roe > 15:
            insights.append(f"📈 Excellent ROE: {roe:.1f}%")
        elif roe < 5:
            insights.append(f"📉 Low ROE: {roe:.1f}%")

    # Liquidity insights
    if "current_ratio" in liquidity:
        ratio = liquidity["current_ratio"]
        if ratio > 2:
            insights.append(f"✅ Strong liquidity position (Current Ratio: {ratio:.2f})")
        elif ratio < 1:
            insights.append(f"⚠️ Liquidity concerns (Current Ratio: {ratio:.2f})")

    # Leverage insights
    if "debt_to_equity" in leverage:
        de_ratio = leverage["debt_to_equity"]
        if de_ratio > 2:
            insights.append(f"⚠️ High leverage (D/E: {de_ratio:.2f})")
        elif de_ratio < 0.5:
            insights.append(f"💪 Conservative capital structure (D/E: {de_ratio:.2f})")

    # Valuation insights
    if "pe_ratio" in valuation:
        pe = valuation["pe_ratio"]
        if pe > 30:
            insights.append(f"📊 Premium valuation (P/E: {pe:.1f})")
        elif pe < 10 and pe > 0:
            insights.append(f"📊 Value opportunity? (P/E: {pe:.1f})")

    if not insights:
        insights.append("Financial metrics calculated successfully")

    return insights
