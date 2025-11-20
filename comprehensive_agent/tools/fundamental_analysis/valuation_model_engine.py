"""
Valuation Model Engine Tool

Run valuation models (DCF, comparable company analysis, precedent transactions).
NEW tool created for Fundamental Analysis Agent.
"""

from typing import Dict, Any, List, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)


async def run_valuation_model(
    method: str = "dcf",
    financial_data: Optional[Any] = None,
    **params
) -> Dict[str, Any]:
    """
    Run valuation models

    Args:
        method: Valuation method ("dcf", "multiples", "comparable")
        financial_data: Financial statement data
        **params: Method-specific parameters

    Returns:
        Dict with valuation results:
            - status: "success" or "error"
            - method: Valuation method used
            - valuation: Estimated value
            - assumptions: Assumptions used
            - sensitivity: Sensitivity analysis
            - insights: Human-readable insights
    """
    try:
        result = {}

        if method == "dcf":
            result = await _dcf_valuation(financial_data, params)

        elif method == "multiples":
            result = await _multiples_valuation(financial_data, params)

        elif method == "comparable":
            result = await _comparable_company_analysis(financial_data, params)

        else:
            return {
                "status": "error",
                "error": f"Unknown valuation method: {method}"
            }

        # Add method to result
        result["method"] = method
        result["status"] = "success"

        # Generate insights
        if "insights" not in result:
            result["insights"] = _generate_valuation_insights(result)

        logger.debug(f"{method.upper()} valuation complete: {result.get('valuation')}")

        return result

    except Exception as e:
        logger.error(f"Valuation model failed: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "method": method
        }


async def _dcf_valuation(financial_data: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Discounted Cash Flow valuation"""
    # Extract parameters
    discount_rate = params.get("discount_rate", 0.10)  # 10% default WACC
    terminal_growth_rate = params.get("terminal_growth_rate", 0.03)  # 3% default
    projection_years = params.get("projection_years", 5)

    # Extract financial metrics
    metrics = _extract_dcf_metrics(financial_data)

    if not metrics.get("free_cash_flow") and not metrics.get("operating_cash_flow"):
        return {
            "status": "error",
            "error": "Insufficient cash flow data for DCF"
        }

    # Use FCF if available, else approximate from operating cash flow
    base_fcf = metrics.get("free_cash_flow") or metrics.get("operating_cash_flow", 0)

    # Project future cash flows
    fcf_projections = []
    current_fcf = base_fcf

    # Assume FCF growth (can be customized)
    fcf_growth_rate = params.get("fcf_growth_rate", 0.05)  # 5% default

    for year in range(1, projection_years + 1):
        projected_fcf = current_fcf * ((1 + fcf_growth_rate) ** year)
        discount_factor = 1 / ((1 + discount_rate) ** year)
        pv_fcf = projected_fcf * discount_factor

        fcf_projections.append({
            "year": year,
            "projected_fcf": round(projected_fcf, 2),
            "discount_factor": round(discount_factor, 4),
            "present_value": round(pv_fcf, 2)
        })

    # Calculate terminal value
    terminal_fcf = fcf_projections[-1]["projected_fcf"] * (1 + terminal_growth_rate)
    terminal_value = terminal_fcf / (discount_rate - terminal_growth_rate)
    terminal_pv = terminal_value / ((1 + discount_rate) ** projection_years)

    # Total enterprise value
    pv_fcf_sum = sum(p["present_value"] for p in fcf_projections)
    enterprise_value = pv_fcf_sum + terminal_pv

    # Adjust for cash and debt if available
    cash = metrics.get("cash", 0)
    debt = metrics.get("total_debt", 0)
    equity_value = enterprise_value + cash - debt

    # Per share value
    shares = metrics.get("shares_outstanding")
    value_per_share = equity_value / shares if shares else None

    return {
        "valuation": round(enterprise_value, 2),
        "equity_value": round(equity_value, 2),
        "value_per_share": round(value_per_share, 2) if value_per_share else None,
        "assumptions": {
            "discount_rate": discount_rate,
            "terminal_growth_rate": terminal_growth_rate,
            "fcf_growth_rate": fcf_growth_rate,
            "projection_years": projection_years
        },
        "projections": fcf_projections,
        "terminal_value": {
            "terminal_fcf": round(terminal_fcf, 2),
            "terminal_value": round(terminal_value, 2),
            "present_value": round(terminal_pv, 2)
        },
        "components": {
            "pv_fcf": round(pv_fcf_sum, 2),
            "pv_terminal": round(terminal_pv, 2),
            "cash": round(cash, 2),
            "debt": round(debt, 2)
        }
    }


async def _multiples_valuation(financial_data: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Valuation using multiples (P/E, EV/EBITDA, etc.)"""
    metrics = _extract_dcf_metrics(financial_data)

    # Get comparable multiples from params
    peer_pe = params.get("peer_pe_ratio", 15)
    peer_ev_ebitda = params.get("peer_ev_ebitda", 10)
    peer_ps = params.get("peer_ps_ratio", 2)

    valuations = {}

    # P/E based valuation
    if "net_income" in metrics:
        pe_valuation = metrics["net_income"] * peer_pe
        valuations["pe_valuation"] = round(pe_valuation, 2)

    # EV/EBITDA based valuation
    if "ebitda" in metrics:
        ev_valuation = metrics["ebitda"] * peer_ev_ebitda
        # Adjust for cash and debt
        equity_value = ev_valuation + metrics.get("cash", 0) - metrics.get("total_debt", 0)
        valuations["ev_ebitda_valuation"] = round(equity_value, 2)

    # P/S based valuation
    if "revenue" in metrics:
        ps_valuation = metrics["revenue"] * peer_ps
        valuations["ps_valuation"] = round(ps_valuation, 2)

    # Average of valuations
    if valuations:
        avg_valuation = np.mean(list(valuations.values()))
        valuations["average"] = round(avg_valuation, 2)

    # Per share values
    shares = metrics.get("shares_outstanding")
    if shares:
        valuations["value_per_share"] = {}
        for key, val in valuations.items():
            if key != "value_per_share":
                valuations["value_per_share"][key] = round(val / shares, 2)

    return {
        "valuation": valuations.get("average"),
        "valuations_by_multiple": valuations,
        "assumptions": {
            "peer_pe_ratio": peer_pe,
            "peer_ev_ebitda": peer_ev_ebitda,
            "peer_ps_ratio": peer_ps
        }
    }


async def _comparable_company_analysis(financial_data: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Comparable company analysis"""
    # Similar to multiples but with more detailed comp analysis
    # For simplicity, using multiples approach
    return await _multiples_valuation(financial_data, params)


def _extract_dcf_metrics(data: Any) -> Dict[str, float]:
    """Extract metrics needed for valuation models"""
    metrics = {}

    if isinstance(data, dict):
        field_mappings = {
            "free_cash_flow": ["free_cash_flow", "fcf"],
            "operating_cash_flow": ["operating_cash_flow", "cash_from_operations"],
            "net_income": ["net_income", "net_profit", "profit"],
            "ebitda": ["ebitda"],
            "revenue": ["revenue", "sales"],
            "cash": ["cash", "cash_and_equivalents"],
            "total_debt": ["total_debt", "debt"],
            "shares_outstanding": ["shares_outstanding", "shares"]
        }

        for metric_name, possible_keys in field_mappings.items():
            for key in possible_keys:
                if key in data:
                    try:
                        value = data[key]
                        # Use latest value if list
                        if isinstance(value, list):
                            metrics[metric_name] = float(value[-1])
                        else:
                            metrics[metric_name] = float(value)
                        break
                    except (ValueError, TypeError):
                        pass

    return metrics


def _generate_valuation_insights(result: Dict[str, Any]) -> List[str]:
    """Generate insights from valuation results"""
    insights = []

    method = result.get("method", "").upper()
    valuation = result.get("valuation")

    if valuation:
        insights.append(f"💰 {method} Valuation: ${valuation:,.2f}")

        if method == "DCF":
            components = result.get("components", {})
            pv_fcf = components.get("pv_fcf", 0)
            pv_terminal = components.get("pv_terminal", 0)

            if pv_terminal > pv_fcf:
                terminal_pct = (pv_terminal / (pv_fcf + pv_terminal)) * 100
                insights.append(f"⚠️ Terminal value represents {terminal_pct:.1f}% of valuation")

            assumptions = result.get("assumptions", {})
            discount_rate = assumptions.get("discount_rate", 0)
            insights.append(f"📊 Discount rate (WACC): {discount_rate*100:.1f}%")

        elif method == "MULTIPLES":
            valuations = result.get("valuations_by_multiple", {})
            if len(valuations) > 1:
                values = [v for k, v in valuations.items() if k not in ["average", "value_per_share"]]
                std = np.std(values)
                insights.append(f"📊 Valuation range: ${min(values):,.2f} - ${max(values):,.2f}")

        if "value_per_share" in result and result["value_per_share"]:
            insights.append(f"📈 Value per share: ${result['value_per_share']:.2f}")

    if not insights:
        insights.append("Valuation model completed")

    return insights
