"""
Fundamental Analysis Agent

Analyzes financial statements, calculates metrics, and performs valuations.
Part of Phase 2b-2 in the multi-agent orchestration architecture.
"""

from typing import Dict, Any, List, Optional
import logging
from comprehensive_agent.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class FundamentalAnalysisAgent(BaseAgent):
    """
    Specialized agent for fundamental analysis

    Capabilities:
    - Financial statement processing (Excel, CSV)
    - Financial metrics calculation (P/E, ROE, ROA, etc.)
    - Valuation models (DCF, comparable company analysis)
    - Growth analysis (revenue, earnings, CAGR)
    - Profitability analysis (margins, ROIC)
    - Financial table generation
    """

    def __init__(self):
        super().__init__(
            name="Fundamental Analysis Agent",
            description="Analyze financial statements, metrics, and valuations",
            capabilities=[
                "financial_statements",
                "metrics_calculation",
                "valuation_models",
                "growth_analysis",
                "profitability_analysis",
                "spreadsheet_processing",
                "financial_tables"
            ],
            priority=3
        )
        self._register_tools()

    def _register_tools(self):
        """Register all 7 fundamental analysis tools"""
        from comprehensive_agent.tools.fundamental_analysis import (
            process_spreadsheet,
            process_advanced_spreadsheet,
            calculate_financial_metrics,
            run_valuation_model,
            analyze_growth,
            analyze_profitability,
            generate_financial_table
        )

        self.register_tool("process_spreadsheet", process_spreadsheet)
        self.register_tool("process_advanced_spreadsheet", process_advanced_spreadsheet)
        self.register_tool("calculate_financial_metrics", calculate_financial_metrics)
        self.register_tool("run_valuation_model", run_valuation_model)
        self.register_tool("analyze_growth", analyze_growth)
        self.register_tool("analyze_profitability", analyze_profitability)
        self.register_tool("generate_financial_table", generate_financial_table)

        logger.info(f"Fundamental Analysis Agent initialized with {len(self.tools)} tools")

    def can_handle(self, intent: str, context: Dict[str, Any]) -> float:
        """
        Determine if this agent can handle the request

        Args:
            intent: The classified intent type
            context: Request context with widget_data, query, etc.

        Returns:
            Confidence score (0.0 to 1.0)
        """
        confidence = 0.0

        # Check intent type
        if intent == "fundamental_analysis":
            confidence = 0.95

        # Check for fundamental analysis keywords
        query = context.get("query", "").lower()

        # Financial statement keywords
        statement_keywords = [
            "income statement", "balance sheet", "cash flow", "financial statement",
            "10-k", "10-q", "earnings", "revenue", "profit", "assets", "liabilities"
        ]
        if any(keyword in query for keyword in statement_keywords):
            confidence = max(confidence, 0.90)

        # Metrics keywords
        metrics_keywords = [
            "p/e", "pe ratio", "price to earnings", "roe", "roa", "eps",
            "price to book", "p/b", "debt to equity", "current ratio",
            "quick ratio", "margin", "profitability"
        ]
        if any(keyword in query for keyword in metrics_keywords):
            confidence = max(confidence, 0.90)

        # Valuation keywords
        valuation_keywords = [
            "valuation", "dcf", "discounted cash flow", "intrinsic value",
            "fair value", "comparable", "multiples", "enterprise value", "ev/ebitda"
        ]
        if any(keyword in query for keyword in valuation_keywords):
            confidence = max(confidence, 0.85)

        # Growth keywords
        growth_keywords = ["growth", "cagr", "yoy", "year over year", "growth rate"]
        if any(keyword in query for keyword in growth_keywords):
            confidence = max(confidence, 0.80)

        # Spreadsheet/file keywords
        file_keywords = ["excel", "spreadsheet", "csv", "xlsx", "financial data"]
        if any(keyword in query for keyword in file_keywords):
            confidence = max(confidence, 0.75)

        # Check for uploaded files
        if context.get("uploaded_files"):
            for file_info in context["uploaded_files"]:
                filename = file_info.get("filename", "").lower()
                if any(ext in filename for ext in [".xlsx", ".xls", ".csv"]):
                    confidence = max(confidence, 0.85)

        # Widget type checks
        widget_type = context.get("widget_type", "")
        if widget_type in ["financial_statement", "spreadsheet", "table"]:
            confidence = max(confidence, 0.80)

        return confidence

    async def process(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process fundamental analysis request

        Args:
            query: User query
            context: Request context

        Returns:
            Processing results with financial analysis and insights
        """
        try:
            logger.info(f"Fundamental Analysis Agent processing: {query[:100]}")

            results = {
                "agent": self.name,
                "query": query,
                "status": "success",
                "financial_data": {},
                "metrics": {},
                "valuations": {},
                "analysis": {},
                "tables": [],
                "insights": []
            }

            query_lower = query.lower()

            # 1. Process spreadsheet files if available
            uploaded_files = context.get("uploaded_files", [])
            for file_info in uploaded_files:
                filename = file_info.get("filename", "").lower()
                if any(ext in filename for ext in [".xlsx", ".xls", ".csv"]):
                    # Use advanced processor for complex spreadsheets
                    spreadsheet_result = await self.tools["process_advanced_spreadsheet"](
                        file_data=file_info.get("data"),
                        filename=filename
                    )

                    if spreadsheet_result.get("status") == "success":
                        results["financial_data"]["spreadsheet"] = spreadsheet_result
                        results["insights"].append(f"Processed financial data from {filename}")

            # 2. Calculate financial metrics if we have financial data
            financial_data = results["financial_data"].get("spreadsheet", {}).get("data")
            if financial_data or context.get("financial_data"):
                data_to_analyze = financial_data or context.get("financial_data")

                metrics_result = await self.tools["calculate_financial_metrics"](
                    financial_data=data_to_analyze
                )

                if metrics_result.get("status") == "success":
                    results["metrics"] = metrics_result
                    results["insights"].extend(metrics_result.get("insights", []))

            # 3. Growth analysis if requested
            if any(kw in query_lower for kw in ["growth", "cagr", "yoy"]):
                growth_result = await self.tools["analyze_growth"](
                    financial_data=financial_data or context.get("financial_data")
                )

                if growth_result.get("status") == "success":
                    results["analysis"]["growth"] = growth_result
                    results["insights"].extend(growth_result.get("insights", []))

            # 4. Profitability analysis if requested
            if any(kw in query_lower for kw in ["profit", "margin", "roic", "roe", "roa"]):
                profitability_result = await self.tools["analyze_profitability"](
                    financial_data=financial_data or context.get("financial_data")
                )

                if profitability_result.get("status") == "success":
                    results["analysis"]["profitability"] = profitability_result
                    results["insights"].extend(profitability_result.get("insights", []))

            # 5. Valuation if requested
            if any(kw in query_lower for kw in ["valuation", "dcf", "fair value", "intrinsic"]):
                # Extract parameters from query or context
                valuation_params = {
                    "method": "dcf" if "dcf" in query_lower else "multiples",
                    "financial_data": financial_data or context.get("financial_data")
                }

                valuation_result = await self.tools["run_valuation_model"](
                    **valuation_params
                )

                if valuation_result.get("status") == "success":
                    results["valuations"] = valuation_result
                    results["insights"].extend(valuation_result.get("insights", []))

            # 6. Generate financial tables for visualization
            if results["metrics"] or results["financial_data"]:
                table_data = results["metrics"].get("data") or results["financial_data"]

                table_result = await self.tools["generate_financial_table"](
                    data=table_data,
                    table_type="metrics_summary"
                )

                if table_result.get("status") == "success":
                    results["tables"].append(table_result)

            # Generate summary
            results["summary"] = self._generate_summary(results)

            logger.info(f"Fundamental Analysis complete: {len(results['metrics'])} metrics, "
                       f"{len(results['tables'])} tables, {len(results['insights'])} insights")

            return results

        except Exception as e:
            logger.error(f"Fundamental Analysis Agent error: {e}", exc_info=True)
            return {
                "agent": self.name,
                "status": "error",
                "error": str(e),
                "metrics": {},
                "insights": []
            }

    def _generate_summary(self, results: Dict[str, Any]) -> str:
        """Generate a summary of the fundamental analysis"""
        parts = []

        if results.get("financial_data"):
            parts.append("Processed financial statements")

        if results.get("metrics"):
            metric_count = len(results["metrics"].get("ratios", {}))
            if metric_count > 0:
                parts.append(f"Calculated {metric_count} financial metrics")

        if results.get("analysis", {}).get("growth"):
            parts.append("Growth analysis completed")

        if results.get("analysis", {}).get("profitability"):
            parts.append("Profitability analysis completed")

        if results.get("valuations"):
            valuation_method = results["valuations"].get("method", "")
            if valuation_method:
                parts.append(f"{valuation_method.upper()} valuation completed")

        if results.get("insights"):
            parts.append(f"{len(results['insights'])} key insights")

        return " | ".join(parts) if parts else "Fundamental analysis completed"
