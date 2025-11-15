"""
Result Aggregator for OpenBB Comprehensive Agent

This module aggregates results from multiple agents into a unified response.
Handles conflict resolution, data merging, and formatting of combined outputs.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import logging


# Configure logging
logger = logging.getLogger(__name__)


class ResultAggregator:
    """
    Aggregate results from multiple agents into a unified response

    Handles:
    - Merging data from multiple agents
    - Conflict resolution
    - Formatting combined outputs
    - Generating unified metadata
    """

    def __init__(self):
        """Initialize the result aggregator"""
        logger.info("Result aggregator initialized")

    def aggregate(
        self,
        agent_results: List[Dict[str, Any]],
        merge_strategy: str = "combine"
    ) -> Dict[str, Any]:
        """
        Aggregate results from multiple agents

        Args:
            agent_results: List of results from different agents
                Each result dict contains:
                - agent: Agent name
                - result: Agent's output (or error info)
                - error: Error message if agent failed
            merge_strategy: How to merge results
                - "combine": Combine all results
                - "priority": Use highest priority result
                - "consensus": Use most common result

        Returns:
            Aggregated result dictionary with:
                - status: Overall status
                - data: Combined data
                - insights: Aggregated insights
                - visualizations: Charts and tables
                - metadata: Aggregation metadata
        """
        if not agent_results:
            return self._empty_result("No agent results to aggregate")

        # Separate successful and failed results
        successful = [r for r in agent_results if "error" not in r]
        failed = [r for r in agent_results if "error" in r]

        logger.info(
            f"Aggregating {len(agent_results)} results: "
            f"{len(successful)} successful, {len(failed)} failed"
        )

        # If all agents failed, return error
        if not successful:
            return self._error_result(
                "All agents failed to process the request",
                failed_agents=[r["agent"] for r in failed]
            )

        # Apply merge strategy
        if merge_strategy == "combine":
            return self._combine_results(successful, failed)
        elif merge_strategy == "priority":
            return self._priority_result(successful)
        elif merge_strategy == "consensus":
            return self._consensus_result(successful)
        else:
            logger.warning(f"Unknown merge strategy: {merge_strategy}, using 'combine'")
            return self._combine_results(successful, failed)

    def _combine_results(
        self,
        successful: List[Dict],
        failed: List[Dict]
    ) -> Dict[str, Any]:
        """
        Combine all successful results into one unified response

        Args:
            successful: List of successful agent results
            failed: List of failed agent results

        Returns:
            Combined result dictionary
        """
        combined = {
            "status": "success",
            "data": {},
            "insights": [],
            "visualizations": {
                "charts": [],
                "tables": []
            },
            "citations": [],
            "metadata": {
                "agents_used": [],
                "timestamp": datetime.now().isoformat(),
                "merge_strategy": "combine",
                "successful_agents": len(successful),
                "failed_agents": len(failed)
            }
        }

        # Process each successful result
        for agent_result in successful:
            agent_name = agent_result.get("agent", "Unknown")
            result_data = agent_result.get("result", {})

            # Track which agents were used
            combined["metadata"]["agents_used"].append(agent_name)

            # Merge data
            if isinstance(result_data, dict):
                data = result_data.get("data", {})
                if isinstance(data, dict):
                    combined["data"][agent_name] = data
                elif data:  # Non-empty data
                    combined["data"][agent_name] = data

                # Merge insights
                insights = result_data.get("insights", [])
                if insights:
                    if isinstance(insights, list):
                        combined["insights"].extend([
                            {"agent": agent_name, "insight": insight}
                            for insight in insights
                        ])
                    else:
                        combined["insights"].append({
                            "agent": agent_name,
                            "insight": insights
                        })

                # Merge visualizations
                visualizations = result_data.get("visualizations", {})
                if visualizations:
                    charts = visualizations.get("charts", [])
                    if charts:
                        combined["visualizations"]["charts"].extend(charts)

                    tables = visualizations.get("tables", [])
                    if tables:
                        combined["visualizations"]["tables"].extend(tables)

                # Merge citations
                citations = result_data.get("citations", [])
                if citations:
                    combined["citations"].extend(citations)

        # Add warnings about failed agents
        if failed:
            combined["metadata"]["warnings"] = [
                f"Agent '{r['agent']}' failed: {r.get('error', 'Unknown error')}"
                for r in failed
            ]

        logger.debug(
            f"Combined results from {len(successful)} agents: "
            f"{combined['metadata']['agents_used']}"
        )

        return combined

    def _priority_result(self, successful: List[Dict]) -> Dict[str, Any]:
        """
        Return the result from the highest priority agent

        Args:
            successful: List of successful agent results

        Returns:
            Result from highest priority agent
        """
        if not successful:
            return self._empty_result("No successful results to prioritize")

        # For now, just take the first result
        # In the future, we could use actual priority metadata
        best_result = successful[0]
        agent_name = best_result.get("agent", "Unknown")

        logger.debug(f"Using priority result from agent: {agent_name}")

        result_data = best_result.get("result", {})

        return {
            "status": "success",
            "data": result_data.get("data", {}),
            "insights": result_data.get("insights", []),
            "visualizations": result_data.get("visualizations", {}),
            "citations": result_data.get("citations", []),
            "metadata": {
                "agent_used": agent_name,
                "timestamp": datetime.now().isoformat(),
                "merge_strategy": "priority",
                "total_agents": len(successful)
            }
        }

    def _consensus_result(self, successful: List[Dict]) -> Dict[str, Any]:
        """
        Generate consensus result from multiple agents

        Currently just combines results, but could implement voting logic

        Args:
            successful: List of successful agent results

        Returns:
            Consensus result
        """
        logger.debug("Generating consensus from multiple agents")
        # For now, consensus is the same as combine
        # Could implement more sophisticated voting/agreement logic
        return self._combine_results(successful, [])

    def _empty_result(self, message: str) -> Dict[str, Any]:
        """
        Generate an empty result response

        Args:
            message: Message explaining why result is empty

        Returns:
            Empty result dictionary
        """
        return {
            "status": "empty",
            "message": message,
            "data": {},
            "insights": [],
            "visualizations": {"charts": [], "tables": []},
            "citations": [],
            "metadata": {
                "timestamp": datetime.now().isoformat()
            }
        }

    def _error_result(
        self,
        message: str,
        failed_agents: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate an error result response

        Args:
            message: Error message
            failed_agents: List of agents that failed

        Returns:
            Error result dictionary
        """
        return {
            "status": "error",
            "error": message,
            "data": {},
            "metadata": {
                "failed_agents": failed_agents or [],
                "timestamp": datetime.now().isoformat()
            }
        }

    def format_for_streaming(self, aggregated: Dict[str, Any]) -> List[Dict]:
        """
        Format aggregated result for SSE streaming

        Args:
            aggregated: Aggregated result dictionary

        Returns:
            List of event dictionaries for streaming
        """
        events = []

        # Send status update
        events.append({
            "type": "status",
            "data": {
                "status": aggregated.get("status"),
                "agents_used": aggregated.get("metadata", {}).get("agents_used", [])
            }
        })

        # Send insights
        insights = aggregated.get("insights", [])
        if insights:
            for insight_data in insights:
                events.append({
                    "type": "insight",
                    "data": insight_data
                })

        # Send visualizations
        visualizations = aggregated.get("visualizations", {})

        for chart in visualizations.get("charts", []):
            events.append({
                "type": "chart",
                "data": chart
            })

        for table in visualizations.get("tables", []):
            events.append({
                "type": "table",
                "data": table
            })

        # Send citations
        citations = aggregated.get("citations", [])
        if citations:
            events.append({
                "type": "citations",
                "data": citations
            })

        # Send final data
        events.append({
            "type": "data",
            "data": aggregated.get("data", {})
        })

        # Send completion
        events.append({
            "type": "complete",
            "data": {
                "metadata": aggregated.get("metadata", {})
            }
        })

        return events

    def summarize_results(self, aggregated: Dict[str, Any]) -> str:
        """
        Generate a text summary of aggregated results

        Args:
            aggregated: Aggregated result dictionary

        Returns:
            Human-readable summary string
        """
        status = aggregated.get("status", "unknown")
        metadata = aggregated.get("metadata", {})
        agents_used = metadata.get("agents_used", [])

        summary_parts = []

        # Status summary
        if status == "success":
            summary_parts.append(
                f"Successfully processed request using {len(agents_used)} agent(s): "
                f"{', '.join(agents_used)}"
            )
        elif status == "error":
            error_msg = aggregated.get("error", "Unknown error")
            summary_parts.append(f"Error: {error_msg}")
        elif status == "empty":
            message = aggregated.get("message", "No results")
            summary_parts.append(message)

        # Data summary
        data = aggregated.get("data", {})
        if data:
            summary_parts.append(f"Retrieved data from {len(data)} source(s)")

        # Insights summary
        insights = aggregated.get("insights", [])
        if insights:
            summary_parts.append(f"Generated {len(insights)} insight(s)")

        # Visualizations summary
        visualizations = aggregated.get("visualizations", {})
        chart_count = len(visualizations.get("charts", []))
        table_count = len(visualizations.get("tables", []))

        if chart_count or table_count:
            viz_parts = []
            if chart_count:
                viz_parts.append(f"{chart_count} chart(s)")
            if table_count:
                viz_parts.append(f"{table_count} table(s)")
            summary_parts.append(f"Created {', '.join(viz_parts)}")

        # Warnings
        warnings = metadata.get("warnings", [])
        if warnings:
            summary_parts.append(f"Warnings: {'; '.join(warnings)}")

        return ". ".join(summary_parts) + "."

    def __repr__(self) -> str:
        """String representation of the aggregator"""
        return "<ResultAggregator>"
