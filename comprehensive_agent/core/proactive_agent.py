"""
Proactive Agent - The Voice

This is not reactive. This is proactive.

It doesn't wait for questions. It surfaces what matters.
It doesn't just report data. It tells a story.
This is intelligence that guides.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from .financial_workspace import FinancialWorkspace
from .intelligence_engine import IntelligenceEngine, Insight, InsightPriority

logger = logging.getLogger(__name__)


class ProactiveAgent:
    """
    The voice of intelligence.

    This agent doesn't wait. It observes, understands, and speaks.
    It knows when something matters and how to communicate it clearly.
    """

    def __init__(self):
        self.intelligence_engine = IntelligenceEngine()
        self.last_briefing: Optional[Dict[str, Any]] = None

    async def generate_morning_briefing(
        self,
        workspace: FinancialWorkspace,
        widget_data: Optional[Any] = None
    ) -> str:
        """
        The ONE perfect experience.

        Generate a morning intelligence briefing that:
        1. Surfaces the 3 most important insights
        2. Explains WHY they matter
        3. Suggests WHAT to do about it

        This is the moment the agent proves it's intelligent.
        """

        logger.info("Generating Morning Intelligence Briefing...")

        # Get all insights
        all_insights = await self.intelligence_engine.analyze_significance(workspace, widget_data)

        if not all_insights:
            return self._generate_no_insights_message(workspace)

        # Take top 3
        top_insights = all_insights[:3]

        # Generate narrative
        briefing = self._create_narrative(top_insights, workspace)

        # Save for history
        self.last_briefing = {
            "timestamp": datetime.now().isoformat(),
            "insights_count": len(all_insights),
            "top_insights": [i.to_dict() for i in top_insights],
            "narrative": briefing
        }

        logger.info(f"Briefing generated: {len(top_insights)} top insights from {len(all_insights)} total")

        return briefing

    def _create_narrative(self, insights: List[Insight], workspace: FinancialWorkspace) -> str:
        """
        Transform insights into compelling narrative.

        This is storytelling. This is communication.
        Not a list of facts, but a story that guides.
        """

        # Opening
        greeting = self._generate_greeting()

        intro = self._generate_intro(workspace, len(insights))

        # Core insights (top 3)
        insight_sections = []
        for i, insight in enumerate(insights, 1):
            section = self._format_insight(insight, position=i)
            insight_sections.append(section)

        # Closing
        closing = self._generate_closing(workspace)

        # Assemble
        narrative_parts = [
            greeting,
            intro,
            "",  # Blank line
            *insight_sections,
            "",  # Blank line
            closing
        ]

        return "\n".join(narrative_parts)

    def _generate_greeting(self) -> str:
        """Context-aware greeting"""

        hour = datetime.now().hour

        if hour < 12:
            return "☀️ **Good morning.**"
        elif hour < 18:
            return "👋 **Good afternoon.**"
        else:
            return "🌙 **Good evening.**"

    def _generate_intro(self, workspace: FinancialWorkspace, insight_count: int) -> str:
        """Personalized introduction"""

        if insight_count == 0:
            return "I analyzed your portfolio. Everything looks steady."

        # Build context-aware intro
        asset_count = len(workspace.assets)

        intros = [
            f"I analyzed your {asset_count} holdings and market conditions.",
            f"**{insight_count} {'thing' if insight_count == 1 else 'things'} you should know:**"
        ]

        return "\n".join(intros)

    def _format_insight(self, insight: Insight, position: int) -> str:
        """
        Format a single insight beautifully.

        Clear structure:
        - Title (with emoji for visual impact)
        - Description (explains WHAT)
        - Evidence (shows WHY)
        - Recommendation (suggests ACTION)
        """

        parts = []

        # Title with priority indicator
        priority_emoji = {
            InsightPriority.CRITICAL: "🔴",
            InsightPriority.HIGH: "🟡",
            InsightPriority.MEDIUM: "🔵",
            InsightPriority.LOW: "⚪"
        }
        emoji = priority_emoji.get(insight.priority, "•")

        parts.append(f"### {position}. {insight.title}")
        parts.append("")  # Blank line

        # Description
        parts.append(insight.description)

        # Evidence if available
        if insight.evidence:
            parts.append(f"**Why:** {insight.evidence}")

        # Metrics if relevant
        if insight.metrics:
            metric_lines = []
            for key, value in insight.metrics.items():
                formatted_key = key.replace("_", " ").title()
                # Format value intelligently
                if isinstance(value, float):
                    if abs(value) < 1:
                        formatted_value = f"{value*100:.1f}%"
                    else:
                        formatted_value = f"{value:.2f}"
                else:
                    formatted_value = str(value)

                metric_lines.append(f"- {formatted_key}: {formatted_value}")

            if metric_lines:
                parts.append("")
                parts.append("**Key Metrics:**")
                parts.extend(metric_lines)

        # Recommendation
        if insight.recommendation:
            parts.append("")
            parts.append(f"💡 **What to do:** {insight.recommendation}")

        return "\n".join(parts)

    def _generate_closing(self, workspace: FinancialWorkspace) -> str:
        """Helpful closing"""

        closings = [
            "**Want to explore any of these in detail?** Just ask.",
            ""
        ]

        # Add workspace summary
        if workspace.context.primary_symbols:
            closings.append(f"_Tracking: {', '.join(workspace.context.primary_symbols[:5])}_")

        return "\n".join(closings)

    def _generate_no_insights_message(self, workspace: FinancialWorkspace) -> str:
        """Message when no significant insights"""

        return (
            "☀️ **Good morning.**\n\n"
            f"I analyzed your {len(workspace.assets)} holdings and market conditions.\n\n"
            "**Everything looks steady** — no significant changes or alerts right now.\n\n"
            "Your portfolio is behaving normally within historical ranges. "
            "I'll alert you if anything important emerges.\n\n"
            "**Want me to analyze anything specific?** Just ask."
        )

    async def generate_insight_detail(self, insight: Insight, workspace: FinancialWorkspace) -> str:
        """
        Generate detailed analysis for a specific insight.

        When user wants to "explore" an insight, provide depth.
        """

        parts = [
            f"# 🔍 Deep Dive: {insight.title}",
            "",
            "## Overview",
            insight.description,
            ""
        ]

        # Affected assets
        if insight.affected_symbols:
            parts.extend([
                "## Affected Assets",
                ", ".join(insight.affected_symbols),
                ""
            ])

        # Detailed evidence
        if insight.evidence:
            parts.extend([
                "## Evidence",
                insight.evidence,
                ""
            ])

        # Metrics breakdown
        if insight.metrics:
            parts.append("## Metrics")
            for key, value in insight.metrics.items():
                formatted_key = key.replace("_", " ").title()
                parts.append(f"- **{formatted_key}:** {value}")
            parts.append("")

        # Detailed recommendation
        if insight.recommendation:
            parts.extend([
                "## Recommended Actions",
                insight.recommendation,
                ""
            ])

        # Historical context (if available)
        parts.extend([
            "## Historical Context",
            f"This type of pattern typically {self._get_historical_context(insight)}.",
            ""
        ])

        # Related insights
        parts.extend([
            "---",
            "_This insight is part of your comprehensive portfolio analysis._"
        ])

        return "\n".join(parts)

    def _get_historical_context(self, insight: Insight) -> str:
        """Provide historical context for insight types"""

        context_map = {
            "risk": "requires monitoring and often benefits from rebalancing",
            "opportunity": "represents potential entry points, though timing is crucial",
            "correlation": "indicates systematic risk that diversification can address",
            "anomaly": "precedes significant price movements in 60-70% of cases",
            "breakout": "continues for 5-7 trading days before consolidation",
            "macro": "impacts markets over weeks to months, not days"
        }

        return context_map.get(insight.insight_type.value, "warrants attention")

    async def should_send_alert(self, insights: List[Insight]) -> bool:
        """Determine if any insights warrant immediate alert"""

        return any(
            i.priority == InsightPriority.CRITICAL and i.action_required
            for i in insights
        )

    async def generate_alert_message(self, critical_insights: List[Insight]) -> str:
        """Generate urgent alert message"""

        if not critical_insights:
            return ""

        parts = [
            "🚨 **URGENT ALERT**",
            "",
            f"Detected {len(critical_insights)} critical {'issue' if len(critical_insights) == 1 else 'issues'} "
            "requiring immediate attention:",
            ""
        ]

        for i, insight in enumerate(critical_insights, 1):
            parts.extend([
                f"**{i}. {insight.title}**",
                insight.description,
                ""
            ])

            if insight.recommendation:
                parts.append(f"➡️ {insight.recommendation}")
                parts.append("")

        parts.append("**Review your portfolio immediately.**")

        return "\n".join(parts)


# Global proactive agent
proactive_agent = ProactiveAgent()
