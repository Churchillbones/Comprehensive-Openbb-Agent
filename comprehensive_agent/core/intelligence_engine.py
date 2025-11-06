"""
Intelligence Engine - The Brain

This doesn't just process data. It understands what matters.

It sees patterns. It detects significance. It knows when to speak.
This is the difference between analysis and intelligence.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

import numpy as np

from .financial_workspace import FinancialWorkspace, Asset, InvestmentStyle

logger = logging.getLogger(__name__)


class InsightType(Enum):
    """Categories of insights"""
    RISK = "risk"                      # Risk alerts
    OPPORTUNITY = "opportunity"        # Potential opportunities
    CORRELATION = "correlation"        # Correlation patterns
    ANOMALY = "anomaly"               # Unusual patterns
    BREAKOUT = "breakout"             # Technical breakouts
    VALUATION = "valuation"           # Valuation concerns
    MACRO = "macro"                   # Macro events
    NEWS = "news"                     # News impact


class InsightPriority(Enum):
    """How urgent/important"""
    CRITICAL = "critical"    # Act now
    HIGH = "high"           # Very important
    MEDIUM = "medium"       # Worth knowing
    LOW = "low"             # FYI


@dataclass
class Insight:
    """A single piece of intelligence"""

    # What
    title: str
    description: str
    insight_type: InsightType
    priority: InsightPriority

    # Evidence
    affected_symbols: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    evidence: str = ""

    # Action
    recommendation: Optional[str] = None
    action_required: bool = False

    # Metadata
    confidence: float = 0.8  # How confident we are
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "title": self.title,
            "description": self.description,
            "type": self.insight_type.value,
            "priority": self.priority.value,
            "affected_symbols": self.affected_symbols,
            "metrics": self.metrics,
            "evidence": self.evidence,
            "recommendation": self.recommendation,
            "action_required": self.action_required,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat()
        }


class IntelligenceEngine:
    """
    The system that knows what matters.

    Not every change is significant.
    Not every pattern is actionable.
    This engine separates signal from noise.
    """

    def __init__(self):
        self.thresholds = {
            "correlation_risk": 0.7,          # Assets correlated above this
            "concentration_risk": 0.30,       # Single position above this %
            "volatility_spike": 1.5,          # Volatility above X times normal
            "price_change_significant": 0.05,  # 5% price move
            "sentiment_extreme": 0.7,          # Strong sentiment score
            "news_volume_spike": 2.0,          # News volume above X times normal
        }

    async def analyze_significance(
        self,
        workspace: FinancialWorkspace,
        widget_data: Optional[Any] = None
    ) -> List[Insight]:
        """
        The core intelligence function.

        Given the workspace state, what matters right now?
        Returns insights ranked by importance.
        """

        logger.info("Intelligence Engine: Analyzing significance...")

        insights: List[Insight] = []

        # 1. Risk Detection
        insights.extend(await self._detect_risks(workspace))

        # 2. Opportunity Detection
        insights.extend(await self._detect_opportunities(workspace))

        # 3. Anomaly Detection
        insights.extend(await self._detect_anomalies(workspace))

        # 4. Correlation Analysis
        insights.extend(await self._analyze_correlations(workspace))

        # 5. Change Analysis
        changes = workspace.get_changes_since_last()
        insights.extend(await self._analyze_changes(changes, workspace))

        # Rank by priority and confidence
        ranked_insights = self._rank_insights(insights, workspace)

        logger.info(f"Generated {len(ranked_insights)} insights ({len([i for i in ranked_insights if i.priority == InsightPriority.CRITICAL])} critical)")

        return ranked_insights

    async def _detect_risks(self, workspace: FinancialWorkspace) -> List[Insight]:
        """Detect portfolio risks"""

        risks = []

        if not workspace.portfolio:
            return risks

        # Concentration Risk
        if workspace.portfolio.largest_position_pct > self.thresholds["concentration_risk"]:
            risks.append(Insight(
                title="🔴 Concentration Risk Detected",
                description=(
                    f"Your largest position represents {workspace.portfolio.largest_position_pct*100:.1f}% "
                    f"of portfolio value. High concentration means a single stock's performance "
                    f"dominates your returns."
                ),
                insight_type=InsightType.RISK,
                priority=InsightPriority.HIGH,
                metrics={
                    "largest_position_pct": workspace.portfolio.largest_position_pct,
                    "top_5_concentration": workspace.portfolio.top_5_concentration
                },
                recommendation="Consider reducing largest position to 15-20% through gradual trimming",
                action_required=True,
                confidence=0.9
            ))

        # Correlation Risk
        if workspace.portfolio.average_correlation and workspace.portfolio.average_correlation > self.thresholds["correlation_risk"]:

            # Find which assets are highly correlated
            correlated_assets = self._find_correlated_assets(workspace)

            if correlated_assets:
                risks.append(Insight(
                    title="🔗 High Correlation Risk",
                    description=(
                        f"Your holdings move together (avg correlation: {workspace.portfolio.average_correlation:.2f}). "
                        f"When one drops, they all tend to drop. This reduces diversification benefits."
                    ),
                    insight_type=InsightType.CORRELATION,
                    priority=InsightPriority.HIGH,
                    affected_symbols=correlated_assets,
                    metrics={"average_correlation": workspace.portfolio.average_correlation},
                    evidence=f"Assets {', '.join(correlated_assets[:3])} are highly correlated",
                    recommendation="Add uncorrelated assets (bonds, commodities, international) to reduce systematic risk",
                    action_required=True,
                    confidence=0.85
                ))

        # Volatility Risk
        if workspace.portfolio.portfolio_volatility:
            if workspace.portfolio.portfolio_volatility > 0.25:  # >25% annualized vol
                risks.append(Insight(
                    title="⚠️ High Portfolio Volatility",
                    description=(
                        f"Your portfolio volatility is {workspace.portfolio.portfolio_volatility*100:.1f}% annualized. "
                        f"Expect significant swings in value."
                    ),
                    insight_type=InsightType.RISK,
                    priority=InsightPriority.MEDIUM,
                    metrics={"volatility": workspace.portfolio.portfolio_volatility},
                    recommendation="Consider adding lower-volatility assets or reducing position sizes",
                    confidence=0.8
                ))

        return risks

    async def _detect_opportunities(self, workspace: FinancialWorkspace) -> List[Insight]:
        """Detect potential opportunities"""

        opportunities = []

        for symbol, asset in workspace.assets.items():
            # Technical Breakout
            if self._is_technical_breakout(asset):
                opportunities.append(Insight(
                    title=f"📈 {symbol} Technical Breakout",
                    description=(
                        f"{symbol} price crossed above key technical level with strong volume. "
                        f"This pattern historically continues for 5-7 days."
                    ),
                    insight_type=InsightType.BREAKOUT,
                    priority=InsightPriority.MEDIUM,
                    affected_symbols=[symbol],
                    evidence="Price above 200-day MA, volume 3x average",
                    recommendation=f"Monitor {symbol} for continuation, set stop loss below breakout level",
                    confidence=0.7
                ))

            # Sentiment Opportunity
            if asset.sentiment_score and asset.sentiment_score < -0.7:
                # Extremely negative sentiment could be oversold
                opportunities.append(Insight(
                    title=f"💡 {symbol} Potential Oversold",
                    description=(
                        f"News sentiment for {symbol} is extremely negative ({asset.sentiment_score:.2f}). "
                        f"This could present a contrarian opportunity if fundamentals remain intact."
                    ),
                    insight_type=InsightType.OPPORTUNITY,
                    priority=InsightPriority.LOW,
                    affected_symbols=[symbol],
                    metrics={"sentiment_score": asset.sentiment_score},
                    recommendation=f"Review {symbol} fundamentals to see if negativity is overdone",
                    confidence=0.6
                ))

        return opportunities

    async def _detect_anomalies(self, workspace: FinancialWorkspace) -> List[Insight]:
        """Detect unusual patterns"""

        anomalies = []

        for symbol, asset in workspace.assets.items():
            # Volume Spike
            if asset.volume_history and len(asset.volume_history) >= 20:
                recent_volume = asset.volume_history[-1] if asset.volume_history else 0
                avg_volume = np.mean(asset.volume_history[-20:]) if len(asset.volume_history) >= 20 else recent_volume

                if avg_volume > 0 and recent_volume > avg_volume * self.thresholds["news_volume_spike"]:
                    anomalies.append(Insight(
                        title=f"🔔 {symbol} Unusual Volume",
                        description=(
                            f"{symbol} trading volume is {recent_volume/avg_volume:.1f}x above average. "
                            f"Investigate potential news or events driving activity."
                        ),
                        insight_type=InsightType.ANOMALY,
                        priority=InsightPriority.MEDIUM,
                        affected_symbols=[symbol],
                        metrics={"volume_ratio": recent_volume/avg_volume},
                        recommendation=f"Check news sources for {symbol} - unusual volume often precedes significant moves",
                        confidence=0.75
                    ))

            # News Volume Spike
            if asset.news_count > 20:  # Arbitrary threshold
                anomalies.append(Insight(
                    title=f"📰 {symbol} High Media Attention",
                    description=(
                        f"{symbol} has {asset.news_count} news articles. "
                        f"High media coverage increases volatility."
                    ),
                    insight_type=InsightType.NEWS,
                    priority=InsightPriority.LOW,
                    affected_symbols=[symbol],
                    metrics={"news_count": asset.news_count},
                    confidence=0.7
                ))

        return anomalies

    async def _analyze_correlations(self, workspace: FinancialWorkspace) -> List[Insight]:
        """Deep correlation analysis"""

        insights = []

        # Check sector concentration
        if workspace.portfolio and workspace.portfolio.sector_allocation:
            largest_sector = max(
                workspace.portfolio.sector_allocation.items(),
                key=lambda x: x[1]
            ) if workspace.portfolio.sector_allocation else None

            if largest_sector and largest_sector[1] > 0.5:  # >50% in one sector
                insights.append(Insight(
                    title=f"🎯 Sector Concentration: {largest_sector[0]}",
                    description=(
                        f"{largest_sector[1]*100:.0f}% of your portfolio is in {largest_sector[0]} sector. "
                        f"Sector-specific events could have outsized impact."
                    ),
                    insight_type=InsightType.RISK,
                    priority=InsightPriority.MEDIUM,
                    metrics={"sector_concentration": largest_sector[1]},
                    recommendation=f"Diversify beyond {largest_sector[0]} into uncorrelated sectors",
                    confidence=0.85
                ))

        return insights

    async def _analyze_changes(self, changes: Dict[str, Any], workspace: FinancialWorkspace) -> List[Insight]:
        """Analyze what changed significantly"""

        insights = []

        # First analysis
        if changes.get("first_analysis"):
            return insights

        # New assets added
        if changes.get("new_assets"):
            for symbol in changes["new_assets"]:
                insights.append(Insight(
                    title=f"➕ New Asset Added: {symbol}",
                    description=f"Started tracking {symbol} in workspace",
                    insight_type=InsightType.ANOMALY,
                    priority=InsightPriority.LOW,
                    affected_symbols=[symbol],
                    confidence=1.0
                ))

        # Significant price changes
        for symbol, change_pct in changes.get("price_changes", {}).items():
            if abs(change_pct) > self.thresholds["price_change_significant"]:
                direction = "up" if change_pct > 0 else "down"
                priority = InsightPriority.HIGH if abs(change_pct) > 0.10 else InsightPriority.MEDIUM

                insights.append(Insight(
                    title=f"{'📈' if change_pct > 0 else '📉'} {symbol} Price Movement",
                    description=(
                        f"{symbol} moved {direction} {abs(change_pct)*100:.1f}% since last check. "
                        f"{'Strong gain' if change_pct > 0 else 'Notable decline'} worth investigating."
                    ),
                    insight_type=InsightType.ANOMALY,
                    priority=priority,
                    affected_symbols=[symbol],
                    metrics={"price_change_pct": change_pct},
                    recommendation=f"Review what drove {symbol}'s {'gain' if change_pct > 0 else 'decline'}",
                    confidence=0.9
                ))

        return insights

    def _rank_insights(self, insights: List[Insight], workspace: FinancialWorkspace) -> List[Insight]:
        """
        Rank insights by importance to THIS user.

        Not all insights matter equally. Prioritize based on:
        1. Priority level
        2. Confidence
        3. Relevance to user's style
        4. Actionability
        """

        def score_insight(insight: Insight) -> float:
            """Calculate importance score"""

            # Base score from priority
            priority_scores = {
                InsightPriority.CRITICAL: 100,
                InsightPriority.HIGH: 75,
                InsightPriority.MEDIUM: 50,
                InsightPriority.LOW: 25
            }
            score = priority_scores.get(insight.priority, 50)

            # Boost by confidence
            score *= insight.confidence

            # Boost for actionable insights
            if insight.action_required:
                score *= 1.2

            # Adjust for user's investment style
            if workspace.context.investment_style == InvestmentStyle.TRADER:
                # Traders care more about breakouts and anomalies
                if insight.insight_type in [InsightType.BREAKOUT, InsightType.ANOMALY]:
                    score *= 1.3
            elif workspace.context.investment_style == InvestmentStyle.INVESTOR:
                # Investors care more about valuation and fundamentals
                if insight.insight_type in [InsightType.VALUATION, InsightType.RISK]:
                    score *= 1.3

            return score

        # Score and sort
        scored_insights = [(score_insight(i), i) for i in insights]
        scored_insights.sort(key=lambda x: x[0], reverse=True)

        return [i for _, i in scored_insights]

    def _find_correlated_assets(self, workspace: FinancialWorkspace) -> List[str]:
        """Find which assets are highly correlated"""

        # Placeholder - would use actual correlation matrix
        # For now, simple heuristic: stocks in same sector

        tech_stocks = []
        for symbol, asset in workspace.assets.items():
            if asset.sector and "tech" in asset.sector.lower():
                tech_stocks.append(symbol)

        return tech_stocks[:5] if len(tech_stocks) > 3 else []

    def _is_technical_breakout(self, asset: Asset) -> bool:
        """Detect if asset is in technical breakout"""

        # Placeholder - would use actual technical analysis
        # For now, simple check
        if not asset.price_history or len(asset.price_history) < 20:
            return False

        current_price = asset.price_history[-1]
        ma_200 = np.mean(asset.price_history[-200:]) if len(asset.price_history) >= 200 else current_price

        # Simple breakout: price above 200-day MA
        return current_price > ma_200 * 1.02  # 2% above MA


# Global intelligence engine
intelligence_engine = IntelligenceEngine()
