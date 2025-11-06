"""
Financial Workspace - The Living Model

This is not just data. This is the user's financial world.

Every widget tells a story. Every asset has relationships.
The workspace understands the whole, not just the parts.
"""

from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import json

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class InvestmentStyle(Enum):
    """How the user approaches investing"""
    TRADER = "trader"              # Short-term, technical focus
    INVESTOR = "investor"          # Long-term, fundamental focus
    BALANCED = "balanced"          # Mix of both
    PASSIVE = "passive"            # Index/ETF focused
    UNKNOWN = "unknown"


class AssetType(Enum):
    """What kind of asset"""
    STOCK = "stock"
    ETF = "etf"
    CRYPTO = "crypto"
    OPTION = "option"
    BOND = "bond"
    COMMODITY = "commodity"
    CURRENCY = "currency"
    INDEX = "index"
    UNKNOWN = "unknown"


@dataclass
class Asset:
    """A single financial asset in the workspace"""
    symbol: str
    asset_type: AssetType = AssetType.UNKNOWN
    current_price: Optional[float] = None
    price_history: List[float] = field(default_factory=list)
    volume_history: List[float] = field(default_factory=list)

    # Tracking metadata
    widget_sources: Set[str] = field(default_factory=set)  # Which widgets track this
    last_updated: Optional[datetime] = None

    # Analytics
    volatility: Optional[float] = None
    beta: Optional[float] = None
    correlation_with_market: Optional[float] = None

    # Fundamentals (if available)
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    sector: Optional[str] = None

    # News sentiment (if available)
    sentiment_score: Optional[float] = None
    news_count: int = 0


@dataclass
class Portfolio:
    """User's portfolio holdings"""
    holdings: Dict[str, float] = field(default_factory=dict)  # symbol -> quantity/value
    total_value: float = 0.0

    # Composition
    asset_allocation: Dict[AssetType, float] = field(default_factory=dict)
    sector_allocation: Dict[str, float] = field(default_factory=dict)

    # Risk metrics
    portfolio_volatility: Optional[float] = None
    portfolio_beta: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None

    # Concentration
    largest_position_pct: float = 0.0
    top_5_concentration: float = 0.0

    # Correlation
    average_correlation: Optional[float] = None
    correlation_matrix: Optional[pd.DataFrame] = None


@dataclass
class WorkspaceContext:
    """Understanding of user's focus and behavior"""
    investment_style: InvestmentStyle = InvestmentStyle.UNKNOWN

    # Focus areas
    primary_symbols: List[str] = field(default_factory=list)  # Most tracked
    primary_sectors: List[str] = field(default_factory=list)

    # Preferences inferred from widgets
    uses_technical_analysis: bool = False
    uses_fundamental_analysis: bool = False
    tracks_news: bool = False
    tracks_economic_data: bool = False

    # Timeframe preference
    typical_timeframe: Optional[str] = None  # "intraday", "daily", "weekly", etc.

    # Risk profile (inferred)
    risk_tolerance: str = "medium"  # "low", "medium", "high"


class FinancialWorkspace:
    """
    The living model of a user's financial world.

    Not just widgets. Not just data.
    This understands the user's portfolio, their focus, their style.
    It sees the whole picture.
    """

    def __init__(self):
        # Core state
        self.assets: Dict[str, Asset] = {}
        self.portfolio: Optional[Portfolio] = None
        self.context: WorkspaceContext = WorkspaceContext()

        # History for change detection
        self.previous_state: Optional[Dict[str, Any]] = None
        self.state_history: List[Dict[str, Any]] = []

        # Metadata
        self.last_analysis: Optional[datetime] = None
        self.widget_count: int = 0

    async def build_from_widgets(self, widgets: List[Any], widget_data_map: Optional[Dict] = None) -> None:
        """
        Parse the dashboard into a coherent financial world.

        This is where we transform raw widgets into understanding.
        """
        logger.info(f"Building financial workspace from {len(widgets)} widgets")

        # Save previous state for change detection
        if self.assets:
            self.previous_state = self._capture_state()

        # Reset current state
        self.assets = {}
        self.widget_count = len(widgets)

        # Import widget intelligence for context
        from ..processors.widget_intelligence import widget_intelligence

        # Process each widget
        for widget in widgets:
            try:
                # Get widget context
                widget_ctx = await widget_intelligence.analyze_widget(widget, widget_data_map)

                # Extract assets from this widget
                await self._extract_assets_from_widget(widget, widget_ctx, widget_data_map)

                # Update user context based on widget type
                self._update_user_context(widget_ctx)

            except Exception as e:
                logger.warning(f"Error processing widget {widget.name}: {e}")
                continue

        # Build portfolio view
        await self._build_portfolio_view()

        # Infer investment style
        self._infer_investment_style()

        # Calculate correlations
        await self._calculate_correlations()

        # Save state
        self.last_analysis = datetime.now()
        self._save_to_history()

        logger.info(f"Workspace built: {len(self.assets)} assets, style={self.context.investment_style.value}")

    async def _extract_assets_from_widget(
        self,
        widget: Any,
        widget_ctx: Any,
        widget_data_map: Optional[Dict]
    ) -> None:
        """Extract asset information from a single widget"""

        # Get symbols from widget context
        symbols = widget_ctx.symbols

        for symbol in symbols:
            # Get or create asset
            if symbol not in self.assets:
                self.assets[symbol] = Asset(symbol=symbol)

            asset = self.assets[symbol]

            # Track which widget provides this data
            asset.widget_sources.add(widget.name)
            asset.last_updated = datetime.now()

            # Infer asset type from widget
            asset.asset_type = self._infer_asset_type(widget_ctx, symbol)

            # Extract data based on widget type
            if widget_ctx.detected_type == "price_chart":
                await self._extract_price_data(asset, widget, widget_data_map)

            elif widget_ctx.detected_type == "financial_statement":
                await self._extract_fundamental_data(asset, widget, widget_data_map)

            elif widget_ctx.detected_type == "news":
                await self._extract_news_data(asset, widget, widget_data_map)

            elif widget_ctx.detected_type == "technical_indicator":
                await self._extract_technical_data(asset, widget, widget_data_map)

    def _infer_asset_type(self, widget_ctx: Any, symbol: str) -> AssetType:
        """Infer what type of asset this is"""

        # Check category
        if widget_ctx.category:
            cat = widget_ctx.category.lower()
            if "crypto" in cat or "bitcoin" in cat:
                return AssetType.CRYPTO
            elif "etf" in cat or "fund" in cat:
                return AssetType.ETF
            elif "option" in cat:
                return AssetType.OPTION

        # Check symbol patterns
        if len(symbol) <= 5 and symbol.isupper():
            return AssetType.STOCK

        if symbol.upper() in ["BTC", "ETH", "DOGE"]:
            return AssetType.CRYPTO

        if symbol.startswith("^"):
            return AssetType.INDEX

        return AssetType.UNKNOWN

    async def _extract_price_data(self, asset: Asset, widget: Any, data_map: Optional[Dict]) -> None:
        """Extract price and volume data"""
        # This would parse actual widget data
        # For now, placeholder
        pass

    async def _extract_fundamental_data(self, asset: Asset, widget: Any, data_map: Optional[Dict]) -> None:
        """Extract fundamental metrics"""
        pass

    async def _extract_news_data(self, asset: Asset, widget: Any, data_map: Optional[Dict]) -> None:
        """Extract news sentiment"""
        pass

    async def _extract_technical_data(self, asset: Asset, widget: Any, data_map: Optional[Dict]) -> None:
        """Extract technical indicators"""
        pass

    def _update_user_context(self, widget_ctx: Any) -> None:
        """Learn about user's preferences from widgets"""

        widget_type = widget_ctx.detected_type

        if widget_type in ["technical_indicator", "price_chart"]:
            self.context.uses_technical_analysis = True

        if widget_type in ["financial_statement", "fundamental_ratio"]:
            self.context.uses_fundamental_analysis = True

        if widget_type == "news":
            self.context.tracks_news = True

        if widget_type == "economic_indicator":
            self.context.tracks_economic_data = True

    async def _build_portfolio_view(self) -> None:
        """Build coherent portfolio from tracked assets"""

        if not self.assets:
            return

        self.portfolio = Portfolio()

        # Simple heuristic: all tracked assets with prices are holdings
        # (In reality, would look for portfolio widget or holdings data)
        for symbol, asset in self.assets.items():
            if asset.current_price:
                # Default to equal weight if no actual holdings data
                self.portfolio.holdings[symbol] = 1.0

        # Calculate concentrations
        if self.portfolio.holdings:
            total = sum(self.portfolio.holdings.values())
            sorted_holdings = sorted(
                self.portfolio.holdings.items(),
                key=lambda x: x[1],
                reverse=True
            )

            if sorted_holdings:
                self.portfolio.largest_position_pct = sorted_holdings[0][1] / total if total > 0 else 0
                top_5_value = sum(h[1] for h in sorted_holdings[:5])
                self.portfolio.top_5_concentration = top_5_value / total if total > 0 else 0

    def _infer_investment_style(self) -> None:
        """Infer how the user invests"""

        tech_score = 1 if self.context.uses_technical_analysis else 0
        fund_score = 1 if self.context.uses_fundamental_analysis else 0

        # Count ETFs vs stocks
        etf_count = sum(1 for a in self.assets.values() if a.asset_type == AssetType.ETF)
        stock_count = sum(1 for a in self.assets.values() if a.asset_type == AssetType.STOCK)

        if etf_count > stock_count * 2:
            self.context.investment_style = InvestmentStyle.PASSIVE
        elif tech_score > 0 and fund_score == 0:
            self.context.investment_style = InvestmentStyle.TRADER
        elif fund_score > 0 and tech_score == 0:
            self.context.investment_style = InvestmentStyle.INVESTOR
        elif tech_score > 0 and fund_score > 0:
            self.context.investment_style = InvestmentStyle.BALANCED
        else:
            self.context.investment_style = InvestmentStyle.UNKNOWN

        logger.info(f"Inferred investment style: {self.context.investment_style.value}")

    async def _calculate_correlations(self) -> None:
        """Calculate asset correlations"""

        if len(self.assets) < 2:
            return

        # Would use actual price history here
        # Placeholder for now
        correlations = {}

        symbols = list(self.assets.keys())
        for i, sym1 in enumerate(symbols):
            for sym2 in symbols[i+1:]:
                # Placeholder correlation
                correlations[f"{sym1}_{sym2}"] = 0.0

        if self.portfolio:
            self.portfolio.average_correlation = np.mean(list(correlations.values())) if correlations else None

    def _capture_state(self) -> Dict[str, Any]:
        """Capture current state for change detection"""
        return {
            "timestamp": datetime.now().isoformat(),
            "asset_count": len(self.assets),
            "assets": {
                symbol: {
                    "price": asset.current_price,
                    "volatility": asset.volatility,
                    "sentiment": asset.sentiment_score
                }
                for symbol, asset in self.assets.items()
            },
            "portfolio": {
                "total_value": self.portfolio.total_value if self.portfolio else 0,
                "concentration": self.portfolio.largest_position_pct if self.portfolio else 0
            } if self.portfolio else None
        }

    def _save_to_history(self) -> None:
        """Save state to history"""
        current = self._capture_state()
        self.state_history.append(current)

        # Keep last 100 states
        if len(self.state_history) > 100:
            self.state_history = self.state_history[-100:]

    def get_changes_since_last(self) -> Dict[str, Any]:
        """Detect what changed since last analysis"""

        if not self.previous_state:
            return {"first_analysis": True}

        changes = {
            "new_assets": [],
            "removed_assets": [],
            "price_changes": {},
            "sentiment_changes": {}
        }

        current_symbols = set(self.assets.keys())
        previous_symbols = set(self.previous_state.get("assets", {}).keys())

        changes["new_assets"] = list(current_symbols - previous_symbols)
        changes["removed_assets"] = list(previous_symbols - current_symbols)

        # Detect price changes
        for symbol in current_symbols & previous_symbols:
            prev = self.previous_state["assets"].get(symbol, {})
            curr_asset = self.assets[symbol]

            if prev.get("price") and curr_asset.current_price:
                change_pct = (curr_asset.current_price - prev["price"]) / prev["price"]
                if abs(change_pct) > 0.01:  # >1% change
                    changes["price_changes"][symbol] = change_pct

        return changes

    def get_summary(self) -> str:
        """Human-readable workspace summary"""

        lines = [
            f"📊 Financial Workspace Summary",
            f"",
            f"Assets Tracked: {len(self.assets)}",
        ]

        if self.portfolio:
            lines.extend([
                f"Portfolio Holdings: {len(self.portfolio.holdings)}",
                f"Largest Position: {self.portfolio.largest_position_pct*100:.1f}%",
                f"Top 5 Concentration: {self.portfolio.top_5_concentration*100:.1f}%"
            ])

        lines.extend([
            f"",
            f"Investment Style: {self.context.investment_style.value.title()}",
            f"Technical Analysis: {'Yes' if self.context.uses_technical_analysis else 'No'}",
            f"Fundamental Analysis: {'Yes' if self.context.uses_fundamental_analysis else 'No'}",
            f"News Tracking: {'Yes' if self.context.tracks_news else 'No'}"
        ])

        if self.context.primary_symbols:
            lines.append(f"Primary Focus: {', '.join(self.context.primary_symbols[:5])}")

        return "\n".join(lines)


# Global workspace instance
workspace = FinancialWorkspace()
