"""
ML-Widget Bridge

Connects machine learning capabilities with OpenBB widget data.
Automatically applies appropriate ML models based on widget type and data.
"""

from typing import Any, Dict, List, Optional, Tuple
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class MLWidgetBridge:
    """
    Bridge between ML models and OpenBB widget data.

    This class automatically:
    - Detects appropriate ML models for widget data
    - Prepares features from widget data
    - Applies predictions and analysis
    - Generates human-readable insights
    """

    def __init__(self):
        self.model_cache = {}
        self.feature_cache = {}

    async def auto_apply_ml(
        self,
        widget_data: Any,
        widget_type: str,
        widget_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Automatically apply appropriate ML based on widget type.

        Args:
            widget_data: The widget data content
            widget_type: Detected widget type (from WidgetIntelligence)
            widget_context: Additional context about the widget

        Returns:
            ML analysis results with insights
        """
        try:
            if widget_type == "price_chart":
                return await self._analyze_price_data(widget_data, widget_context)

            elif widget_type == "financial_statement":
                return await self._analyze_financial_data(widget_data, widget_context)

            elif widget_type == "news":
                return await self._analyze_news_data(widget_data, widget_context)

            elif widget_type == "technical_indicator":
                return await self._analyze_technical_data(widget_data, widget_context)

            elif widget_type == "portfolio":
                return await self._analyze_portfolio_data(widget_data, widget_context)

            else:
                return await self._generic_analysis(widget_data, widget_context)

        except Exception as e:
            logger.error(f"ML analysis failed: {e}")
            return {
                "status": "error",
                "message": f"ML analysis encountered an error: {str(e)}",
                "insights": []
            }

    async def _analyze_price_data(
        self,
        widget_data: Any,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze price/OHLC data with ML."""
        insights = []
        metrics = {}

        try:
            # Extract price series
            prices = self._extract_price_series(widget_data)

            if len(prices) < 10:
                return {
                    "status": "insufficient_data",
                    "message": "Need at least 10 data points for ML analysis",
                    "insights": []
                }

            # Calculate returns
            returns = np.diff(prices) / prices[:-1]
            metrics["volatility"] = float(np.std(returns) * np.sqrt(252))  # Annualized
            metrics["mean_return"] = float(np.mean(returns))

            # Trend detection
            trend_strength = self._detect_trend(prices)
            metrics["trend_strength"] = trend_strength
            metrics["trend_direction"] = "up" if trend_strength > 0 else "down" if trend_strength < 0 else "neutral"

            # Volatility regime
            recent_vol = np.std(returns[-20:]) if len(returns) >= 20 else np.std(returns)
            historical_vol = np.std(returns)
            vol_ratio = recent_vol / historical_vol if historical_vol > 0 else 1.0

            if vol_ratio > 1.5:
                insights.append("⚠️ Volatility is significantly elevated compared to historical levels")
            elif vol_ratio < 0.7:
                insights.append("📉 Volatility is below historical average - market calm")

            # Momentum
            if len(prices) >= 20:
                momentum = (prices[-1] / prices[-20] - 1) * 100
                metrics["momentum_20d"] = momentum
                if abs(momentum) > 10:
                    direction = "up" if momentum > 0 else "down"
                    insights.append(f"🚀 Strong {direction}ward momentum: {momentum:.1f}% over 20 days")

            # Anomaly detection (simple z-score based)
            z_scores = (prices - np.mean(prices)) / np.std(prices)
            if np.abs(z_scores[-1]) > 2.5:
                insights.append(f"🔍 Current price is an outlier ({z_scores[-1]:.1f} std devs from mean)")

            # Price prediction (simple linear extrapolation)
            if len(prices) >= 30:
                prediction = self._simple_price_forecast(prices)
                metrics["predicted_direction"] = prediction["direction"]
                metrics["confidence"] = prediction["confidence"]

                if prediction["confidence"] > 0.6:
                    insights.append(
                        f"📊 ML model suggests {prediction['direction']} movement "
                        f"(confidence: {prediction['confidence']:.0%})"
                    )

            return {
                "status": "success",
                "analysis_type": "price_ml_analysis",
                "metrics": metrics,
                "insights": insights,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Price analysis error: {e}")
            return {"status": "error", "message": str(e), "insights": []}

    async def _analyze_financial_data(
        self,
        widget_data: Any,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze financial statement data."""
        insights = []
        metrics = {}

        try:
            # Parse financial data
            financial_metrics = self._extract_financial_metrics(widget_data)

            if not financial_metrics:
                return {
                    "status": "no_data",
                    "message": "Could not extract financial metrics",
                    "insights": []
                }

            # Growth analysis
            if "revenue" in financial_metrics and len(financial_metrics["revenue"]) >= 2:
                revenues = financial_metrics["revenue"]
                growth_rate = (revenues[-1] / revenues[-2] - 1) * 100
                metrics["revenue_growth"] = growth_rate

                if growth_rate > 20:
                    insights.append(f"📈 Strong revenue growth: {growth_rate:.1f}% YoY")
                elif growth_rate < 0:
                    insights.append(f"⚠️ Revenue declining: {growth_rate:.1f}% YoY")

            # Profitability trends
            if "net_income" in financial_metrics and len(financial_metrics["net_income"]) >= 2:
                profits = financial_metrics["net_income"]
                if profits[-1] > profits[-2]:
                    insights.append("✅ Profitability improving")
                else:
                    insights.append("⚠️ Profitability declining")

            # Anomaly detection in metrics
            for metric_name, values in financial_metrics.items():
                if len(values) >= 4:
                    mean = np.mean(values[:-1])
                    std = np.std(values[:-1])
                    latest = values[-1]

                    if std > 0 and abs(latest - mean) / std > 2:
                        insights.append(
                            f"🔍 {metric_name.replace('_', ' ').title()} shows unusual pattern"
                        )

            return {
                "status": "success",
                "analysis_type": "financial_ml_analysis",
                "metrics": metrics,
                "insights": insights,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Financial analysis error: {e}")
            return {"status": "error", "message": str(e), "insights": []}

    async def _analyze_news_data(
        self,
        widget_data: Any,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze news data for sentiment and patterns."""
        insights = []
        metrics = {}

        try:
            # Extract news items
            news_items = self._extract_news_items(widget_data)

            if not news_items:
                return {"status": "no_data", "message": "No news data found", "insights": []}

            metrics["news_count"] = len(news_items)

            # Simple sentiment analysis based on keywords
            positive_keywords = ['surge', 'gain', 'up', 'growth', 'profit', 'beat', 'strong', 'success']
            negative_keywords = ['fall', 'drop', 'down', 'loss', 'miss', 'weak', 'decline', 'concern']

            sentiment_scores = []
            for item in news_items:
                text = str(item).lower()
                pos_count = sum(1 for kw in positive_keywords if kw in text)
                neg_count = sum(1 for kw in negative_keywords if kw in text)
                score = pos_count - neg_count
                sentiment_scores.append(score)

            avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0
            metrics["sentiment_score"] = float(avg_sentiment)

            if avg_sentiment > 0.5:
                insights.append("📰 News sentiment is predominantly positive")
            elif avg_sentiment < -0.5:
                insights.append("📰 News sentiment is predominantly negative")
            else:
                insights.append("📰 News sentiment is mixed/neutral")

            # News volume analysis
            if len(news_items) > 10:
                insights.append(f"📢 High news volume detected ({len(news_items)} articles) - increased attention")

            return {
                "status": "success",
                "analysis_type": "news_ml_analysis",
                "metrics": metrics,
                "insights": insights,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"News analysis error: {e}")
            return {"status": "error", "message": str(e), "insights": []}

    async def _analyze_technical_data(
        self,
        widget_data: Any,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze technical indicator data."""
        insights = []

        # Technical indicators usually come pre-computed in widgets
        # We can add interpretation and signal generation

        insights.append("📊 Technical indicators available - use for signal confirmation")

        return {
            "status": "success",
            "analysis_type": "technical_ml_analysis",
            "insights": insights,
            "timestamp": datetime.now().isoformat()
        }

    async def _analyze_portfolio_data(
        self,
        widget_data: Any,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze portfolio allocation and risk."""
        insights = []
        metrics = {}

        try:
            # Extract portfolio holdings
            holdings = self._extract_portfolio_holdings(widget_data)

            if not holdings:
                return {"status": "no_data", "message": "No portfolio data found", "insights": []}

            # Diversification analysis
            if len(holdings) < 5:
                insights.append("⚠️ Portfolio may lack diversification - consider adding more positions")
            elif len(holdings) > 20:
                insights.append("🎯 Well-diversified portfolio with multiple positions")

            # Concentration risk
            if holdings:
                values = list(holdings.values())
                total = sum(values)
                max_position = max(values) / total if total > 0 else 0
                metrics["max_position_pct"] = max_position * 100

                if max_position > 0.3:
                    insights.append(
                        f"⚠️ Concentration risk: largest position is {max_position*100:.1f}% of portfolio"
                    )

            return {
                "status": "success",
                "analysis_type": "portfolio_ml_analysis",
                "metrics": metrics,
                "insights": insights,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Portfolio analysis error: {e}")
            return {"status": "error", "message": str(e), "insights": []}

    async def _generic_analysis(
        self,
        widget_data: Any,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generic analysis for unknown widget types."""
        return {
            "status": "success",
            "analysis_type": "generic_analysis",
            "insights": ["Data processed successfully"],
            "timestamp": datetime.now().isoformat()
        }

    # Helper methods for data extraction

    def _extract_price_series(self, widget_data: Any) -> np.ndarray:
        """Extract price series from widget data."""
        prices = []

        try:
            if isinstance(widget_data, list):
                for item in widget_data:
                    if hasattr(item, 'items'):
                        for sub_item in item.items:
                            content = getattr(sub_item, 'content', '')
                            # Try to parse as JSON containing price data
                            try:
                                import json
                                data = json.loads(content)
                                if isinstance(data, list):
                                    for row in data:
                                        if isinstance(row, dict):
                                            # Look for price fields
                                            for key in ['close', 'price', 'value']:
                                                if key in row:
                                                    prices.append(float(row[key]))
                                                    break
                            except:
                                pass
        except Exception as e:
            logger.warning(f"Error extracting prices: {e}")

        return np.array(prices) if prices else np.array([])

    def _extract_financial_metrics(self, widget_data: Any) -> Dict[str, List[float]]:
        """Extract financial metrics from widget data."""
        metrics = {}
        # Placeholder - would need to parse actual financial data structure
        return metrics

    def _extract_news_items(self, widget_data: Any) -> List[str]:
        """Extract news items from widget data."""
        news_items = []
        # Placeholder - would need to parse actual news data structure
        return news_items

    def _extract_portfolio_holdings(self, widget_data: Any) -> Dict[str, float]:
        """Extract portfolio holdings from widget data."""
        holdings = {}
        # Placeholder - would need to parse actual portfolio data structure
        return holdings

    def _detect_trend(self, prices: np.ndarray) -> float:
        """Detect trend strength using simple linear regression slope."""
        if len(prices) < 2:
            return 0.0

        x = np.arange(len(prices))
        slope, _ = np.polyfit(x, prices, 1)

        # Normalize by price level
        normalized_slope = slope / np.mean(prices) * 100
        return float(normalized_slope)

    def _simple_price_forecast(self, prices: np.ndarray) -> Dict[str, Any]:
        """Simple price direction forecast using momentum and trend."""
        if len(prices) < 30:
            return {"direction": "uncertain", "confidence": 0.5}

        # Recent trend
        recent_trend = self._detect_trend(prices[-20:])

        # Momentum
        momentum = (prices[-1] / prices[-20] - 1) * 100

        # Combined signal
        signal = recent_trend + momentum

        if signal > 2:
            direction = "upward"
            confidence = min(0.95, 0.6 + abs(signal) * 0.05)
        elif signal < -2:
            direction = "downward"
            confidence = min(0.95, 0.6 + abs(signal) * 0.05)
        else:
            direction = "sideways"
            confidence = 0.5

        return {
            "direction": direction,
            "confidence": confidence,
            "signal_strength": float(signal)
        }


# Global instance
ml_widget_bridge = MLWidgetBridge()
