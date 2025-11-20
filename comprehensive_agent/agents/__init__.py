"""
Agents module for OpenBB Comprehensive Agent

This module contains all specialized financial analysis agents:
- BaseAgent: Abstract base class for all agents

Specialized agents (to be implemented in Phase 2):
- MarketDataAgent: Market data retrieval and processing
- TechnicalAnalysisAgent: Technical analysis and charting
- FundamentalAnalysisAgent: Fundamental analysis and valuations
- RiskAnalyticsAgent: Risk assessment and analytics
- PortfolioManagementAgent: Portfolio optimization and management
- EconomicAnalysisAgent: Economic forecasting and analysis
- NewsSentimentAgent: News aggregation and sentiment analysis
"""

from .base_agent import BaseAgent, AgentState

# Import specialized agents when they are implemented
from .market_data_agent import MarketDataAgent
# from .technical_analysis_agent import TechnicalAnalysisAgent
# from .fundamental_analysis_agent import FundamentalAnalysisAgent
# from .risk_analytics_agent import RiskAnalyticsAgent
# from .portfolio_management_agent import PortfolioManagementAgent
# from .economic_analysis_agent import EconomicAnalysisAgent
# from .news_sentiment_agent import NewsSentimentAgent

__all__ = [
    "BaseAgent",
    "AgentState",
    # Specialized agents (Phase 2a)
    "MarketDataAgent",
    # Remaining agents will be added in Phase 2b/2c
    # "TechnicalAnalysisAgent",
    # "FundamentalAnalysisAgent",
    # "RiskAnalyticsAgent",
    # "PortfolioManagementAgent",
    # "EconomicAnalysisAgent",
    # "NewsSentimentAgent"
]
