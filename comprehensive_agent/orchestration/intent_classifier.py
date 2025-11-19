"""
Intent Classifier for OpenBB Comprehensive Agent

This module classifies user queries into intents to determine which agent(s)
should handle the request.
"""

from enum import Enum
from typing import List, Dict, Set
import re
import logging
from comprehensive_agent.config import INTENT_KEYWORDS


# Configure logging
logger = logging.getLogger(__name__)


class IntentType(Enum):
    """Classification of user query intents"""
    MARKET_DATA = "market_data"
    TECHNICAL_ANALYSIS = "technical_analysis"
    FUNDAMENTAL_ANALYSIS = "fundamental_analysis"
    RISK_ASSESSMENT = "risk_assessment"
    PORTFOLIO_ANALYSIS = "portfolio_analysis"
    ECONOMIC_FORECAST = "economic_forecast"
    NEWS_SENTIMENT = "news_sentiment"
    MULTI_AGENT = "multi_agent"
    GENERAL = "general"


class IntentClassifier:
    """
    Classify user queries into intents for agent routing

    Uses keyword matching and pattern recognition to determine:
    - Single intent queries → Route to one specialized agent
    - Multi-intent queries → Route to multiple agents
    - General queries → Route to general agent or orchestrator
    """

    def __init__(self):
        """Initialize the intent classifier"""
        self.intent_keywords = INTENT_KEYWORDS

        # Precompile regex patterns for common symbols
        self.ticker_pattern = re.compile(r'\b[A-Z]{1,5}\b')
        self.price_pattern = re.compile(r'\$\s*\d+|\d+\.\d+')
        self.percentage_pattern = re.compile(r'\d+\.?\d*\s*%')

        logger.info("Intent classifier initialized")

    def classify(self, query: str, context: Dict = None) -> List[IntentType]:
        """
        Classify query and return list of detected intents

        Args:
            query: User query string
            context: Additional context (widget data, uploaded files, etc.)

        Returns:
            List of IntentType enums representing detected intents
        """
        if not query or not query.strip():
            return [IntentType.GENERAL]

        query_lower = query.lower().strip()
        detected_intents = set()

        # Check each intent type
        for intent_str, keywords in self.intent_keywords.items():
            if self._matches_keywords(query_lower, keywords):
                # Map string to IntentType enum
                try:
                    intent_type = IntentType(intent_str)
                    detected_intents.add(intent_type)
                except ValueError:
                    logger.warning(f"Unknown intent type: {intent_str}")

        # Context-based detection
        if context:
            detected_intents.update(self._detect_from_context(context))

        # Pattern-based detection
        detected_intents.update(self._detect_from_patterns(query))

        # Handle multi-agent scenarios
        if len(detected_intents) > 1:
            logger.info(f"Multi-agent query detected: {[i.value for i in detected_intents]}")
            return [IntentType.MULTI_AGENT] + list(detected_intents)

        # Return detected intents or default to GENERAL
        if detected_intents:
            logger.debug(f"Classified query as: {[i.value for i in detected_intents]}")
            return list(detected_intents)

        logger.debug("No specific intent detected, defaulting to GENERAL")
        return [IntentType.GENERAL]

    def _matches_keywords(self, query: str, keywords: List[str]) -> bool:
        """
        Check if query contains any of the keywords

        Args:
            query: Lowercase query string
            keywords: List of keywords to match

        Returns:
            True if any keyword is found
        """
        return any(keyword in query for keyword in keywords)

    def _detect_from_context(self, context: Dict) -> Set[IntentType]:
        """
        Detect intents from request context

        Args:
            context: Request context dictionary

        Returns:
            Set of detected IntentType enums
        """
        intents = set()

        # Widget data indicates market data intent
        if context.get("widget_data"):
            intents.add(IntentType.MARKET_DATA)

            # Analyze widget type
            widget_type = context.get("widget_type", "").lower()
            if "chart" in widget_type or "technical" in widget_type:
                intents.add(IntentType.TECHNICAL_ANALYSIS)
            elif "financial" in widget_type or "fundamental" in widget_type:
                intents.add(IntentType.FUNDAMENTAL_ANALYSIS)
            elif "portfolio" in widget_type:
                intents.add(IntentType.PORTFOLIO_ANALYSIS)

        # File uploads can indicate different intents
        if context.get("uploaded_files"):
            for file_info in context["uploaded_files"]:
                filename = file_info.get("filename", "").lower()

                if filename.endswith((".pdf",)):
                    intents.add(IntentType.NEWS_SENTIMENT)
                elif filename.endswith((".xlsx", ".xls", ".csv")):
                    intents.add(IntentType.FUNDAMENTAL_ANALYSIS)

        # Web search request
        if context.get("web_search_requested"):
            intents.add(IntentType.NEWS_SENTIMENT)

        return intents

    def _detect_from_patterns(self, query: str) -> Set[IntentType]:
        """
        Detect intents from query patterns

        Args:
            query: User query string

        Returns:
            Set of detected IntentType enums
        """
        intents = set()

        # Ticker symbols suggest market data or technical analysis
        if self.ticker_pattern.search(query):
            # Check if it's more technical or fundamental
            if any(word in query.lower() for word in ["chart", "price", "trend", "technical"]):
                intents.add(IntentType.TECHNICAL_ANALYSIS)
            elif any(word in query.lower() for word in ["financial", "earnings", "revenue"]):
                intents.add(IntentType.FUNDAMENTAL_ANALYSIS)
            else:
                intents.add(IntentType.MARKET_DATA)

        # Price mentions
        if self.price_pattern.search(query):
            intents.add(IntentType.MARKET_DATA)

        # Percentage mentions with risk keywords
        if self.percentage_pattern.search(query):
            if any(word in query.lower() for word in ["risk", "volatility", "var"]):
                intents.add(IntentType.RISK_ASSESSMENT)

        # Question patterns
        if self._is_forecast_question(query):
            intents.add(IntentType.ECONOMIC_FORECAST)

        if self._is_news_question(query):
            intents.add(IntentType.NEWS_SENTIMENT)

        return intents

    def _is_forecast_question(self, query: str) -> bool:
        """Check if query is asking for forecasts or predictions"""
        forecast_patterns = [
            r'\bwill\b.*\b(go|increase|decrease|rise|fall)\b',
            r'\bwhat.*\b(happen|expect|predict)\b',
            r'\b(next|future|upcoming)\b.*\b(quarter|month|year)\b',
            r'\b(forecast|prediction|outlook|projection)\b'
        ]

        return any(re.search(pattern, query.lower()) for pattern in forecast_patterns)

    def _is_news_question(self, query: str) -> bool:
        """Check if query is asking for news or recent events"""
        news_patterns = [
            r'\b(latest|recent|new|current)\b.*\b(news|article|announcement)\b',
            r'\bwhat.*\b(happening|happened|going on)\b',
            r'\b(search|find|look up)\b'
        ]

        return any(re.search(pattern, query.lower()) for pattern in news_patterns)

    def get_primary_intent(self, intents: List[IntentType]) -> IntentType:
        """
        Get the primary intent from a list of intents

        Args:
            intents: List of detected intents

        Returns:
            Primary IntentType (first non-MULTI_AGENT intent)
        """
        if not intents:
            return IntentType.GENERAL

        # Skip MULTI_AGENT marker if present
        for intent in intents:
            if intent != IntentType.MULTI_AGENT:
                return intent

        return IntentType.GENERAL

    def is_multi_agent(self, intents: List[IntentType]) -> bool:
        """
        Check if the classified intents require multiple agents

        Args:
            intents: List of detected intents

        Returns:
            True if MULTI_AGENT is in the list
        """
        return IntentType.MULTI_AGENT in intents

    def get_agent_intents(self, intents: List[IntentType]) -> List[IntentType]:
        """
        Get the agent-specific intents (excluding MULTI_AGENT marker)

        Args:
            intents: List of detected intents

        Returns:
            List of IntentType without MULTI_AGENT or GENERAL
        """
        return [
            intent for intent in intents
            if intent not in (IntentType.MULTI_AGENT, IntentType.GENERAL)
        ]

    def classify_with_confidence(
        self,
        query: str,
        context: Dict = None
    ) -> List[tuple[IntentType, float]]:
        """
        Classify query and return intents with confidence scores

        Args:
            query: User query string
            context: Additional context

        Returns:
            List of (IntentType, confidence_score) tuples
        """
        if not query or not query.strip():
            return [(IntentType.GENERAL, 1.0)]

        query_lower = query.lower().strip()
        intent_scores = {}

        # Score each intent based on keyword matches
        for intent_str, keywords in self.intent_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in query_lower)

            if matches > 0:
                # Calculate confidence based on number of matches
                confidence = min(matches / 3.0, 1.0)  # Cap at 1.0

                try:
                    intent_type = IntentType(intent_str)
                    intent_scores[intent_type] = confidence
                except ValueError:
                    continue

        # Boost scores from context
        if context:
            context_intents = self._detect_from_context(context)
            for intent in context_intents:
                intent_scores[intent] = intent_scores.get(intent, 0.0) + 0.3

        # Boost scores from patterns
        pattern_intents = self._detect_from_patterns(query)
        for intent in pattern_intents:
            intent_scores[intent] = intent_scores.get(intent, 0.0) + 0.2

        # Cap all scores at 1.0
        intent_scores = {
            intent: min(score, 1.0)
            for intent, score in intent_scores.items()
        }

        # Sort by confidence descending
        sorted_intents = sorted(
            intent_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        if sorted_intents:
            # Add MULTI_AGENT if multiple high-confidence intents
            high_confidence = [i for i, s in sorted_intents if s >= 0.5]
            if len(high_confidence) > 1:
                sorted_intents.insert(0, (IntentType.MULTI_AGENT, 1.0))

            logger.debug(f"Intent confidence scores: {[(i.value, f'{s:.2f}') for i, s in sorted_intents]}")
            return sorted_intents

        logger.debug("No specific intent detected, defaulting to GENERAL")
        return [(IntentType.GENERAL, 1.0)]

    def __repr__(self) -> str:
        """String representation of the classifier"""
        return f"<IntentClassifier intents={len(self.intent_keywords)}>"
