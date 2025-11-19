# Multi-Agent Orchestration Architecture Plan for OpenBB Comprehensive Agent

## Overview
This plan transforms the current single-agent OpenBB system into a **multi-agent orchestration architecture** modeled after the Financial Assistant pattern, where an orchestrator routes user queries to specialized sub-agents, each with 6-7 focused tools.

---

## Architecture Transformation

### Current State (Single Agent)
```
USER QUERY
    ↓
FastAPI Gateway (main.py)
    ↓
Sequential Processor Pipeline
    ├─ Widget Intelligence
    ├─ ML Widget Bridge
    ├─ Web Search
    ├─ PDF/Spreadsheet Processing
    └─ Visualization Generation
    ↓
Ollama LLM → Response
```

### Target State (Multi-Agent Orchestration)
```
USER QUERY
    ↓
ORCHESTRATOR AGENT
"OpenBB Financial Intelligence Orchestrator"
- Intent analysis
- Agent routing
- Result aggregation
    ↓
┌─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┐
│ Market  │Technical│ Fundamental│ Risk  │Portfolio│Economic │ News &  │
│ Data    │Analysis │  Analysis  │Analytics│Management│Analysis│Sentiment│
│ Agent   │ Agent   │   Agent    │ Agent  │  Agent   │ Agent  │ Agent   │
└─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┘
                              ↓
                        TOOLS LAYER
                    (6-7 tools per agent)
```

---

## Specialized Agent Definitions

### 1. **Market Data Agent**
**Purpose:** Retrieve, process, and analyze market data from OpenBB and widgets

**Tools (7):**
```
├─ Get Widget Data
├─ Process Price Data
├─ Extract OHLCV Data
├─ Analyze Widget Context
├─ Fetch Real-time Data
├─ Historical Data Retrieval
└─ Data Validation & Quality Check
```

**Mapped from existing:**
- `processors/widgets.py` → `process_widget_data()`, `extract_data_for_visualization()`
- `processors/widget_intelligence.py` → `WidgetIntelligence.analyze_widget()`
- `processors/data_validator.py` → `DataValidator`
- `processors/api_data.py` → `process_api_data()`
- `processors/api_data_processor.py` → `APIDataProcessor`

---

### 2. **Technical Analysis Agent**
**Purpose:** Perform technical analysis, chart patterns, indicators, and price action analysis

**Tools (7):**
```
├─ Generate Charts (Line, Candlestick, OHLC)
├─ Technical Indicators (RSI, MACD, BB, etc.)
├─ Trend Detection
├─ Support/Resistance Levels
├─ Volume Analysis
├─ Pattern Recognition
└─ Interactive Chart Features
```

**Mapped from existing:**
- `visualizations/charts.py` → `generate_charts()`, `detect_chart_types()`
- `visualizations/financial_charts.py` → Candlestick, OHLC charts
- `visualizations/interactive_charts.py` → Annotations, indicators
- `core/ml_widget_bridge.py` → Technical analysis methods
- NEW: Pattern recognition tools (to be implemented)
- NEW: Support/resistance detection (to be implemented)

---

### 3. **Fundamental Analysis Agent**
**Purpose:** Analyze financial statements, metrics, valuations, and company fundamentals

**Tools (7):**
```
├─ Process Spreadsheet Data (Financials)
├─ Financial Metrics Calculator
├─ Valuation Models (DCF, P/E, etc.)
├─ Growth Analysis
├─ Profitability Analysis
├─ Generate Financial Tables
└─ Balance Sheet/P&L Analysis
```

**Mapped from existing:**
- `processors/spreadsheet.py` → `process_spreadsheet_data()`
- `processors/spreadsheet_processor.py` → `SpreadsheetProcessor`
- `visualizations/tables.py` → `generate_tables()`
- `core/ml_widget_bridge.py` → `_analyze_financial_data()`
- NEW: Valuation models (to be implemented)
- NEW: Financial ratios calculator (to be implemented)
- NEW: Growth rate analysis (to be implemented)

---

### 4. **Risk Analytics Agent**
**Purpose:** Risk assessment, volatility analysis, stress testing, and correlation analysis

**Tools (6):**
```
├─ Volatility Analysis
├─ VaR (Value at Risk) Calculation
├─ Correlation Analysis
├─ Stress Testing
├─ Risk Metrics Dashboard
└─ Sentiment-Price Correlation
```

**Mapped from existing:**
- `core/ml_widget_bridge.py` → `_analyze_price_data()` (volatility)
- `utils/data_correlator.py` → `correlate_sentiment_with_prices()`
- `core/model_engine.py` → `ModelEngine` (risk models)
- NEW: VaR calculator (to be implemented)
- NEW: Stress testing framework (to be implemented)
- NEW: Risk dashboard generator (to be implemented)

---

### 5. **Portfolio Management Agent**
**Purpose:** Portfolio analysis, optimization, diversification, and performance tracking

**Tools (7):**
```
├─ Portfolio Analysis
├─ Diversification Metrics
├─ Asset Allocation
├─ Performance Attribution
├─ Rebalancing Recommendations
├─ Portfolio Optimization
└─ Portfolio Visualization
```

**Mapped from existing:**
- `core/ml_widget_bridge.py` → `_analyze_portfolio_data()`
- `core/model_engine.py` → Clustering for diversification
- `visualizations/financial_charts.py` → Heatmaps
- NEW: Sharpe ratio calculator (to be implemented)
- NEW: Asset allocation optimizer (to be implemented)
- NEW: Performance attribution (to be implemented)
- NEW: Rebalancing engine (to be implemented)

---

### 6. **Economic Analysis Agent**
**Purpose:** Macroeconomic analysis, economic indicators, forecasting, and market intelligence

**Tools (7):**
```
├─ Time Series Forecasting
├─ Economic Indicator Analysis
├─ Trend Prediction
├─ Regression Analysis
├─ Feature Engineering
├─ Market Intelligence Reports
└─ Scenario Analysis
```

**Mapped from existing:**
- `core/model_engine.py` → `ModelEngine.train_time_series()`, `predict_time_series()`
- `processors/feature_engineering.py` → `FeatureEngineer`
- `core/ml_widget_bridge.py` → Momentum, trend analysis
- NEW: ARIMA/LSTM forecasting (to be implemented)
- NEW: Economic indicator tracker (to be implemented)
- NEW: Scenario modeling (to be implemented)

---

### 7. **News & Sentiment Agent**
**Purpose:** News aggregation, sentiment analysis, web search, and market intelligence

**Tools (7):**
```
├─ Web Search (General)
├─ Financial News Search
├─ Sentiment Analysis
├─ News Aggregation
├─ Citation Generation
├─ PDF Document Processing
└─ Market Alert Generation
```

**Mapped from existing:**
- `processors/web_search.py` → `process_web_search()`, `detect_web_search_request()`
- `processors/financial_web_search.py` → `FinancialWebSearcher`
- `processors/pdf.py` → `process_pdf_data()`
- `processors/citations.py` → `generate_citations()`
- `utils/alerting.py` → `send_alert()`
- `core/ml_widget_bridge.py` → `_analyze_news_data()` (sentiment)
- NEW: News aggregation API (to be implemented)

---

## Orchestrator Design

### Orchestrator Agent: "OpenBB Intelligence Orchestrator"

**Core Responsibilities:**
1. **Query Intent Analysis** - Understand what the user wants
2. **Agent Selection** - Choose appropriate specialist agent(s)
3. **Request Routing** - Dispatch queries to selected agents
4. **Result Aggregation** - Combine responses from multiple agents
5. **Context Management** - Maintain conversation state
6. **Error Handling** - Graceful fallbacks and retries

**Intent Classification Categories:**
```python
IntentType = Enum('IntentType', [
    'MARKET_DATA',          # "Get Apple stock price"
    'TECHNICAL_ANALYSIS',   # "Show me RSI for TSLA"
    'FUNDAMENTAL_ANALYSIS', # "Analyze MSFT financials"
    'RISK_ASSESSMENT',      # "What's the volatility of BTC?"
    'PORTFOLIO_ANALYSIS',   # "Optimize my portfolio"
    'ECONOMIC_FORECAST',    # "Predict next quarter GDP"
    'NEWS_SENTIMENT',       # "Latest news on Tesla"
    'MULTI_AGENT'           # Requires multiple agents
])
```

**Routing Logic:**
```python
Query: "Analyze Tesla's stock with technical indicators and recent news"
    ↓
Intent: MULTI_AGENT (TECHNICAL_ANALYSIS + NEWS_SENTIMENT)
    ↓
Route to:
    ├─ Technical Analysis Agent → RSI, MACD, trends
    └─ News & Sentiment Agent → Recent news, sentiment score
    ↓
Aggregate Results → Combined response with charts + news
```

---

## Implementation Architecture

### Directory Structure
```
comprehensive_agent/
├── orchestration/                 # NEW - Orchestration layer
│   ├── __init__.py
│   ├── orchestrator.py           # Main orchestrator logic
│   ├── agent_registry.py         # Agent discovery & management
│   ├── intent_classifier.py      # Query intent analysis
│   ├── routing_engine.py         # Agent selection & dispatch
│   ├── result_aggregator.py      # Combine multi-agent results
│   └── context_manager.py        # Conversation state management
│
├── agents/                        # NEW - Specialized agents
│   ├── __init__.py
│   ├── base_agent.py             # Abstract base for all agents
│   ├── market_data_agent.py      # Market Data Agent
│   ├── technical_analysis_agent.py
│   ├── fundamental_analysis_agent.py
│   ├── risk_analytics_agent.py
│   ├── portfolio_management_agent.py
│   ├── economic_analysis_agent.py
│   └── news_sentiment_agent.py
│
├── tools/                         # NEW - Refactored tool layer
│   ├── __init__.py
│   ├── market_data/              # Tools for Market Data Agent
│   │   ├── widget_processor.py
│   │   ├── api_data_fetcher.py
│   │   ├── data_validator.py
│   │   └── __init__.py
│   ├── technical/                # Tools for Technical Analysis Agent
│   │   ├── chart_generator.py
│   │   ├── indicator_calculator.py
│   │   ├── pattern_detector.py
│   │   └── __init__.py
│   ├── fundamental/              # Tools for Fundamental Analysis Agent
│   │   ├── spreadsheet_processor.py
│   │   ├── financial_metrics.py
│   │   ├── valuation_models.py
│   │   └── __init__.py
│   ├── risk/                     # Tools for Risk Analytics Agent
│   │   ├── volatility_analyzer.py
│   │   ├── var_calculator.py
│   │   ├── correlation_engine.py
│   │   └── __init__.py
│   ├── portfolio/                # Tools for Portfolio Management Agent
│   │   ├── portfolio_analyzer.py
│   │   ├── optimizer.py
│   │   ├── allocation_engine.py
│   │   └── __init__.py
│   ├── economic/                 # Tools for Economic Analysis Agent
│   │   ├── forecasting_engine.py
│   │   ├── feature_engineer.py
│   │   ├── regression_models.py
│   │   └── __init__.py
│   └── news/                     # Tools for News & Sentiment Agent
│       ├── web_searcher.py
│       ├── sentiment_analyzer.py
│       ├── pdf_processor.py
│       ├── citation_generator.py
│       └── __init__.py
│
├── core/                          # EXISTING - Core engines
│   ├── data_science_agent.py     # Keep as base class
│   ├── model_engine.py           # ML model management
│   └── ml_widget_bridge.py       # Auto ML (refactor into tools)
│
├── processors/                    # DEPRECATED - Will migrate to tools/
│   └── (existing 14 modules - to be refactored)
│
├── visualizations/                # Keep as shared utilities
│   ├── charts.py
│   ├── tables.py
│   ├── financial_charts.py
│   └── interactive_charts.py
│
├── utils/                         # Shared utilities
│   ├── alerting.py
│   ├── data_correlator.py
│   └── error_handler.py
│
├── config.py                      # NEW - Configuration management
├── main.py                        # MODIFIED - Use orchestrator
├── prompts.py                     # MODIFIED - Agent-specific prompts
└── __init__.py
```

---

## Tool Distribution Matrix

| Agent | Tool Count | Source Modules (Existing → New Location) |
|-------|------------|------------------------------------------|
| **Market Data** | 7 | `processors/widgets.py` → `tools/market_data/widget_processor.py`<br>`processors/api_data*.py` → `tools/market_data/api_data_fetcher.py`<br>`processors/data_validator.py` → `tools/market_data/data_validator.py` |
| **Technical Analysis** | 7 | `visualizations/charts.py` → `tools/technical/chart_generator.py`<br>`visualizations/financial_charts.py` → `tools/technical/chart_generator.py`<br>NEW pattern detection, indicators |
| **Fundamental** | 7 | `processors/spreadsheet*.py` → `tools/fundamental/spreadsheet_processor.py`<br>`visualizations/tables.py` → `tools/fundamental/table_generator.py`<br>NEW valuation, metrics |
| **Risk Analytics** | 6 | `core/ml_widget_bridge.py` (volatility) → `tools/risk/volatility_analyzer.py`<br>`utils/data_correlator.py` → `tools/risk/correlation_engine.py`<br>NEW VaR, stress testing |
| **Portfolio** | 7 | `core/ml_widget_bridge.py` (portfolio) → `tools/portfolio/portfolio_analyzer.py`<br>NEW optimization, allocation, rebalancing |
| **Economic** | 7 | `core/model_engine.py` (time series) → `tools/economic/forecasting_engine.py`<br>`processors/feature_engineering.py` → `tools/economic/feature_engineer.py`<br>NEW ARIMA/LSTM |
| **News & Sentiment** | 7 | `processors/web_search.py` → `tools/news/web_searcher.py`<br>`processors/financial_web_search.py` → `tools/news/web_searcher.py`<br>`processors/pdf.py` → `tools/news/pdf_processor.py`<br>`processors/citations.py` → `tools/news/citation_generator.py` |

**Total:** 48 tools across 7 agents (average 6.9 tools/agent)

---

## Orchestration Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER QUERY                              │
│              "Analyze AAPL with technicals and news"            │
└───────────────────────────┬─────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                   ORCHESTRATOR AGENT                            │
│  1. Intent Classification: MULTI_AGENT                          │
│     - Technical Analysis (charts, RSI, MACD)                    │
│     - News & Sentiment (recent news, sentiment score)           │
│  2. Agent Selection: [TechnicalAnalysisAgent, NewsSentimentAgent]│
│  3. Context Preparation: {symbol: "AAPL", timeframe: "1d"}      │
└──────────┬─────────────────────────────────┬────────────────────┘
           ↓                                 ↓
┌──────────────────────────┐    ┌──────────────────────────────┐
│ Technical Analysis Agent │    │  News & Sentiment Agent      │
│                          │    │                              │
│ Tools:                   │    │ Tools:                       │
│ ✓ Generate Charts        │    │ ✓ Financial News Search      │
│ ✓ Technical Indicators   │    │ ✓ Sentiment Analysis         │
│ ✓ Trend Detection        │    │ ✓ Citation Generation        │
│                          │    │                              │
│ Returns:                 │    │ Returns:                     │
│ - Candlestick chart      │    │ - 5 recent news articles     │
│ - RSI: 68 (overbought)   │    │ - Sentiment: 0.72 (positive) │
│ - MACD: bullish cross    │    │ - Citations list             │
│ - Uptrend confirmed      │    │                              │
└────────────┬─────────────┘    └──────────────┬───────────────┘
             ↓                                 ↓
┌─────────────────────────────────────────────────────────────────┐
│                    RESULT AGGREGATOR                            │
│  1. Combine outputs from both agents                            │
│  2. Resolve conflicts (none in this case)                       │
│  3. Format unified response                                     │
│  4. Generate visualizations                                     │
└───────────────────────────┬─────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                   UNIFIED RESPONSE (SSE)                        │
│                                                                 │
│  Technical Analysis:                                            │
│  📊 [Candlestick Chart]                                         │
│  • Uptrend confirmed with higher highs                          │
│  • RSI at 68 (approaching overbought)                           │
│  • MACD showing bullish crossover                               │
│                                                                 │
│  Market Sentiment:                                              │
│  • Overall sentiment: Positive (0.72/1.0)                       │
│  • 5 recent articles analyzed                                   │
│  • Key themes: Product launches, earnings beat                  │
│                                                                 │
│  📚 Sources: [citations from news search]                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Components to Implement

### 1. Base Agent Class
```python
# agents/base_agent.py
from abc import ABC, abstractmethod
from typing import Dict, List, Any

class BaseAgent(ABC):
    """Base class for all specialized agents"""

    def __init__(self, name: str, description: str, capabilities: List[str]):
        self.name = name
        self.description = description
        self.capabilities = capabilities
        self.tools = {}
        self.state = "idle"

    @abstractmethod
    def can_handle(self, intent: str, context: Dict) -> float:
        """Return confidence score (0-1) if agent can handle this request"""
        pass

    @abstractmethod
    async def process(self, query: str, context: Dict) -> Dict[str, Any]:
        """Process the query and return results"""
        pass

    def register_tool(self, tool_name: str, tool_function: callable):
        """Register a tool with this agent"""
        self.tools[tool_name] = tool_function

    def get_metadata(self) -> Dict:
        """Return agent metadata for registry"""
        return {
            "name": self.name,
            "description": self.description,
            "capabilities": self.capabilities,
            "tools": list(self.tools.keys()),
            "state": self.state
        }
```

### 2. Agent Registry
```python
# orchestration/agent_registry.py
from typing import Dict, List
from agents.base_agent import BaseAgent

class AgentRegistry:
    """Central registry for all available agents"""

    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}

    def register(self, agent: BaseAgent):
        """Register a new agent"""
        self.agents[agent.name] = agent

    def get_agent(self, name: str) -> BaseAgent:
        """Get agent by name"""
        return self.agents.get(name)

    def find_agents_by_capability(self, capability: str) -> List[BaseAgent]:
        """Find all agents with a specific capability"""
        return [
            agent for agent in self.agents.values()
            if capability in agent.capabilities
        ]

    def get_all_agents(self) -> List[Dict]:
        """Get metadata for all registered agents"""
        return [agent.get_metadata() for agent in self.agents.values()]
```

### 3. Intent Classifier
```python
# orchestration/intent_classifier.py
from enum import Enum
from typing import List, Dict
import re

class IntentType(Enum):
    MARKET_DATA = "market_data"
    TECHNICAL_ANALYSIS = "technical_analysis"
    FUNDAMENTAL_ANALYSIS = "fundamental_analysis"
    RISK_ASSESSMENT = "risk_assessment"
    PORTFOLIO_ANALYSIS = "portfolio_analysis"
    ECONOMIC_FORECAST = "economic_forecast"
    NEWS_SENTIMENT = "news_sentiment"
    MULTI_AGENT = "multi_agent"

class IntentClassifier:
    """Classify user queries into intents"""

    def __init__(self):
        self.intent_keywords = {
            IntentType.MARKET_DATA: [
                "price", "quote", "current", "latest", "widget", "data"
            ],
            IntentType.TECHNICAL_ANALYSIS: [
                "technical", "chart", "rsi", "macd", "indicator", "pattern",
                "support", "resistance", "trend", "candlestick"
            ],
            IntentType.FUNDAMENTAL_ANALYSIS: [
                "fundamental", "financials", "earnings", "revenue", "profit",
                "balance sheet", "income statement", "valuation", "pe ratio"
            ],
            IntentType.RISK_ASSESSMENT: [
                "risk", "volatility", "var", "correlation", "stress test",
                "downside", "beta"
            ],
            IntentType.PORTFOLIO_ANALYSIS: [
                "portfolio", "diversification", "allocation", "optimize",
                "rebalance", "sharpe", "performance"
            ],
            IntentType.ECONOMIC_FORECAST: [
                "forecast", "predict", "projection", "trend", "economic",
                "gdp", "inflation", "time series"
            ],
            IntentType.NEWS_SENTIMENT: [
                "news", "sentiment", "article", "search", "latest",
                "headlines", "media", "announcement"
            ]
        }

    def classify(self, query: str) -> List[IntentType]:
        """Classify query and return list of intents"""
        query_lower = query.lower()
        detected_intents = []

        for intent, keywords in self.intent_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                detected_intents.append(intent)

        # If multiple intents detected, mark as MULTI_AGENT
        if len(detected_intents) > 1:
            return [IntentType.MULTI_AGENT] + detected_intents

        return detected_intents if detected_intents else [IntentType.MARKET_DATA]
```

### 4. Orchestrator
```python
# orchestration/orchestrator.py
from typing import Dict, List, Any
from .agent_registry import AgentRegistry
from .intent_classifier import IntentClassifier, IntentType
from .result_aggregator import ResultAggregator

class Orchestrator:
    """Main orchestrator for routing queries to specialized agents"""

    def __init__(self, registry: AgentRegistry):
        self.registry = registry
        self.intent_classifier = IntentClassifier()
        self.result_aggregator = ResultAggregator()
        self.context = {}

    async def process_query(self, query: str, context: Dict = None) -> Dict[str, Any]:
        """
        Main orchestration logic:
        1. Classify intent
        2. Select appropriate agent(s)
        3. Route query to agent(s)
        4. Aggregate results
        5. Return unified response
        """
        # Step 1: Classify intent
        intents = self.intent_classifier.classify(query)

        # Step 2: Select agents based on intents
        selected_agents = self._select_agents(intents)

        if not selected_agents:
            return {"error": "No suitable agent found for this query"}

        # Step 3: Route to agent(s)
        results = []
        for agent in selected_agents:
            try:
                result = await agent.process(query, context or {})
                results.append({
                    "agent": agent.name,
                    "result": result
                })
            except Exception as e:
                results.append({
                    "agent": agent.name,
                    "error": str(e)
                })

        # Step 4: Aggregate results
        aggregated = self.result_aggregator.aggregate(results)

        return aggregated

    def _select_agents(self, intents: List[IntentType]) -> List:
        """Select agents based on detected intents"""
        agent_mapping = {
            IntentType.MARKET_DATA: "MarketDataAgent",
            IntentType.TECHNICAL_ANALYSIS: "TechnicalAnalysisAgent",
            IntentType.FUNDAMENTAL_ANALYSIS: "FundamentalAnalysisAgent",
            IntentType.RISK_ASSESSMENT: "RiskAnalyticsAgent",
            IntentType.PORTFOLIO_ANALYSIS: "PortfolioManagementAgent",
            IntentType.ECONOMIC_FORECAST: "EconomicAnalysisAgent",
            IntentType.NEWS_SENTIMENT: "NewsSentimentAgent"
        }

        selected = []
        for intent in intents:
            if intent == IntentType.MULTI_AGENT:
                continue  # Skip the MULTI_AGENT marker

            agent_name = agent_mapping.get(intent)
            if agent_name:
                agent = self.registry.get_agent(agent_name)
                if agent:
                    selected.append(agent)

        return selected
```

---

## Migration Strategy

### Phase 1: Foundation (Week 1)
1. ✅ Create `config.py` for centralized configuration
2. ✅ Implement base agent class (`agents/base_agent.py`)
3. ✅ Build orchestration infrastructure:
   - `orchestration/agent_registry.py`
   - `orchestration/intent_classifier.py`
   - `orchestration/orchestrator.py`
   - `orchestration/result_aggregator.py`

### Phase 2: Agent Implementation (Week 2-3)
1. ✅ Implement Market Data Agent + tools
2. ✅ Implement Technical Analysis Agent + tools
3. ✅ Implement News & Sentiment Agent + tools
4. ⚠️ Implement remaining 4 agents (Fundamental, Risk, Portfolio, Economic)

### Phase 3: Tool Migration (Week 3-4)
1. ✅ Refactor existing processors into `tools/` structure
2. ✅ Create tool registration system
3. ✅ Migrate visualization utilities
4. ✅ Test each tool independently

### Phase 4: Integration (Week 4-5)
1. ✅ Modify `main.py` to use orchestrator
2. ✅ Update prompts for agent-specific contexts
3. ✅ Implement result aggregation logic
4. ✅ Add error handling and fallbacks
5. ✅ Maintain backward compatibility

### Phase 5: Testing & Optimization (Week 5-6)
1. ⚠️ End-to-end testing of multi-agent flows
2. ⚠️ Performance optimization
3. ⚠️ Load testing
4. ⚠️ Documentation updates

---

## Example Agent Implementation

### Market Data Agent (Example)
```python
# agents/market_data_agent.py
from .base_agent import BaseAgent
from typing import Dict, Any
from tools.market_data.widget_processor import WidgetProcessor
from tools.market_data.api_data_fetcher import APIDataFetcher
from tools.market_data.data_validator import DataValidator

class MarketDataAgent(BaseAgent):
    """Agent specialized in retrieving and processing market data"""

    def __init__(self):
        super().__init__(
            name="MarketDataAgent",
            description="Retrieves and processes market data from OpenBB widgets and APIs",
            capabilities=[
                "widget_data_extraction",
                "price_data_retrieval",
                "data_validation",
                "real_time_data",
                "historical_data"
            ]
        )

        # Register tools
        self.widget_processor = WidgetProcessor()
        self.api_fetcher = APIDataFetcher()
        self.validator = DataValidator()

        self.register_tool("get_widget_data", self.widget_processor.extract_data)
        self.register_tool("validate_data", self.validator.validate)
        self.register_tool("fetch_api_data", self.api_fetcher.fetch)

    def can_handle(self, intent: str, context: Dict) -> float:
        """Return confidence score for handling this request"""
        if intent in ["market_data", "price_data", "widget_data"]:
            return 0.95
        return 0.0

    async def process(self, query: str, context: Dict) -> Dict[str, Any]:
        """Process market data requests"""
        # Extract widget data if present
        widget_data = context.get("widget_data")

        if widget_data:
            # Process widget data
            processed = await self.widget_processor.process(widget_data)
            validated = self.validator.validate(processed)

            return {
                "status": "success",
                "data": validated,
                "metadata": {
                    "agent": self.name,
                    "tools_used": ["widget_processor", "data_validator"]
                }
            }

        return {"status": "error", "message": "No widget data provided"}
```

---

## Benefits of This Architecture

### 1. **Modularity**
- Each agent is self-contained with focused responsibilities
- Easy to add new agents without affecting existing ones
- Clear separation of concerns

### 2. **Scalability**
- Agents can be scaled independently
- Tools can be distributed across services
- Parallel execution of multiple agents

### 3. **Maintainability**
- Tool organization by domain (6-7 tools per agent)
- Clear boundaries between components
- Easier testing and debugging

### 4. **Flexibility**
- Easy to add new capabilities by creating new agents
- Tools can be shared across agents if needed
- Orchestrator can route to multiple agents for complex queries

### 5. **Performance**
- Agents can be cached and reused
- Parallel agent execution for multi-intent queries
- Optimized tool selection per agent

---

## Configuration Management

### config.py (NEW)
```python
from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    # Server
    server_host: str = "0.0.0.0"
    server_port: int = 7777

    # LLM
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "gemma2:9b"

    # OpenBB
    openbb_agent_name: str = "OpenBB Comprehensive Agent"
    openbb_agent_description: str = "Multi-agent financial analysis system"
    openbb_agent_image: str = "openbb-logo.png"

    # Orchestration
    enable_orchestration: bool = True
    max_parallel_agents: int = 3
    agent_timeout: int = 30  # seconds

    # Feature Flags
    enable_widget_intelligence: bool = True
    enable_ml_predictions: bool = True
    enable_caching: bool = True

    # Agents
    enabled_agents: List[str] = [
        "MarketDataAgent",
        "TechnicalAnalysisAgent",
        "FundamentalAnalysisAgent",
        "RiskAnalyticsAgent",
        "PortfolioManagementAgent",
        "EconomicAnalysisAgent",
        "NewsSentimentAgent"
    ]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
```

---

## Success Metrics

### Completion Criteria:
- ✅ 7 specialized agents implemented
- ✅ 48 tools distributed (6-7 per agent)
- ✅ Orchestrator routing queries correctly
- ✅ Multi-agent queries handled (e.g., "technical + news")
- ✅ All existing functionality preserved
- ✅ Response time < 3 seconds for single-agent queries
- ✅ Response time < 5 seconds for multi-agent queries
- ✅ 95% test coverage
- ✅ Documentation complete

---

## Next Steps

1. **Review & Approve** this architecture plan
2. **Clarify** any questions or adjustments needed
3. **Begin Implementation** starting with Phase 1 (Foundation)
4. **Iterate** based on feedback and testing

---

**Ready to proceed with implementation? Let me know if you'd like to adjust the agent definitions, tool distributions, or any other aspect of this architecture!**
