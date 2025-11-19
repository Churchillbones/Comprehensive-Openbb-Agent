# Phase 2: Seven Specialized Agents - Implementation Plan

## Overview
This phase implements 7 specialized financial analysis agents, each with 6-7 domain-specific tools. Each agent will inherit from `BaseAgent` and leverage existing processors/utilities where possible.

---

## Agent 1: Market Data Agent

**Purpose:** Retrieve and process market data from OpenBB widgets, APIs, and real-time sources

**Capabilities:**
- Widget data extraction and parsing
- API data fetching and normalization
- Data quality validation
- Real-time and historical data retrieval
- OHLCV data processing
- Multi-source data integration

**7 Tools:**

### 1.1 Widget Data Processor
- **Source:** `processors/widgets.py` → `tools/market_data/widget_processor.py`
- **Function:** `process_widget_data()`, `extract_data_for_visualization()`
- **Purpose:** Extract and process data from OpenBB workspace widgets
- **Input:** Widget data JSON
- **Output:** Cleaned, structured data ready for analysis

### 1.2 Widget Intelligence Analyzer
- **Source:** `processors/widget_intelligence.py` → `tools/market_data/widget_intelligence.py`
- **Function:** `WidgetIntelligence.analyze_widget()`, `detect_widget_type()`
- **Purpose:** Analyze widget context, detect type, extract symbols/timeframes
- **Input:** Widget data
- **Output:** Widget metadata (type, symbols, timeframe, context)

### 1.3 API Data Fetcher
- **Source:** `processors/api_data.py` → `tools/market_data/api_data_fetcher.py`
- **Function:** `process_api_data()`, `prepare_api_data_for_visualization()`
- **Purpose:** Fetch and prepare data from external APIs
- **Input:** API endpoint, parameters
- **Output:** Normalized API response data

### 1.4 API Data Processor (Advanced)
- **Source:** `processors/api_data_processor.py` → `tools/market_data/api_data_processor.py`
- **Function:** `APIDataProcessor.normalize()`, `merge_data_sources()`
- **Purpose:** Advanced normalization, merging multi-source data
- **Input:** Raw API data from multiple sources
- **Output:** Unified, normalized dataset

### 1.5 Data Validator
- **Source:** `processors/data_validator.py` → `tools/market_data/data_validator.py`
- **Function:** `DataValidator.validate()`, `check_data_quality()`
- **Purpose:** Validate data integrity, completeness, and quality
- **Input:** Dataset
- **Output:** Validation report, cleaned data

### 1.6 OHLCV Extractor
- **Source:** NEW - Create from `widgets.py` logic
- **Function:** `extract_ohlcv()`, `calculate_returns()`
- **Purpose:** Extract OHLCV (Open, High, Low, Close, Volume) data specifically
- **Input:** Price data
- **Output:** OHLCV dataframe with calculated returns

### 1.7 Real-time Data Stream Manager
- **Source:** NEW - Wrapper around existing data sources
- **Function:** `stream_real_time_data()`, `get_latest_quote()`
- **Purpose:** Manage real-time data streams and latest quotes
- **Input:** Symbol, data source
- **Output:** Real-time data stream or latest quote

**Implementation File:** `comprehensive_agent/agents/market_data_agent.py`

---

## Agent 2: Technical Analysis Agent

**Purpose:** Perform technical analysis with charts, indicators, and pattern recognition

**Capabilities:**
- Chart generation (line, candlestick, OHLC, bar)
- Technical indicator calculation (RSI, MACD, Bollinger Bands, etc.)
- Trend detection and analysis
- Support/resistance level identification
- Volume analysis
- Pattern recognition
- Interactive chart features

**7 Tools:**

### 2.1 Chart Generator
- **Source:** `visualizations/charts.py` → `tools/technical/chart_generator.py`
- **Function:** `generate_charts()`, `detect_chart_types()`
- **Purpose:** Generate various chart types for technical analysis
- **Input:** Price data, chart type
- **Output:** Chart object for OpenBB

### 2.2 Financial Chart Generator
- **Source:** `visualizations/financial_charts.py` → `tools/technical/financial_chart_generator.py`
- **Function:** Candlestick, OHLC, heatmaps, waterfall charts
- **Purpose:** Specialized financial charts
- **Input:** OHLCV data
- **Output:** Financial-specific chart objects

### 2.3 Interactive Chart Builder
- **Source:** `visualizations/interactive_charts.py` → `tools/technical/interactive_chart_builder.py`
- **Function:** Add annotations, overlays, technical indicators to charts
- **Purpose:** Create interactive, annotated charts
- **Input:** Base chart, annotations
- **Output:** Interactive chart with features

### 2.4 Technical Indicator Calculator
- **Source:** NEW - Extract from `ml_widget_bridge.py` + add more
- **Function:** `calculate_rsi()`, `calculate_macd()`, `calculate_bollinger_bands()`, etc.
- **Purpose:** Calculate all major technical indicators
- **Input:** Price data, indicator parameters
- **Output:** Indicator values, signals

### 2.5 Trend Detector
- **Source:** NEW - From `ml_widget_bridge.py` trend analysis
- **Function:** `detect_trend()`, `identify_trend_strength()`, `find_trend_reversals()`
- **Purpose:** Identify trends and trend changes
- **Input:** Price data
- **Output:** Trend direction, strength, reversal points

### 2.6 Support/Resistance Finder
- **Source:** NEW
- **Function:** `find_support_levels()`, `find_resistance_levels()`, `identify_breakouts()`
- **Purpose:** Identify key price levels
- **Input:** Price history
- **Output:** Support/resistance levels, breakout signals

### 2.7 Volume Analyzer
- **Source:** NEW - Extract from existing volume analysis
- **Function:** `analyze_volume_profile()`, `detect_volume_spikes()`, `calculate_vwap()`
- **Purpose:** Volume-based analysis
- **Input:** Volume and price data
- **Output:** Volume analysis metrics, VWAP

**Implementation File:** `comprehensive_agent/agents/technical_analysis_agent.py`

---

## Agent 3: Fundamental Analysis Agent

**Purpose:** Analyze financial statements, metrics, and valuations

**Capabilities:**
- Financial statement processing
- Metrics calculation (P/E, ROE, etc.)
- Valuation models (DCF, multiples)
- Growth analysis
- Profitability analysis
- Spreadsheet data processing
- Financial table generation

**7 Tools:**

### 3.1 Spreadsheet Processor
- **Source:** `processors/spreadsheet.py` → `tools/fundamental/spreadsheet_processor.py`
- **Function:** `process_spreadsheet_data()`, multi-sheet handling
- **Purpose:** Process Excel/CSV files with financial data
- **Input:** Spreadsheet file
- **Output:** Structured financial data

### 3.2 Advanced Spreadsheet Processor
- **Source:** `processors/spreadsheet_processor.py` → `tools/fundamental/advanced_spreadsheet_processor.py`
- **Function:** `SpreadsheetProcessor` with advanced parsing
- **Purpose:** Handle complex spreadsheets, multiple tables
- **Input:** Complex spreadsheet
- **Output:** Parsed financial statements

### 3.3 Financial Metrics Calculator
- **Source:** NEW - From `ml_widget_bridge.py` financial analysis
- **Function:** `calculate_pe_ratio()`, `calculate_roe()`, `calculate_growth_rate()`, etc.
- **Purpose:** Calculate financial ratios and metrics
- **Input:** Financial statement data
- **Output:** Calculated metrics (P/E, P/B, ROE, ROA, etc.)

### 3.4 Valuation Model Engine
- **Source:** NEW
- **Function:** `dcf_valuation()`, `comparable_company_analysis()`, `precedent_transactions()`
- **Purpose:** Run valuation models
- **Input:** Financial data, assumptions
- **Output:** Valuation estimates, sensitivity analysis

### 3.5 Growth Analyzer
- **Source:** NEW - From `ml_widget_bridge.py`
- **Function:** `analyze_revenue_growth()`, `analyze_earnings_growth()`, `calculate_cagr()`
- **Purpose:** Analyze growth trends
- **Input:** Historical financials
- **Output:** Growth rates, trends, forecasts

### 3.6 Profitability Analyzer
- **Source:** NEW - From `ml_widget_bridge.py`
- **Function:** `analyze_margins()`, `analyze_profitability_trends()`, `calculate_roic()`
- **Purpose:** Analyze profitability metrics
- **Input:** Income statement data
- **Output:** Margin analysis, profitability trends

### 3.7 Financial Table Generator
- **Source:** `visualizations/tables.py` → `tools/fundamental/table_generator.py`
- **Function:** `generate_tables()`, format financial statements
- **Purpose:** Create formatted tables for financial data
- **Input:** Financial data
- **Output:** Formatted table objects

**Implementation File:** `comprehensive_agent/agents/fundamental_analysis_agent.py`

---

## Agent 4: Risk Analytics Agent

**Purpose:** Risk assessment, volatility analysis, and correlation studies

**Capabilities:**
- Volatility analysis (historical, implied)
- Value at Risk (VaR) calculation
- Correlation and covariance analysis
- Stress testing and scenario analysis
- Risk metrics dashboard
- Sentiment-price correlation

**6 Tools:**

### 4.1 Volatility Analyzer
- **Source:** `core/ml_widget_bridge.py` → `tools/risk/volatility_analyzer.py`
- **Function:** From `_analyze_price_data()` - volatility calculation
- **Purpose:** Calculate historical and realized volatility
- **Input:** Price data
- **Output:** Volatility metrics (std dev, annualized vol, vol trends)

### 4.2 VaR Calculator
- **Source:** NEW - Use `core/model_engine.py` for distributions
- **Function:** `calculate_var()`, `calculate_cvar()`, `parametric_var()`, `historical_var()`
- **Purpose:** Calculate Value at Risk and Conditional VaR
- **Input:** Returns data, confidence level
- **Output:** VaR estimates, CVaR

### 4.3 Correlation Engine
- **Source:** `utils/data_correlator.py` → `tools/risk/correlation_engine.py`
- **Function:** `correlate_sentiment_with_prices()`, `calculate_correlation_matrix()`
- **Purpose:** Analyze correlations between assets and factors
- **Input:** Multiple asset data, sentiment data
- **Output:** Correlation matrix, key relationships

### 4.4 Stress Testing Framework
- **Source:** NEW
- **Function:** `run_stress_test()`, `scenario_analysis()`, `monte_carlo_simulation()`
- **Purpose:** Test portfolio under various scenarios
- **Input:** Portfolio data, stress scenarios
- **Output:** Stress test results, worst-case scenarios

### 4.5 Risk Metrics Dashboard
- **Source:** NEW - Combines multiple risk metrics
- **Function:** `generate_risk_dashboard()`, `calculate_sharpe_ratio()`, `calculate_beta()`
- **Purpose:** Comprehensive risk metrics overview
- **Input:** Portfolio/asset data
- **Output:** Risk dashboard with multiple metrics

### 4.6 Drawdown Analyzer
- **Source:** NEW
- **Function:** `calculate_max_drawdown()`, `analyze_drawdown_periods()`, `recovery_time()`
- **Purpose:** Analyze drawdown characteristics
- **Input:** Price/returns data
- **Output:** Max drawdown, drawdown history, recovery analysis

**Implementation File:** `comprehensive_agent/agents/risk_analytics_agent.py`

---

## Agent 5: Portfolio Management Agent

**Purpose:** Portfolio optimization, allocation, and performance tracking

**Capabilities:**
- Portfolio analysis and metrics
- Diversification measurement
- Asset allocation optimization
- Performance attribution
- Rebalancing recommendations
- Portfolio visualization
- Modern Portfolio Theory (MPT)

**7 Tools:**

### 5.1 Portfolio Analyzer
- **Source:** `core/ml_widget_bridge.py` → `tools/portfolio/portfolio_analyzer.py`
- **Function:** From `_analyze_portfolio_data()`
- **Purpose:** Analyze portfolio composition and performance
- **Input:** Portfolio holdings
- **Output:** Portfolio metrics, composition analysis

### 5.2 Diversification Calculator
- **Source:** NEW - Use `core/model_engine.py` clustering
- **Function:** `calculate_diversification_ratio()`, `analyze_concentration()`, `sector_allocation()`
- **Purpose:** Measure portfolio diversification
- **Input:** Portfolio holdings
- **Output:** Diversification metrics, concentration risk

### 5.3 Asset Allocation Optimizer
- **Source:** NEW - Use `core/model_engine.py` for optimization
- **Function:** `optimize_allocation()`, `efficient_frontier()`, `target_return_allocation()`
- **Purpose:** Find optimal asset allocation
- **Input:** Expected returns, covariance, constraints
- **Output:** Optimal weights, efficient frontier

### 5.4 Performance Attribution Engine
- **Source:** NEW
- **Function:** `attribute_performance()`, `factor_returns()`, `sector_contribution()`
- **Purpose:** Attribute returns to factors
- **Input:** Portfolio returns, benchmark, factors
- **Output:** Performance attribution breakdown

### 5.5 Rebalancing Engine
- **Source:** NEW
- **Function:** `calculate_rebalancing_trades()`, `tax_aware_rebalancing()`, `threshold_rebalancing()`
- **Purpose:** Generate rebalancing recommendations
- **Input:** Current portfolio, target allocation
- **Output:** Rebalancing trades, cost estimates

### 5.6 Portfolio Visualizer
- **Source:** `visualizations/financial_charts.py` (heatmaps) → `tools/portfolio/visualizer.py`
- **Function:** Create allocation pie charts, performance charts, heatmaps
- **Purpose:** Visualize portfolio data
- **Input:** Portfolio data
- **Output:** Visualization objects

### 5.7 Risk-Return Calculator
- **Source:** NEW
- **Function:** `calculate_sharpe_ratio()`, `calculate_sortino_ratio()`, `calculate_information_ratio()`
- **Purpose:** Calculate risk-adjusted returns
- **Input:** Portfolio returns, benchmark
- **Output:** Sharpe, Sortino, Information ratios

**Implementation File:** `comprehensive_agent/agents/portfolio_management_agent.py`

---

## Agent 6: Economic Analysis Agent

**Purpose:** Economic forecasting, time series analysis, and macroeconomic insights

**Capabilities:**
- Time series forecasting (ARIMA, Prophet, LSTM)
- Economic indicator analysis
- Trend prediction and regression
- Feature engineering for ML models
- Scenario modeling
- Market intelligence reports

**7 Tools:**

### 6.1 Time Series Forecaster
- **Source:** `core/model_engine.py` → `tools/economic/forecasting_engine.py`
- **Function:** `ModelEngine.train_time_series()`, `predict_time_series()`
- **Purpose:** Forecast future values using time series models
- **Input:** Historical data
- **Output:** Forecasts, confidence intervals

### 6.2 ARIMA Forecaster
- **Source:** NEW - Extend `model_engine.py`
- **Function:** `fit_arima()`, `auto_arima()`, `forecast_arima()`
- **Purpose:** ARIMA-specific forecasting
- **Input:** Time series data
- **Output:** ARIMA forecasts, model diagnostics

### 6.3 Feature Engineer
- **Source:** `processors/feature_engineering.py` → `tools/economic/feature_engineer.py`
- **Function:** `FeatureEngineer` class - create ML features
- **Purpose:** Generate features for predictive models
- **Input:** Raw data
- **Output:** Engineered features (lags, rolling stats, etc.)

### 6.4 Regression Model Engine
- **Source:** `core/model_engine.py` → `tools/economic/regression_engine.py`
- **Function:** From `ModelEngine` - regression methods
- **Purpose:** Run regression analysis
- **Input:** Features, target variable
- **Output:** Regression results, predictions

### 6.5 Trend Predictor
- **Source:** NEW - From `ml_widget_bridge.py` momentum analysis
- **Function:** `predict_trend()`, `extrapolate_trend()`, `identify_cycles()`
- **Purpose:** Predict future trends
- **Input:** Historical data
- **Output:** Trend predictions, cycle identification

### 6.6 Economic Indicator Analyzer
- **Source:** NEW
- **Function:** `analyze_gdp()`, `analyze_inflation()`, `analyze_employment()`, etc.
- **Purpose:** Analyze macroeconomic indicators
- **Input:** Economic data
- **Output:** Indicator analysis, relationships

### 6.7 Scenario Modeler
- **Source:** NEW
- **Function:** `create_scenario()`, `run_scenarios()`, `compare_scenarios()`
- **Purpose:** Model different economic scenarios
- **Input:** Base case, scenario parameters
- **Output:** Scenario outcomes, comparisons

**Implementation File:** `comprehensive_agent/agents/economic_analysis_agent.py`

---

## Agent 7: News & Sentiment Agent

**Purpose:** News aggregation, sentiment analysis, web search, and market intelligence

**Capabilities:**
- Web search (general and financial)
- Financial news aggregation
- Sentiment analysis (TextBlob, VADER)
- Citation generation
- PDF document processing
- Market alerts

**7 Tools:**

### 7.1 Web Search Engine
- **Source:** `processors/web_search.py` → `tools/news/web_searcher.py`
- **Function:** `process_web_search()`, `detect_web_search_request()`
- **Purpose:** General web search capability
- **Input:** Search query
- **Output:** Search results

### 7.2 Financial News Searcher
- **Source:** `processors/financial_web_search.py` → `tools/news/financial_news_searcher.py`
- **Function:** `FinancialWebSearcher.search_and_analyze()`
- **Purpose:** Search for financial news specifically
- **Input:** Search query, symbols
- **Output:** Financial news results

### 7.3 Sentiment Analyzer
- **Source:** `core/ml_widget_bridge.py` → `tools/news/sentiment_analyzer.py`
- **Function:** From `_analyze_news_data()` - sentiment analysis
- **Purpose:** Analyze sentiment from text (news, social media)
- **Input:** Text content
- **Output:** Sentiment score, polarity, subjectivity

### 7.4 News Aggregator
- **Source:** NEW - Wrapper around web search
- **Function:** `aggregate_news()`, `filter_by_source()`, `deduplicate_news()`
- **Purpose:** Aggregate news from multiple sources
- **Input:** Topic, sources, date range
- **Output:** Aggregated, deduplicated news

### 7.5 Citation Generator
- **Source:** `processors/citations.py` → `tools/news/citation_generator.py`
- **Function:** `generate_citations()`
- **Purpose:** Generate citations for sources
- **Input:** Source URLs, titles, dates
- **Output:** Formatted citations

### 7.6 PDF Processor
- **Source:** `processors/pdf.py` → `tools/news/pdf_processor.py`
- **Function:** `process_pdf_data()`, `_extract_pdf_from_base64()`
- **Purpose:** Extract text from PDF documents
- **Input:** PDF file (base64 or path)
- **Output:** Extracted text content

### 7.7 Market Alert Generator
- **Source:** `utils/alerting.py` → `tools/news/alert_generator.py`
- **Function:** `send_alert()`, create market alerts
- **Purpose:** Generate alerts based on news/sentiment
- **Input:** Conditions, thresholds
- **Output:** Alert messages

**Implementation File:** `comprehensive_agent/agents/news_sentiment_agent.py`

---

## Implementation Strategy

### Step 1: Create Agent Skeletons (All 7)
For each agent, create the basic structure:
```python
class XYZAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="XYZAgent",
            description="...",
            capabilities=[...],
            priority=X
        )
        self._register_tools()

    def _register_tools(self):
        # Register 6-7 tools
        pass

    def can_handle(self, intent, context):
        # Implement capability matching
        pass

    async def process(self, query, context):
        # Implement processing logic
        pass
```

### Step 2: Migrate Tools (Priority Order)
1. **Market Data** (highest priority - foundation for others)
2. **News & Sentiment** (independent, well-defined)
3. **Technical Analysis** (depends on Market Data)
4. **Fundamental Analysis** (depends on Market Data)
5. **Risk Analytics** (depends on Market Data + Technical)
6. **Portfolio Management** (depends on Market Data + Risk)
7. **Economic Analysis** (can work independently)

### Step 3: Tool Migration Process
For each tool:
1. Copy source file to `tools/{domain}/`
2. Refactor to be standalone (no main.py dependencies)
3. Add proper imports
4. Create tool wrapper function if needed
5. Register with agent

### Step 4: Testing Each Agent
1. Unit tests for each tool
2. Integration test for agent
3. Test with orchestrator

---

## Tool Distribution Summary

| Agent | Existing Tools | New Tools | Total |
|-------|---------------|-----------|-------|
| Market Data | 5 | 2 | 7 |
| Technical Analysis | 3 | 4 | 7 |
| Fundamental Analysis | 3 | 4 | 7 |
| Risk Analytics | 1 | 5 | 6 |
| Portfolio Management | 1 | 6 | 7 |
| Economic Analysis | 2 | 5 | 7 |
| News & Sentiment | 5 | 2 | 7 |
| **TOTAL** | **20** | **28** | **48** |

**Reuse of existing processors:** 20/48 (42%)
**New tools to create:** 28/48 (58%)

---

## Timeline Estimate

- **Agent Skeletons:** 2-3 hours (all 7)
- **Tool Migration:** 6-8 hours (20 existing tools)
- **New Tool Creation:** 10-12 hours (28 new tools)
- **Integration & Testing:** 4-6 hours
- **Documentation:** 2 hours

**Total:** 24-31 hours of development

---

## Next Steps

1. Create stub implementations for all 7 agents
2. Start with Market Data Agent (foundation)
3. Migrate tools one agent at a time
4. Test each agent with orchestrator
5. Commit and push incrementally

**Ready to proceed?**
