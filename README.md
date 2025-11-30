# OpenBB Comprehensive Agent

A comprehensive OpenBB agent featuring advanced data processing, machine learning capabilities, and financial web search integration. Built with FastAPI for seamless integration with OpenBB Workspace.

## Features

- **Multi-format Data Processing**: PDF, Excel, CSV, JSON support
- **Interactive Visualizations**: Charts, tables, and financial visualizations
- **Financial Web Search**: Real-time news aggregation with sentiment analysis
- **Risk Analytics**: VaR, Sharpe ratio, and risk metrics
- **Multi-Model Support**: Ollama, OpenAI, OpenRouter, and Google GenAI

## Quick Start

### Prerequisites
- Python 3.10+
- Git

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd comprehensive-agent

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-ml.txt

# Configure environment
cp .env.example .env
# Edit .env with your settings

# Run the agent
python -m comprehensive_agent.main
# Or use: ./start.sh (Linux/Mac) or start.bat (Windows)
```

The agent will be available at `http://localhost:7777`.

## Configuration

Key environment variables (see `.env.example`):

```bash
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma2:9b
SERVER_HOST=0.0.0.0
SERVER_PORT=7777
OPENBB_API_KEY=your_api_key_here
```

## Project Structure

```
comprehensive_agent/
├── core/           # Core agent logic
├── processors/     # Data processing modules
├── visualizations/ # Chart and table generation
├── utils/          # Utility functions
├── models/         # ML model management
└── tools/          # Agent tools and utilities
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) for details.