# ⚡ QuotaDrift (Multi-AI Switchboard)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![Maintained by Tazwar Yayyyy](https://img.shields.io/badge/Maintained%20by-Tazwar%20Yayyyy-blue)](https://github.com/tazwaryayyyy)

> Created by [Tazwar Yayyyy](https://github.com/tazwaryayyyy) – built with ❤️ and a lot of API keys.

**QuotaDrift** is a self-hosted AI orchestration layer that provides automatic failover across 8 free-tier LLM providers. Built for developers who need high-reliability LLM access with zero downtime.

> 🎯 **Never worry about AI quota limits again** - QuotaDrift automatically switches between providers when one hits rate limits or fails.

## ✨ Key Features

- 🔄 **Automatic Failover** - Seamlessly switches between 8 free-tier providers
- 🧠 **Persistent Memory** - SQLite + ChromaDB for conversation history and RAG
- 🔍 **Hybrid RAG** - BM25 + Vector search for intelligent context retrieval  
- 📁 **Code Indexing** - Index and search your codebase for AI-assisted development
- 🤖 **Self-Healing Agent Loop** - AI writes code → runs it → fixes errors automatically
- 🔌 **MCP Server** - Model Context Protocol integration for Claude Desktop, Cursor, etc.
- 🐳 **Docker Ready** - Containerized deployment with one command
- 📊 **Observability** - Prometheus metrics, structured logging, health checks

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Application                │
├─────────────────────────────────────────────────────────────┤
│  Circuit Breaker │  Model Manager  │  Metrics     │
│  - Failure Track │  - Dynamic Score │  - Prometheus │
│  - Auto Recovery │  - Load Balance │  - Request ID  │
│  - Health Checks │  - Circuit State │  - JSON Logs  │
└─────────────────────────────────────────────────────────────┘
                           │
                    ┌─────────────┴─────────────┐
                    │    LiteLLM Router        │
                    │  - Fallback Chain        │
                    │  - Rate Limiting        │
                    │  - Provider Health      │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │     AI Providers            │
                    │  Groq, GitHub, Mistral     │
                    │  Silicon Flow, Hugging Face │
                    │  Cloudflare, OpenRouter    │
                    └──────────────────────────────┘
```

## 🌐 Provider Priority Chain

| Priority | Provider | Model | Free Tier | Rate Limit |
|----------|----------|-------|------------|------------|
| 1 | **Groq** | Llama 3.3 70B | Unlimited | 30 req/min |
| 2 | **GitHub Models** | GPT-4o mini | Unlimited | 4,000 req/hr |
| 3 | **GitHub Models** | Llama 3.3 70B | Unlimited | 4,000 req/hr |
| 4 | **Mistral AI** | Mistral Small | 1B tokens/month | 1 req/sec |
| 5 | **Silicon Flow** | Qwen2.5-7B | 20M tokens | 100 req/min |
| 6 | **Hugging Face** | Mistral-7B | Free inference | 30 req/min |
| 7 | **Cloudflare AI** | Llama 3.3 70B | 10k requests/day | 1 req/sec |
| 8 | **OpenRouter** | Mistral 7B | Free models | 200 req/day |

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker (optional, for enhanced features)
- API keys for AI providers

### Installation

```bash
# Clone the repository
git clone https://github.com/tazwaryayyyy/quotadrift.git
cd quotadrift

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Run the application
python main.py
```

Visit **http://localhost:8000** to start using QuotaDrift!

### Environment Variables

```bash
# Primary - Fastest
GROQ_API_KEY=gsk_your_key_here

# GitHub Models (no scopes needed)
GITHUB_TOKEN=ghp_your_token_here

# Mistral AI (1B tokens/month)
MISTRAL_API_KEY=your_key_here

# Silicon Flow (20M tokens free)
SILICONFLOW_API_KEY=your_key_here

# Hugging Face (free inference)
HUGGINGFACE_API_KEY=hf_your_token_here

# Cloudflare Workers AI (10k requests/day)
CLOUDFLARE_API_KEY=your_token_here
CLOUDFLARE_ACCOUNT_ID=your_account_id_here

# OpenRouter (fallback)
OPENROUTER_API_KEY=sk-or-your_key_here
```

## 🎨 Modern Web Interface

QuotaDrift features a clean, professional web interface built with modern design principles:

### **Design Features**
- 🎯 **Human-Centered Design** - Clean, readable interface without AI-generated aesthetics
- 🌙 **Dark Theme** - Professional dark mode with subtle color palette
- 📱 **Responsive Layout** - Works seamlessly on desktop, tablet, and mobile
- 🔧 **Resizable Panels** - Customizable sidebar and model panel widths
- ⚡ **Real-time Updates** - Live model status and streaming responses

### **Interface Components**
- **Three-Column Layout** - Sidebar (sessions) | Main chat | Model panel
- **Session Management** - Create, load, and switch between conversation sessions
- **Model Pool Dashboard** - Real-time provider health, latency, and usage metrics
- **Code Runner** - Execute code in multiple languages directly in the interface
- **File Indexing** - Drag-and-drop codebase indexing for enhanced RAG
- **Semantic Cache** - View and manage conversation cache statistics

### **User Experience**
- **Markdown Rendering** - Rich text formatting with syntax highlighting
- **Copy Buttons** - One-click code snippet copying
- **Context Display** - Expandable RAG context snippets
- **Toast Notifications** - Non-intrusive feedback messages
- **Keyboard Shortcuts** - Enter to send, Shift+Enter for newlines

### **Mobile Support**
- **Collapsible Panels** - Slide-out sidebar and model panel on mobile
- **Touch-Friendly** - Optimized buttons and interactions
- **Adaptive Layout** - Responsive design for all screen sizes

Visit **http://localhost:8000** to experience the modern interface!

## 🐳 Docker Deployment

```bash
# Build and run
docker build -t quotadrift .
docker run -p 8000:8000 --env-file .env quotadrift

# Or with docker-compose
docker-compose up -d
```

## ☁️ Cloud Deployment

### Railway
```bash
# Install Railway CLI
npm install -g @railway/cli

# Deploy
railway login
railway deploy
```

### Render
1. Connect your GitHub repository to Render
2. Create a new Web Service
3. Set environment variables
4. Deploy!

## 📡 API Overview

### Core Endpoints
- `POST /api/chat/stream` - Streaming chat with SSE
- `GET /api/model-status` - Provider health and metrics
- `POST /api/session/new` - Create new session
- `GET /api/sessions` - List all sessions
- `POST /api/session/share` - Create shareable link

### Enhanced Features
- `POST /api/run-code` - Execute code in Docker sandbox
- `POST /api/chat/edit` - Edit and regenerate messages
- `GET /api/provider-test` - Test provider connectivity
- `GET /metrics` - Prometheus metrics

### Example Usage

```python
import requests

# Streaming chat
response = requests.post(
    "http://localhost:8000/api/chat/stream",
    json={
        "session_id": 1,
        "message": "Hello, how are you?"
    },
    stream=True
)

for line in response.iter_lines():
    if line:
        print(line.decode('utf-8'))
```

## 🛠️ Tech Stack

- **Backend**: FastAPI, LiteLLM, SQLite, ChromaDB
- **Frontend**: Vanilla HTML5/CSS3, JavaScript
- **AI**: 8 free-tier providers via LiteLLM
- **Containerization**: Docker, Docker Compose
- **Monitoring**: Prometheus, structured logging
- **Testing**: pytest, ruff, black

## 🧪 Development

### Setup Development Environment

```bash
# Install development dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Lint and format
ruff check .
ruff format .
black .

# Type checking
mypy .
```

### Project Structure

```
QuotaDrift/
├── main.py                 # FastAPI application
├── router.py              # LiteLLM router integration
├── model_manager.py       # Circuit breaker & scoring
├── config.py              # Provider configuration
├── memory.py              # Session & RAG storage
├── cache.py               # Semantic caching
├── compiler.py            # State compilation
├── enhanced_agent_runner.py # Docker code execution
├── static/                 # Frontend assets
├── tests/                  # Test suite
├── .github/workflows/     # CI/CD pipeline
├── requirements.txt       # Dependencies
├── pyproject.toml         # Project configuration
└── README.md              # This file
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Quick Start for Contributors

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes
4. Run tests and linting: `pytest tests/ && ruff check . && black .`
5. Commit your changes: `git commit -m 'Add amazing feature'`
6. Push to the branch: `git push origin feature/amazing-feature`
7. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## � Maintainer

- **Tazwar Yayyyy** – creator and lead maintainer  
  [GitHub](https://github.com/tazwaryayyyy)

## �🙏 Acknowledgments

- [LiteLLM](https://github.com/BerriAI/litellm) - AI model routing
- [ChromaDB](https://github.com/chroma-core/chroma) - Vector database
- [FastAPI](https://fastapi.tiangolo.com/) - Web framework
- [Prometheus](https://prometheus.io/) - Monitoring
- [Docker](https://www.docker.com/) - Containerization

## 📞 Support

- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/tazwaryayyyy/quotadrift/issues)
- 💡 **Feature Requests**: [GitHub Discussions](https://github.com/tazwaryayyyy/quotadrift/discussions)
- 📖 **Documentation**: [Wiki](https://github.com/tazwaryayyyy/quotadrift/wiki)

---

**QuotaDrift** - Never worry about AI quota limits again. 🚀

*Built with ❤️ by Tazwar Yayyyy and the QuotaDrift community*
