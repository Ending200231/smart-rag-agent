# Contributing to Smart RAG Agent

Thanks for your interest in contributing!

## Development Setup

```bash
git clone https://github.com/YOUR_USERNAME/smart-rag-agent.git
cd smart-rag-agent
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\Activate.ps1  # Windows PowerShell
pip install -e ".[all]"
```

## Running Tests

```bash
pytest tests/
```

## Code Style

This project uses [Ruff](https://github.com/astral-sh/ruff) for linting and formatting:

```bash
ruff check src/
ruff format src/
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes and add tests
4. Run tests and linting
5. Submit a PR with a clear description
