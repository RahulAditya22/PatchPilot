# PatchPilot

AI Software Engineering Agent — autonomously understands repos, plans changes, modifies code, runs tests, and produces PR-ready diffs.

## Quick Start

```bash
# Clone and setup
git clone https://github.com/RahulAditya22/PatchPilot.git
cd PatchPilot
cp .env.example .env  # Fill in your API keys
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# Run via CLI
patchpilot run --repo https://github.com/user/repo --issue "Fix the login bug"

# Run via API
patchpilot serve
# Then POST to http://localhost:8000/api/v1/run

# Run with Docker
docker compose up
```

## Architecture

```
User/CLI/API
     │
     ▼
┌─────────────────────────────────────────────┐
│              Orchestrator Agent              │
└──────────┬──────────────────────────────────┘
           │
     ┌─────┼──────────────┐
     ▼     ▼              ▼
  Repo    Issue      Codebase
  Ingester Parser    Indexer
     │      │           │
     └──────┼───────────┘
            ▼
    Planning Agent  →  Coding Agent  →  Sandbox/Tests
            │                               │
            └── Failure Analyst ← ──────────┘
                     │
               Retry Loop (max N)
                     │
               Security Checks
                     │
              Report + Diff
```

## Components

| Component | Description | Model Attribution |
|-----------|-------------|-------------------|
| Planning Agent | Breaks issues into actionable plans | Claude Opus |
| Coding Agent | Generates code modifications | Claude Opus |
| Failure Analyst | Diagnoses test failures, suggests fixes | Claude Opus |
| Orchestrator | Main pipeline control loop | Claude Opus |
| Security Checker | Secrets scanning + unsafe code detection | Claude Opus |
| Repo Ingester | Clones and parses repositories | Gemini |
| Codebase Indexer | Chunks and embeds code for search | Gemini |
| Code Search | Semantic + lexical hybrid search | Gemini |
| Sandbox Runner | Isolated test execution | Gemini |
| API/CLI | FastAPI + Click interfaces | Gemini |

## Configuration

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

See `.env.example` for all required and optional environment variables.

## Development

```bash
# Run tests
pytest tests/ -v --cov=src/patchpilot

# Lint
ruff check src/ tests/

# Type check
mypy src/

# Format
ruff format src/ tests/
```

## License

MIT
