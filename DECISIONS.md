# DECISIONS.md — PatchPilot Design Decisions & Model Attribution

## Architecture Decisions

### 1. LLM Provider Abstraction
**Decision:** Unified `LLMClient` class supporting both Anthropic Claude and Google Gemini via a single interface.
**Rationale:** Allows runtime model selection per-task. The orchestrator uses Claude for reasoning-heavy work (planning, coding, diagnosis) and can use Gemini for simpler tasks.
**Assumption:** API keys are provided by the user. Tests mock all LLM calls.

### 2. Vector Database: ChromaDB (Embedded)
**Decision:** Use ChromaDB in embedded/persistent mode rather than a separate server.
**Rationale:** Zero external infrastructure required. Single `pip install` gets everything. ChromaDB is fast enough for repo-scale codebases (typically <100k chunks).
**Trade-off:** Not suitable for extremely large monorepos. For those, a hosted vector DB (Pinecone, Weaviate) would be needed.

### 3. Embeddings: sentence-transformers (Local)
**Decision:** Use `all-MiniLM-L6-v2` running locally via sentence-transformers.
**Rationale:** No API cost for indexing/search. Model is small (~80MB), fast, and produces good quality embeddings for code search. Runs on CPU.
**Assumption:** First run downloads the model (~80MB). Docker image caches it.

### 4. Sandbox: Subprocess Fallback
**Decision:** Dual sandbox backend — Docker (preferred) and subprocess (fallback).
**Rationale:** Docker is not installed on the development machine. The subprocess backend uses Python's `asyncio.create_subprocess_exec` with configurable timeouts. Docker is tested in CI.
**Risk:** Subprocess isolation is weaker than Docker. For production use, Docker is strongly recommended.

### 5. Python AST-Based Code Chunking
**Decision:** Chunk Python files by AST nodes (functions, classes) rather than fixed line counts.
**Rationale:** Semantic chunking preserves function/class boundaries, producing more meaningful search results. Falls back to line-based chunking for non-Python files.

### 6. Retry Loop Design
**Decision:** Bounded retry loop (default 3 attempts) with failure analysis between each retry.
**Rationale:** The Failure Analyst provides structured diagnostics that guide the next fix attempt. Unbounded retries risk infinite loops and cost blowout.
**Exit conditions:** All tests pass, or max retries exhausted.

### 7. Security Checks: Regex-Based Static Analysis
**Decision:** Pure regex pattern matching for secrets and unsafe code detection.
**Rationale:** Fast, deterministic, no external tool dependencies. Catches the most common cases (API keys, passwords, eval/exec, shell=True). Not a replacement for a full SAST tool, but sufficient as a pre-commit gate.

### 8. Test Framework: pytest
**Decision:** Use pytest with pytest-asyncio, pytest-cov, and pytest-mock.
**Rationale:** Industry standard for Python testing. Supports async tests, fixtures, parametrization, and coverage reporting.

### 9. API Framework: FastAPI
**Decision:** FastAPI with Pydantic v2 models for request/response validation.
**Rationale:** Async-native, automatic OpenAPI docs, excellent type checking support, Pydantic v2 integration.

### 10. No External Database
**Decision:** No persistent database (PostgreSQL, Redis, etc.).
**Rationale:** PatchPilot is a CLI/API tool that processes individual issues. State is maintained in-memory during a run and persisted as trace files and reports. No need for a database.

---

## Ambiguity Resolutions

| Ambiguity | Resolution | Reasoning |
|-----------|------------|-----------|
| Which Anthropic model to use | `claude-sonnet-4-20250514` | Best balance of capability and cost for code tasks |
| Which Gemini model to use | `gemini-2.0-flash` | Fast and cost-effective for simple tasks |
| How to handle repos without tests | Auto-generate test skeletons | Better than skipping test execution entirely |
| Max file size for indexing | 500KB per file | Prevents OOM on generated/minified files |
| How to detect test framework | Check for pytest.ini, pyproject.toml [tool.pytest] | Default to `python -m pytest` |
| PR creation vs diff only | Generate diff only (no auto-PR) | User should review before creating PR |

---

## Model Attribution

### Claude Opus (High-Stakes Reasoning)

| File | Component | Commit Tag |
|------|-----------|------------|
| `src/patchpilot/agents/planner.py` | Planning Agent | `feat(opus)` |
| `src/patchpilot/agents/coder.py` | Coding Agent | `feat(opus)` |
| `src/patchpilot/agents/failure_analyst.py` | Failure Analyst | `feat(opus)` |
| `src/patchpilot/agents/orchestrator.py` | Orchestrator | `feat(opus)` |
| `src/patchpilot/tools/security.py` | Security Checker | `feat(opus)` |
| `src/patchpilot/llm/client.py` | LLM Client (interface design) | `feat(opus)` |
| `src/patchpilot/llm/schemas.py` | Tool-calling schemas | `feat(opus)` |
| `src/patchpilot/models.py` | Data models (architecture) | `feat(opus)` |
| `src/patchpilot/config.py` | Configuration | `feat(opus)` |
| `tests/unit/test_planner.py` | Planning Agent tests | `test(opus)` |
| `tests/unit/test_coder.py` | Coding Agent tests | `test(opus)` |
| `tests/unit/test_failure_analyst.py` | Failure Analyst tests | `test(opus)` |
| `tests/unit/test_orchestrator.py` | Orchestrator tests | `test(opus)` |
| `tests/unit/test_security.py` | Security tests | `test(opus)` |
| `tests/eval/test_eval_suite.py` | Evaluation suite | `test(opus)` |
| `tests/eval/benchmark.py` | Evaluation runner | `test(opus)` |
| `tests/integration/test_end_to_end.py` | Integration tests | `test(opus)` |

### Gemini (Scaffolding & Wiring)

| File | Component | Commit Tag |
|------|-----------|------------|
| `src/patchpilot/tools/repo_ingester.py` | Repo Ingester | `feat(gemini)` |
| `src/patchpilot/tools/issue_parser.py` | Issue Parser | `feat(gemini)` |
| `src/patchpilot/tools/indexer.py` | Codebase Indexer | `feat(gemini)` |
| `src/patchpilot/tools/code_search.py` | Code Search | `feat(gemini)` |
| `src/patchpilot/tools/dependency_analyzer.py` | Dependency Analyzer | `feat(gemini)` |
| `src/patchpilot/tools/sandbox.py` | Sandbox Runner | `feat(gemini)` |
| `src/patchpilot/tools/diff_generator.py` | Diff Generator | `feat(gemini)` |
| `src/patchpilot/tools/test_generator.py` | Test Generator | `feat(gemini)` |
| `src/patchpilot/llm/token_tracker.py` | Token Tracker | `feat(gemini)` |
| `src/patchpilot/observability/logger.py` | Structured Logger | `feat(gemini)` |
| `src/patchpilot/observability/tracer.py` | Agent Tracer | `feat(gemini)` |
| `src/patchpilot/api/app.py` | FastAPI App | `feat(gemini)` |
| `src/patchpilot/api/routes.py` | API Routes | `feat(gemini)` |
| `src/patchpilot/cli/main.py` | CLI Entry Point | `feat(gemini)` |
| `Dockerfile` | Docker config | `feat(gemini)` |
| `docker-compose.yml` | Docker Compose | `feat(gemini)` |
| `.github/workflows/ci.yml` | GitHub Actions CI | `feat(gemini)` |
| `.env.example` | Environment template | `feat(gemini)` |
| `.gitignore` | Git ignore rules | `feat(gemini)` |
| `README.md` | Documentation | `docs(gemini)` |
| `DECISIONS.md` | Decisions & attribution | `docs(gemini)` |
| `tests/unit/test_config.py` | Config tests | `test(gemini)` |
| `tests/unit/test_models.py` | Model tests | `test(gemini)` |
| `tests/unit/test_token_tracker.py` | Token tracker tests | `test(gemini)` |
| `tests/unit/test_repo_ingester.py` | Repo ingester tests | `test(gemini)` |
| `tests/unit/test_issue_parser.py` | Issue parser tests | `test(gemini)` |
| `tests/unit/test_diff_generator.py` | Diff generator tests | `test(gemini)` |
| `tests/unit/test_code_search.py` | Code search tests | `test(gemini)` |
| `tests/unit/test_sandbox.py` | Sandbox tests | `test(gemini)` |
| `tests/unit/test_indexer.py` | Indexer tests | `test(gemini)` |

---

## Interface Contract Verification

All Gemini-produced modules that call into Opus-written core components have been verified for interface compatibility:

- **API routes** → `Orchestrator.run(repo_url, issue_text, issue_title, max_retries)` ✓
- **CLI** → `Orchestrator.run(...)` ✓
- **Token tracker** → `TokenUsage(model, input_tokens, output_tokens, estimated_cost_usd)` ✓
- **Sandbox** → returns `TestResult` consumed by `FailureAnalyst.analyze()` ✓
- **Issue parser** → returns `IssueRequirement` consumed by `PlanningAgent.create_plan()` ✓
- **Code search** → returns `list[dict]` consumed by orchestrator context building ✓
- **Diff generator** → consumes `list[CodeChange]` from `CodingAgent.apply_plan()` ✓
