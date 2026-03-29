# Personal Memory App — Plan of Action (revised)

Phases 0–3 are complete. This plan covers everything from Phase 3.5 onward.

## LLM stack (no external services)

All ML runs in-process with no external daemons:

- **Embeddings:** sentence-transformers (`all-MiniLM-L6-v2`), already integrated.
- **Intent classification:** cosine similarity against intent exemplars using the embedding model, already integrated.
- **Generative LLM:** `llama-cpp-python` loading a GGUF model file directly. Recommended starting point: Qwen2.5-3B-Instruct Q4_K_M (~2GB RAM, Metal-accelerated on macOS). Used for answer synthesis, summarization, and tag extraction.
- **Reranking:** cross-encoder model via sentence-transformers, loaded in-process.

No Ollama, no HTTP round-trips, no "is it running?" checks. If the model file exists and loads, it works for the lifetime of the process.


## Phase 3.5: Consolidation

**Goal:** clean up the structural debt from Phases 0–3 so that Phase 4 (RAG) and Phase 5 (HTTP API) can build on solid foundations without mid-phase rewrites.

Tasks:

1. Introduce `ServiceContext` — a dataclass holding `conn`, `store`, `embedder`, and later `llm`, `reranker`, and `session`. This is the single object that flows through `dispatch()`, replacing the bare `sqlite3.Connection` parameter. The CLI builds one from Click context; the future HTTP daemon builds one at startup. Test that dispatch works with a `ServiceContext` carrying a `FakeEmbeddingService` and in-memory `VectorStore`.

2. Settle `DispatchResult` on one shape. The codebase has two versions (action/item/items vs success/intent_name/data). Pick the one that serializes cleanly to JSON for Phase 5 — `DispatchResult` with `action: str`, `data: dict`, `items: list[Item]` is the likely winner. Remove the other. Update CLI formatters and all dispatcher tests to match.

3. Upgrade `_handle_search` in the dispatcher to call `hybrid_search` through `ServiceContext` instead of `keyword_search` only. Wire `QueryContext` fields (`date_from`, `date_to`, `topic`, `category`) through to the search call — right now only `raw_query`, `type_filter`, and `status_filter` are used, meaning query understanding output is silently dropped.

4. Make `route()` accept a config/context parameter so classifiers (rules, embedding) and thresholds can be injected and tested independently. This also makes it possible for the RAG pipeline to receive the full `RoutingResult` including `QueryContext` without re-parsing the input.

5. Add an `AppContext` or `App` class that owns resource lifecycle: creates the `ServiceContext` components once, exposes a `shutdown()` that closes connections and releases models. The CLI instantiates it per invocation; the daemon instantiates it once at startup. Test startup and shutdown explicitly.

**Done when:** `dispatch(svc_ctx, "what did I note about deployment?")` calls `hybrid_search` with all query understanding filters, `DispatchResult` has one canonical shape, and resource lifecycle is owned by `AppContext`. All existing tests still pass with the new signatures.


## Phase 4: RAG pipeline

**Goal:** questions get answers synthesized from stored notes.

Tasks:

1. Build context assembly: take top-N retrieval results, format as context for the LLM prompt. The retrieval call uses `ServiceContext` and the `QueryContext` filters extracted by the router.

2. Add a reranking step: use a cross-encoder model (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`) to score retrieval candidates against the query. Loaded in-process via sentence-transformers, lazy-initialized and cached on `ServiceContext` like the embedder. Test that reranking improves relevance over raw vector similarity.

3. Build an LLM client abstraction with a clear interface: `generate(prompt, system=None) -> str` and `is_available() -> bool`. Implementation uses `llama-cpp-python` with a GGUF model file. The model path is a config value; `is_available()` returns whether the model loaded successfully. Add to `ServiceContext` as `llm`, lazy-loaded on first generative call. Build a `FakeLLM` for tests that returns canned responses.

4. Build the answer generation prompt: system message, context, question. Include source references. Keep prompts in a dedicated module — not inline strings in handler code.

5. Build the answer endpoint: router dispatches question → hybrid retrieval → rerank → context assembly → LLM → response with sources. This is wired through `dispatch()` via `ServiceContext` — no new entry point needed.

6. Build non-LLM fallback: when the model file is missing or failed to load, return ranked search results directly instead of a synthesized answer. The dispatcher checks `svc_ctx.llm.is_available()` and degrades. Unlike the old Ollama approach, this should only happen on first startup before the user has downloaded a model — not intermittently mid-session.

7. Add session context: hold last N exchanges in memory on `ServiceContext`. Include in prompt for follow-up questions. Resets when `AppContext` restarts. Test that follow-up questions receive prior context.

8. Add a CLI command for model management: `memask model download` fetches the default GGUF to a known location (e.g. `~/.memask/models/`), `memask model status` shows what's loaded. First-run experience can prompt the user to run this.

**Done when:** asking `"what were my notes about the release plan?"` returns a synthesized answer citing specific notes. Tests cover: retrieval quality, prompt construction, fallback mode, session continuity. All RAG components are accessible through `ServiceContext` and testable with fakes. No external services required.


## Phase 5: Daemon HTTP API

**Goal:** all functionality is accessible over localhost HTTP.

Tasks:

1. Decide sync vs async strategy. SQLite with the `sqlite3` module is synchronous and not thread-safe. Options: (a) use Starlette/Flask with a synchronous server, (b) use FastAPI with `run_in_executor` wrapping around `ServiceContext` calls, (c) switch to `aiosqlite`. Recommendation: start with (a) or (b) — don't rewrite the storage layer for async unless profiling shows it's needed. Document the decision.

2. Build the HTTP server. `AppContext` is created once at startup, shared across all request handlers. Single `/input` endpoint accepts text, calls `dispatch(svc_ctx, text)`, returns `DispatchResult` as JSON. Additional endpoints: `/items` (list), `/search` (direct search), `/health`, `/settings`.

3. Add health endpoint: reports daemon up, LLM model loaded (name, quantization, RAM usage), job queue stats, index stats, embedding model version.

4. Add startup sequence: run migrations, initialize `AppContext` (which lazy-creates `ServiceContext` components), start background worker loop for embedding jobs. LLM and reranker load on first request, not at startup — keeps daemon start fast.

5. Background worker: run `process_all_pending` on a timer (e.g. every 5s). Use a thread or async task depending on the decision from task 1.

6. Write integration tests that hit the HTTP endpoints with a test `AppContext` using `FakeEmbeddingService` and `FakeLLM`.

7. Refactor CLI to become a thin HTTP client: `memask input "buy milk"` calls `POST /input`, `memask search "deployment"` calls `GET /search?q=deployment`. Keep direct-mode (no daemon) as a fallback for offline use.

**Done when:** `curl localhost:PORT/input -d '{"text": "buy milk"}'` creates a todo. The CLI works both as a direct tool and as an HTTP client. Background embedding runs without manual `reindex`.


## Phase 6: Desktop UI (Tauri)

**Goal:** tray icon, global hotkey, capture bar, results display.

Tasks:

1. Set up Tauri project. Tray icon with quit/show actions.
2. Global hotkey opens a floating capture bar.
3. Capture bar sends input to daemon HTTP API, shows response.
4. Results view for search results and synthesized answers (with source references).
5. Todo list view.
6. Status indicator (daemon health, LLM status) — reads from `/health`.
7. First-run experience: check if daemon is running, prompt to start it. If no LLM model downloaded, show a setup prompt pointing to `memask model download`.

**Done when:** you can press a hotkey, type a thought or question, and see a response — all from the tray popup. The daemon handles everything behind the scenes.


## Phase 7: Polish and hardening

Tasks:

1. Startup automation: daemon starts on login, tray app starts on login.
2. Error handling pass: every failure path shows a useful message in the UI.
3. Settings UI: model selection (swap GGUF files), hotkey config, database location.
4. Export/import: dump all items as JSON or markdown.
5. Diagnostic panel: routing log, retrieval scores, job queue status.
6. Performance profiling: capture latency, search latency, embedding throughput, LLM generation time.


## Deferred (not in initial build)

- Multi-window UI
- File/URL ingestion pipeline
- Browser extension
- Advanced automation and agents
- Cross-platform packaging
- Event sourcing
- Encrypted storage
- Remote/cloud LLM backend option (API key based, for users who prefer it over local)
