# Personal Memory App — Plan of Action (revised)

Phases 0–3 are complete. This plan covers everything from Phase 3.5 onward.

## LLM stack (no external services)

All ML runs in-process with no external daemons:

- **Embeddings:** sentence-transformers (`all-MiniLM-L6-v2`), already integrated.
- **Intent classification:** cosine similarity against intent exemplars using the embedding model, already integrated.
- **Generative LLM:** `llama-cpp-python` loading a GGUF model file directly. Recommended starting point: Qwen2.5-3B-Instruct Q4_K_M (~2GB RAM, Metal-accelerated on macOS). Used for answer synthesis, summarization, and tag extraction.
- **Reranking:** cross-encoder model via sentence-transformers, loaded in-process.

No Ollama, no HTTP round-trips, no "is it running?" checks. If the model file exists and loads, it works for the lifetime of the process.


## Phase 3.5: Consolidation [DONE]

**Goal:** clean up the structural debt from Phases 0–3 so that Phase 4 (RAG) and Phase 5 (HTTP API) can build on solid foundations without mid-phase rewrites.

Tasks:

1. Introduce `ServiceContext` — a dataclass holding `conn`, `store`, `embedder`, and later `llm`, `reranker`, and `session`. This is the single object that flows through `dispatch()`, replacing the bare `sqlite3.Connection` parameter. The CLI builds one from Click context; the future HTTP daemon builds one at startup. Test that dispatch works with a `ServiceContext` carrying a `FakeEmbeddingService` and in-memory `VectorStore`.

2. Settle `DispatchResult` on one shape. The codebase has two versions (action/item/items vs success/intent_name/data). Pick the one that serializes cleanly to JSON for Phase 5 — `DispatchResult` with `action: str`, `data: dict`, `items: list[Item]` is the likely winner. Remove the other. Update CLI formatters and all dispatcher tests to match.

3. Upgrade `_handle_search` in the dispatcher to call `hybrid_search` through `ServiceContext` instead of `keyword_search` only. Wire `QueryContext` fields (`date_from`, `date_to`, `topic`, `category`) through to the search call — right now only `raw_query`, `type_filter`, and `status_filter` are used, meaning query understanding output is silently dropped.

4. Make `route()` accept a config/context parameter so classifiers (rules, embedding) and thresholds can be injected and tested independently. This also makes it possible for the RAG pipeline to receive the full `RoutingResult` including `QueryContext` without re-parsing the input.

5. Add an `AppContext` or `App` class that owns resource lifecycle: creates the `ServiceContext` components once, exposes a `shutdown()` that closes connections and releases models. The CLI instantiates it per invocation; the daemon instantiates it once at startup. Test startup and shutdown explicitly.

**Done when:** `dispatch(svc_ctx, "what did I note about deployment?")` calls `hybrid_search` with all query understanding filters, `DispatchResult` has one canonical shape, and resource lifecycle is owned by `AppContext`. All existing tests still pass with the new signatures.


## Phase 4: RAG pipeline [DONE]

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


## Phase 5: Daemon HTTP API [DONE]

**Goal:** all functionality is accessible over localhost HTTP.

### Decisions made

- **Sync strategy:** Flask (sync WSGI). SQLite is sync and single-user on localhost — async adds complexity with no benefit. FastAPI would require `run_in_executor` wrapping everywhere for zero gain.
- **SQLite threading:** `check_same_thread=False` on all connections via `get_connection()`. Safe because WAL mode handles locking and there's one user on localhost. The background worker creates its own connection inside its thread.
- **Background worker:** `threading.Thread` (daemon=True) with `time.sleep(5)` loop. Gets its own SQLite connection created inside the thread. Shares `VectorStore` and embedder from the main `AppContext` (these are thread-safe, unlike `sqlite3.Connection`).
- **Port:** 7394 on 127.0.0.1 (localhost only).
- **Dev server:** Flask's built-in server is fine for localhost single-user. Can swap to `waitress` in Phase 7 if needed.
- **Settings endpoint:** stubbed as `{}` — flesh out in Phase 7.

### Gotchas encountered and resolved

- **SQLite cross-thread error:** `sqlite3.ProgrammingError: SQLite objects created in a thread can only be used in that same thread`. Hit twice: (1) background worker reusing `svc.conn`, fixed by giving the worker its own connection; (2) Flask request threads using a connection created during startup, fixed by adding `check_same_thread=False` to `get_connection()`.
- **LanceDB table already exists:** `VectorStore._get_table()` used `list_tables()` to check existence, then `create_table()`, but newer LanceDB versions have a race/inconsistency where `list_tables()` misses the table but `create_table()` sees it. Fixed by trying `open_table()` first, falling back to `create_table()`.

### Files added

- `src/memask/server.py` — Flask app factory, endpoints: `/input`, `/items`, `/search`, `/health`, `/settings`
- `src/memask/daemon.py` — daemon entry point: startup checks, background worker, Flask server
- `src/memask/worker_thread.py` — background embedding worker thread
- `src/memask/client.py` — `DaemonClient` HTTP client (stdlib `urllib`, no new dependency)
- `src/tests/test_server.py` — HTTP endpoint integration tests
- `src/tests/test_worker_thread.py` — background worker tests
- `src/tests/test_client.py` — daemon client tests (spins up real Flask test server)
- `src/tests/test_cli_daemon.py` — CLI in `--url` daemon mode
- `src/tests/test_vector_store_reopen.py` — VectorStore reopening existing tables

### Files modified

- `src/memask/cli.py` — added `--url` option, daemon client path for `input`/`search`/`list`, `serve` command, `_format_daemon_result()`
- `src/memask/db/connection.py` — added `check_same_thread=False`
- `src/memask/search/vector_store.py` — `_get_table()` now tries `open_table()` first

### Dependencies added

- `flask>=3.0` in `pyproject.toml`

### Tasks completed

1. [DONE] Sync vs async decision: Flask (sync WSGI).
2. [DONE] HTTP server: `create_app()` factory, `/input` dispatches through `dispatch()`, `/items`, `/search`, `/health`, `/settings`.
3. [DONE] Health endpoint: LLM availability, embedder model name, job queue stats, vector index count.
4. [DONE] Startup sequence: migrations via `AppContext.service_context()`, `on_startup()` for stalled jobs / orphans / stale reindex.
5. [DONE] Background worker: daemon thread, 5s interval, own SQLite connection.
6. [DONE] Integration tests: `test_server.py` (20 tests), `test_worker_thread.py`, `test_client.py`, `test_cli_daemon.py`.
7. [DONE] CLI refactored: `--url` for daemon mode, direct mode fallback, `memask serve` command.

**Done when:** `curl localhost:PORT/input -d '{"text": "buy milk"}'` creates a todo. The CLI works both as a direct tool and as an HTTP client. Background embedding runs without manual `reindex`. ✅


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
7. Consider swapping Flask dev server for `waitress` if needed.
8. Flesh out `/settings` endpoint with actual settings management.


## Deferred (not in initial build)

- Multi-window UI
- File/URL ingestion pipeline
- Browser extension
- Advanced automation and agents
- Cross-platform packaging
- Event sourcing
- Encrypted storage
- Remote/cloud LLM backend option (API key based, for users who prefer it over local)
