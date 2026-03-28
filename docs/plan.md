## Plan of Action

### Phase 0: Foundation

**Goal:** runnable daemon with storage, schema migrations, and a CLI for manual testing.

Tasks:
1. Set up project structure: `uv` for dependency management, `pytest` for testing, `src/` layout.
2. Define SQLite schema and migration system. Write the `item` table migration. Write tests for migration up/down.
3. Build the item repository layer — functions for create, read, update, soft-delete, list with filters. Test each operation.
4. Build the background job queue in SQLite — enqueue, dequeue, mark complete, mark failed, retry logic. Test crash recovery (enqueue, simulate crash, restart, verify retry).
5. Build a minimal CLI that can create an item, list items, and show job queue status. This is the test harness for everything that follows.

**Done when:** you can `create_item "buy milk" --type todo` and `list_items --type todo` from the command line, with full test coverage.

### Phase 1: Search (keyword)

**Goal:** items are searchable by text and filters without any ML.

Tasks:
1. Add SQLite FTS5 full-text index on item content and title. Write migration.
2. Build search function: keyword query + optional filters (type, category, date range, status). Return ranked results.
3. Add search to the CLI.
4. Write tests covering: exact match, partial match, filter combinations, empty results, special characters.

**Done when:** keyword search works reliably from the CLI. This is your non-LLM baseline.

### Phase 2: Embeddings and vector search

**Goal:** semantic search works alongside keyword search.

Tasks:
1. Integrate sentence-transformers. Pick a model (e.g., `all-MiniLM-L6-v2` for speed). Write an embedding service with a clear interface.
2. Set up LanceDB. Store embeddings with item ID and model version.
3. Build the chunking logic for long content — split with overlap, link chunks to parent items. Test boundary cases (short content, exactly-at-threshold, long content).
4. Build background embedding job: when an item is created, enqueue embedding. Worker picks it up, embeds, stores in LanceDB.
5. Build semantic search function: embed query, search LanceDB, return candidates with scores.
6. Build hybrid search: combine FTS5 results and LanceDB results. Merge, deduplicate, rank. Configurable weights.
7. Build SQLite-LanceDB reconciliation job: find orphaned vectors, clean up.
8. Add embedding model version tracking. Write reindexing job.

**Done when:** `search "that thing about deployment"` returns relevant results even if the note says "shipping to production." Tests cover: embedding, chunking, hybrid merge, reconciliation, model version mismatch detection.

### Phase 3: Intent router

**Goal:** free-text input is classified and dispatched correctly.

Tasks:
1. Define the intent taxonomy: capture, search, todo_create, todo_list, todo_complete, app_command.
2. Build rule-based router: prefix patterns (`/todo`, `?`, `!`), keyword triggers, heuristics. Test each intent with parametrized examples.
3. Build query understanding layer: extract date hints, topic filters, type filters from natural language queries. Shared between router and retrieval.
4. Add LLM-based classification (via Ollama) as a fallback when rules are ambiguous. Define "ambiguous" clearly — e.g., confidence below threshold.
5. Build graceful degradation: when Ollama is unavailable, router is 100% rule-based. Test this explicitly.
6. Wire router to existing handlers (create_item, search, todo operations).

**Done when:** `"remind me to buy milk"` creates a todo, `"what did I note about deployment?"` triggers search, `"/todo list"` lists todos — all from a single input endpoint. Tests cover: each intent, ambiguous inputs, Ollama-down fallback.

### Phase 4: RAG pipeline

**Goal:** questions get answers synthesized from stored notes.

Tasks:
1. Build context assembly: take top-N retrieval results, format as context for the LLM prompt.
2. Add a reranking step: use a cross-encoder model to score retrieval candidates against the query. Test that reranking improves relevance over raw vector similarity.
3. Build the answer generation prompt: system message, context, question. Include source references.
4. Build the answer endpoint: router dispatches question → hybrid retrieval → rerank → context assembly → LLM → response with sources.
5. Build non-LLM fallback: when Ollama is down, return ranked search results directly instead of a synthesized answer.
6. Add session context: hold last N exchanges in memory, include in prompt for follow-up questions.

**Done when:** asking `"what were my notes about the release plan?"` returns a synthesized answer citing specific notes. Tests cover: retrieval quality, prompt construction, fallback mode, session continuity.

### Phase 5: Daemon HTTP API

**Goal:** all functionality is accessible over localhost HTTP.

Tasks:
1. Build a lightweight HTTP server (FastAPI or similar). Single input endpoint + specific endpoints for listing, settings, status.
2. Wire all handlers through the HTTP layer.
3. Add health endpoint (reports: daemon up, Ollama available, job queue stats, index stats).
4. Add startup checks: run migrations, check Ollama, start background workers.
5. Write integration tests that hit the HTTP endpoints.

**Done when:** `curl localhost:PORT/input -d '{"text": "buy milk"}' ` creates a todo. The CLI becomes a thin HTTP client.

### Phase 6: Desktop UI (Tauri)

**Goal:** tray icon, global hotkey, capture bar, results display.

Tasks:
1. Set up Tauri project. Tray icon with quit/show actions.
2. Global hotkey opens a floating capture bar.
3. Capture bar sends input to daemon, shows response.
4. Results view for search results and answers.
5. Todo list view.
6. Status indicator (daemon health, Ollama status).
7. First-run experience: check if daemon is running, prompt to start it, check Ollama.

**Done when:** you can press a hotkey, type a thought or question, and see a response — all from the tray popup. The daemon handles everything behind the scenes.

### Phase 7: Polish and hardening

Tasks:
1. Startup automation: daemon starts on login, tray app starts on login.
2. Error handling pass: every failure path shows a useful message in the UI.
3. Settings UI: model selection, hotkey config, database location.
4. Export/import: dump all items as JSON or markdown.
5. Diagnostic panel: routing log, retrieval scores, job queue status.
6. Performance profiling: capture latency, search latency, embedding throughput.

### Deferred (not in initial build)

- Multi-window UI
- File/URL ingestion pipeline
- Browser extension
- Advanced automation and agents
- Cross-platform packaging
- Event sourcing
- Encrypted storage
