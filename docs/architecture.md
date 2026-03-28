# Personal Memory App — Corrected Architecture & Plan of Action

## Architecture (revised)

### Core flow

```
Tray UI → Local Daemon (HTTP) → Intent Router → Command Handler → SQLite → Background Enrichment → LanceDB → Query/RAG → Response
```

The app is split into two processes from day one:

- **Tray app** — thin UI shell (Tauri). Capture bar, results display, tray icon, global hotkey. Talks to the daemon over localhost HTTP.
- **Local daemon** — Python service. Owns all storage, routing, LLM interaction, indexing, and search. Runs independently. Testable without UI.

### Components

**1. Desktop shell (Tauri)**

System tray icon, global hotkey, popup capture bar, results display, notifications. Communicates with the daemon via REST on localhost. No business logic lives here.

**2. Intent router (hybrid, with full rule-based fallback)**

Classifies input as: note capture, question/search, todo command, instruction, app command.

Strategy: rule-based first (prefix patterns, keyword triggers), LLM classification only when rules are ambiguous. When Ollama is unavailable, the router operates entirely on rules — never blocks on LLM.

Router and RAG pipeline share a query-understanding layer. The router doesn't just classify "this is a question" and toss it over — it also extracts filters (date hints, topic, type) that the retrieval layer uses directly.

**3. Capture pipeline (write-first, enrich later)**

1. User enters text.
2. Daemon saves raw item to SQLite immediately, returns success.
3. Background worker picks up the item: classifies, tags, embeds, indexes into LanceDB.

Enrichment failures are recorded and retried. A failed embedding never blocks or loses the original note.

**4. SQLite (source of truth)**

Stores all items, todos, metadata, history, app state, settings, and the background job queue.

Schema uses explicit columns for known fields (type, content, created_at, updated_at, status, category, source). A `metadata` JSON column exists only for genuinely unstructured extension data — not as a dumping ground for fields that should be columns.

Schema migrations are managed via a `schema_versions` table and a sequential migration system (hand-rolled or alembic). Every schema change is a numbered migration. The app checks version on startup and migrates forward.

**5. LanceDB (vector index)**

Stores embeddings for items and chunks. Supports semantic similarity search with metadata filtering.

Sync with SQLite: every LanceDB record references a SQLite item ID. A periodic reconciliation job detects orphaned embeddings (item deleted from SQLite but vector still in LanceDB) and cleans them up. Deletes in SQLite mark items as soft-deleted; the reconciliation worker handles LanceDB cleanup.

**6. Embedding service (in-process sentence-transformers)**

Runs sentence-transformers directly in Python — no HTTP overhead, no Ollama dependency for embeddings. Ollama is used only for generative LLM tasks.

Embedding model version is stored alongside each vector in LanceDB. When the model changes, a reindexing job is queued. During transition, queries run against both old and new indexes, with new results preferred. Stale vectors are detected by comparing stored model version against current.

**7. Local LLM service (Ollama, optional)**

Used for: classification (when rules are ambiguous), tag extraction, answer generation, summarization.

Every LLM-powered feature has a non-LLM fallback:
- Classification → rule-based router
- Tag extraction → keyword extraction
- Answer generation → return search results directly
- Summarization → return full text or truncated snippets

The daemon checks Ollama availability on startup and periodically. Features degrade gracefully.

**8. RAG pipeline**

1. Receive question.
2. Router classifies as query, extracts filters.
3. Hybrid retrieval: vector similarity + keyword matching + metadata filters + configurable recency (off by default, enabled only for queries with temporal intent).
4. **Reranking step**: cross-encoder reranker scores candidates against the actual question. This is critical — raw vector similarity returns "close" results that aren't necessarily relevant.
5. Build context prompt from top-ranked results.
6. LLM generates answer with source references.
7. Return answer + sources to UI.

Without LLM: steps 5-6 are skipped, user gets ranked search results directly.

**9. Todo subsystem**

Stored as structured records in SQLite with explicit columns (title, status, priority, due_date, category). Todo text is also embedded and indexed in LanceDB for semantic search.

**10. Background job system**

Job queue persisted in SQLite. Each job has: id, type, status, payload, attempts, last_error, created_at, updated_at.

On crash recovery: daemon reads incomplete jobs from SQLite on startup and retries. Failed jobs are retried with exponential backoff up to a max attempt count, then marked as permanently failed with error details.

Handles: embedding, classification, tagging, chunking, indexing, reconciliation, reindexing.

**11. Chunking**

Long content is split into chunks with 10-20% overlap between consecutive chunks. Chunks reference their parent item ID. Stored in LanceDB with the parent relationship.

Short notes (under the chunk threshold) are embedded as-is.

**12. Session context**

A sliding window of recent interactions (last N exchanges) is held in memory per daemon lifetime. Follow-up questions use this context for continuity. The window is not persisted — it resets when the daemon restarts. This is intentional: it's conversational context, not memory.

**13. Data model**

Core entity: `item`

| Column | Type | Notes |
|---|---|---|
| id | TEXT (ULID) | Primary key |
| type | TEXT | note, todo, url, decision, guide |
| content | TEXT | Full original text |
| title | TEXT | Optional, extracted or provided |
| status | TEXT | For todos: pending, done, cancelled |
| priority | INTEGER | For todos |
| due_date | TEXT | ISO date, for todos |
| category | TEXT | User or auto-assigned |
| source | TEXT | manual, import, clipboard |
| tags | TEXT | Comma-separated or JSON array |
| created_at | TEXT | ISO timestamp |
| updated_at | TEXT | ISO timestamp |
| deleted_at | TEXT | Soft delete |
| metadata | TEXT | JSON, only for truly unstructured extras |

Separate tables: `chunks`, `jobs`, `schema_versions`, `settings`.

**14. Observability**

Structured logging with levels. In dev mode: full prompt traces, retrieval candidates with scores, routing decisions. In normal mode: events, errors, job failures. Logs are local-only.

A diagnostic CLI command dumps: last N routing decisions, last N retrieval results with scores, job queue status, sync status between SQLite and LanceDB.

---
