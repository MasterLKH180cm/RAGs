# Architecture and security boundaries

## Components

- FastAPI: authenticated tenant-scoped document, upload, query, delete APIs.
- SQLite WAL: transactional document store, FTS5 sparse index, embeddings and audit.
- Offline hashing vector: deterministic end-to-end baseline, not semantic embeddings.
- Optional OpenAI embeddings and OpenAI-compatible or Anthropic text generation.
- Rank fusion: BM25 + dense ranks combined with RRF, then a lexical relevance reranker.
- Query strategies: sparse, dense, hybrid, query-expansion fusion, optional HyDE,
  heading-context retrieval, adjacent-chunk hierarchy, explicit-edge graph-lite,
  simplified corrective expansion and heuristic adaptive routing.
- MCP stdio server: read-only search/list integration for Codex and Claude Code.

## Explicit non-goals in v0.1

This code does NOT yet implement Microsoft GraphRAG community detection and summaries,
RAPTOR learned recursive cluster summaries, HippoRAG, ColBERT, a cross-encoder
reranker, truly multimodal visual embedding, agent autonomous web search, or
the trained reflection-token mechanisms of Self-RAG. Those require independent
implementations and measurable quality gains. The agent workflow repository is
separate and calls RAGs over its authenticated API.

## Trust model

1. API keys are scoped to one tenant. Requests cannot choose tenant IDs.
2. All document and chunk reads/deletes are filtered by tenant.
3. External file URLs are not fetched. Uploaded file size and extension are bounded.
4. Retrieved text is untrusted. It never overrides the generation instruction.
5. SQL values are parameterized, including FTS match strings.
6. APIs accept only a bounded, defined set of retrieval modes.
7. Production requires configured strong API keys. Keep secrets outside Git.
8. For real multi-user tenants, put OIDC identity, document ACLs, quotas, rate
   limiting, audit export and TLS termination at a trusted gateway BEFORE deployment.
9. SQLite is a reliable low-scale baseline but is not horizontally shared across
   replicas. Move to PostgreSQL/pgvector or dedicated indexes when necessary.

## Evaluation

Compare strategies on an immutable test set with relevance-labelled document IDs.
Measure Recall@k, MRR, answer groundedness, refusal quality, p50/p95 response time,
ingest cost, index update semantics, and unauthorized cross-tenant retrieval.
No strategy should be declared superior without measuring the relevant dataset.

## Index migrations

Embedding model changes are NOT transparently compatible. Existing chunks store
their model identifier; queries only compare vectors from the current model.
Reingest source documents when switching embeddings. For large migrations,
run a dual-index migration and switch traffic only after verification.

## Operational rules

- External model failures surface as errors; never silently claim a valid answer.
- SQLite writes, delete and FTS updates occur in the same transaction.
- Keep the demo private to a laptop; use a secret manager and monitoring in production.
- Never log model credentials or full sensitive prompts in an external observability sink.
