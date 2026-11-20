# RAGs v0.2 architecture

This document describes the implemented service. The [specification](SPEC_V02.md) records the original requirements; the [implementation review](IMPLEMENTATION_REVIEW.md) separates exercised mechanisms from unverified external integrations. The [README](../README.md#architecture-and-flows) contains the service, ingestion, and retrieval flowcharts.

## Service boundaries

FastAPI authenticates a bearer credential and derives one tenant before calling the document service or original-image service. The read-only stdio MCP bridge calls that HTTP API. The companion [agentic-system](https://github.com/MasterLKH180cm/agentic-system) has its own API, durable workflow store, and worker; its PostgreSQL support does not change RAGs' SQLite storage.

Document ingestion and retrieval live in `RAGService`. `Providers` supplies hashing or trained single-vector embeddings and optional text generation. `BackendManager` supplies actual token-matrix late interaction or an explicitly configured external research protocol. Original image assets have a separate service and paired CLIP encoder; they do not masquerade as text chunks.

## Source records and derived indexes

| Storage | Responsibility |
| --- | --- |
| `documents` | Current document identity, source content, and index profile |
| `document_revisions` | Source snapshots, recording time, validity interval, and processing profile |
| `chunks` | Original leaf/parent spans, generated summaries, vectors, tokenizer metadata, and lineage |
| `chunks_fts` | FTS5 source/context sparse index |
| `graph_relations` | Entity relations with original source-chunk evidence IDs |
| `audit` | Lifecycle events in the same local database |
| `image_assets` | Tenant-scoped original image bytes, vectors, revisions, and encoder identity |

SQLite uses WAL, foreign keys, and bounded busy waits. Document changes and their FTS, hierarchy, graph, and audit changes commit together. The audit table is an operational record, not a separate tamper-evident audit system.

### Atomic ingestion and reindexing

Ingestion validates the source and processing configuration, resolves the tokenizer, and constructs exact source spans. The canonical profile includes algorithm/configuration, tokenizer identity, embedding model/endpoint/dimensions, active semantic/context identities, and prompt revision. The unchanged shortcut requires matching source and profile, so a tokenizer or model change cannot silently retain an incompatible index.

Provider calls and derived-index construction happen before the write transaction. A compare-and-swap check verifies that the expected source revision is still current, then commits the source snapshot and derived records atomically. Failed inference or a concurrent change preserves the previous index. Explicit reindexing requires the expected revision. Deleting a document purges its stored revisions and derived evidence as well as its current record.

Additive schema migration marks existing unqualified records as `legacy`; it does not invent offsets/token counts or call a model. Vector retrieval rejects an incompatible or unknown profile and requests reindexing. A source reprocessing operation preserves validity when the source has not changed; a source update creates a new revision.

### Version selection and evidence lineage

Authorization and version selection precede ranking. For each document, current or `as_of` reads select the latest recorded revision valid at the requested instant, then apply metadata filters. Validity intervals include their start and exclude their end. A filter therefore cannot resurrect an older version. This supports valid-time reads and explicit updates; it does not provide automatic upstream refresh, change-data-capture connectors, or a separate transaction-time query API.

Leaf and parent records retain exact source offsets. Generated contextual prefixes remain separate from source text. Hierarchy summaries store original-leaf lineage and expand into original leaves for citations. Graph traversal uses relations whose evidence IDs are already in the authorized revision scope. Hypothetical HyDE documents and generated summaries are never accepted as original source citations.

## Text processing and prompt budgets

The six ingestion strategies are fixed, recursive, structure, parent-child, semantic, and contextual. Semantic boundaries require a trained encoder; contextual prefixes require a generator. Missing required dependencies produce an error. Structure parsing recognizes text/Markdown blocks and heading paths; it is not a PDF layout parser.

Unicode code-point counting is an explicit baseline. Tiktoken resolves a known model or explicit encoding and preserves character boundaries when producing source spans. Stored counts identify their tokenizer and counting unit. Source chunking and generation can use different tokenizers; selecting a correct model encoding remains an operator responsibility for custom compatible endpoints.

Before generation, the service packs the rendered instructions, question, citation headers, separators, and evidence within the configured context budget after reserving output and envelope allowances. The response records included/dropped/truncated sources and counting provenance. An oversized fixed prompt fails rather than dropping the question. Unicode counts and reserved framing estimates are not presented as exact provider token accounting.

## Retrieval execution

| Mechanism | Execution and dependency boundary |
| --- | --- |
| Sparse, dense, naive, hybrid | FTS5 ranking, cosine search, direct evidence assembly, and independent rank fusion |
| Parent-child, hierarchy | Persisted parent expansion or recursive vector clusters/summaries with original-leaf lineage |
| Graph | Explicit or validated model-extracted relations; two-hop traversal and label-propagation communities |
| Structured | Typed row/metadata predicates and bounded numeric aggregations; no submitted SQL |
| Temporal | Authorized version selection and recency-aware ranking |
| Fusion, HyDE, contextual | Distinct query-generation, hypothetical-document, or contextual-index mechanisms; required generation/index dependencies are explicit |
| Corrective, adaptive, agentic, self-reflective | Validated routing/planning/evidence review/draft critique, with at most three refinement rounds |
| Multi-vector / ColBERT | Real pretrained query/document token matrices and normalized masked sum-MaxSim |
| Image assets | Paired CLIP text/image vectors, cosine ranking, and original-image citations through `/assets/*` |
| Federated | Operator-defined source IDs and per-tenant credentials, bounded parallel reads, source-qualified IDs, and RRF |

The native hierarchy is not full RAPTOR; the native graph is not Microsoft GraphRAG, LightRAG, or HippoRAG; the reflection loop is not trained reflection-token Self-RAG. Separate identifiers reserve those named methods for configured external engines. Their adapter validates protocol and evidence boundaries, but does not install or establish the correctness of the upstream research implementation.

Model-dependent paths validate structured outputs, permitted choices, and source IDs. Bounded loops record decisions and stopping reasons. There is no arbitrary shell execution or general-purpose tool execution in this RAG service. The companion workflow runtime owns its own tool and approval policies.

## External boundaries and authorization

The caller cannot select a tenant, backend URL, or backend credential. Federation and research endpoints come from operator configuration; each tenant has a distinct mapped credential. Requests carry only already authorized source passages or source-scoped query parameters. Research responses can select IDs only from the supplied original leaf records. Federation IDs are source-qualified to prevent collisions.

HTTP adapters reject redirects and compressed responses, enforce byte limits before buffering, and check total deadlines after transport chunks. Numeric scores, aggregation values, usage fields, and token matrices reject nonfinite values and inappropriate types. Invalid Unicode bearer credentials return unauthorized responses. Provider failures surface as explicit errors rather than being relabelled as successful offline execution.

All document, archive, parent, graph, and original-asset operations use tenant scope. This service's static key map is not OIDC, document-level ACL management, or a complete public API gateway. External services still need appropriate source handling, retention, and credential isolation; local protocol validation cannot prove their internal behavior.

## Evaluation and operational limits

The [measured 36-question synthetic report](evaluation-v02-summary.json) records applicable retrieval, structured/time, answer, citation-record, refusal, and latency metrics. Offline hybrid Recall@3 was `0.85`; its no-answer refusal recall was `0.0`. Record-valid citations establish that returned evidence exists at the recorded source span, not that it entails every generated claim. Paid-model behavior and named upstream research services remain unverified by live integration.

Actual BGE, ColBERT, and paired CLIP checkpoints were exercised separately. These small fixtures verify encoder/adapter execution and selected invariants; they are not domain-quality benchmarks. Ordinary tests also distinguish deterministic control-flow fixtures from real-model inference.

SQLite, full vector scans, on-demand late-interaction encoding, and local federation threads target small deployments. The service does not currently implement a distributed vector index, automatic streaming refresh, production identity gateway, quotas, or semantic answer-verification guarantees. Scaling and public deployment require those concerns to be addressed explicitly; they are not implied by passing local integration tests.
