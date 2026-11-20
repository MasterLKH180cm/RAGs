# RAGs v0.2 specification

Status: implementation contract prepared and committed before feature code, following the requested specification, implementation, and self-review sequence.
Baseline: v0.1 at 68d6d7a3bf59ed48475a11fbfe7f7645e6fcdb30.
Companion: agentic-system/docs/SPEC_V02.md.

## 1. Goal and completion rules

Implement the missing ingestion, retrieval, and evaluation mechanisms from the original twenty-family / six-chunking design. Preserve authenticated document/query/MCP integration. An API name, HTTP success, mocked provider response, or passing legacy test is not evidence that a research method has been reproduced.

Every capability reports its implementation, requirements, availability, and evidence level. Native mechanisms and upstream research engines have distinct identifiers. Missing models, indexes, checkpoints, credentials, or backends produce an explicit error; no advanced mode silently executes hybrid search. Offline hashing remains a clearly identified test/demo encoder.

The delivery ledger must distinguish: implemented and exercised; implemented with external integration unverified; unavailable dependency; not implemented. This contract does not silently remove unverified items from the original scope.

## 2. Text and index contracts

### RAG-01 Tokenization

Provide explicit unicode-codepoints and tiktoken backends. Tiktoken resolves a known model or explicit encoding; an unknown model fails rather than guessing. Unicode fallback is labelled as units, never exact model tokens. Splitting preserves exact source spans, advances monotonically, rejects impossible budgets, and rechecks counts at character boundaries. CJK, emoji, combining marks, special-token-looking strings and long delimiter-free input must work.

### RAG-02 Six chunking strategies

| ID | Mechanism and acceptance |
| --- | --- |
| fixed | Tokenizer-bounded windows with bounded overlap and exact source offsets. |
| recursive | Ordered paragraph/newline/sentence/word subdivision, with safe fixed fallback only for oversized units. |
| structure | Markdown heading paths and code/list/table block boundaries; no claim of PDF layout parsing. |
| parent_child | Persist real parent and child records; children stay inside parents; search leaves and expand parents without duplicating citations. |
| semantic | Sentence-vector discontinuities from an explicitly semantic encoder, plus hard size limits. Reject hashing/absent encoders. |
| contextual | Generate document-aware context for each bounded source chunk, store it separately, index it with BM25 and vectors, retain original source citations. |

Shared module interfaces: TokenizerConfig, resolve_tokenizer, Tokenizer.count/truncate/spans, ChunkConfig, chunk_document -> ChunkPlan. Chunk drafts carry local ID, role, parent ID, ordinal, raw text, start/end character offsets, heading path, tokenizer/count provenance, generated context, and embedding text.

### RAG-03 Budgeting

Pack the fully rendered question/instructions/evidence into a configured context budget after reserving output and envelope allowance. Include citation headers and delimiters. Report selected/dropped IDs, truncation and counting provenance. Never silently drop the question. External-provider exact input counts may be recorded when available; estimates remain labelled.

### RAG-04 Migration and lifecycle

Store a canonical index profile/fingerprint covering algorithm/config/tokenizer/embedding/context model identities. Changes invalidate the unchanged-document shortcut. Additive migrations mark old rows legacy without inventing metadata or calling models. Explicit reindex is tenant-scoped, validates source revision, performs provider work before write locks, and atomically swaps derived records. Failure preserves the previous index. Updates/deletes cover FTS, hierarchy, graph, vectors and archives. Deletion purges history. Incompatible embedding indexes produce a reindex-required error.

## 3. Twenty-family coverage contract

| Original family | Native mechanism / upstream boundary |
| --- | --- |
| Naive | A direct single-retriever evidence-to-answer path. |
| Sparse | FTS5 BM25, with ranking behavior explicitly reported. |
| Dense | Cosine retrieval with encoder identity and dimensions; hashes are labelled. |
| Hybrid | Independent sparse/dense ranking and RRF. |
| RAG-Fusion | Validate multiple generated queries, retrieve independently, fuse ranks; expose generated queries in trace. |
| HyDE | Generate hypothetical text, encode it, retrieve real documents; hypothetical text is never evidence. |
| Multi-vector / ColBERT | Native masked normalized sum-MaxSim over real token matrices; ColBERT requires an actual checkpoint/adapter and separate integration evidence. |
| Contextual | Generated chunk context in both sparse and dense indexes; reject indexes without the required profile. |
| Hierarchical / RAPTOR | Persist recursive clusters/summaries with original-leaf lineage; full RAPTOR is an explicitly identified upstream backend, not an alias. |
| GraphRAG | Native validated entities/relations, graph traversal/communities and source lineage; Microsoft GraphRAG is a separate upstream backend. |
| LightRAG | Explicit upstream backend contract; never alias lightweight edges to LightRAG. |
| HippoRAG | Explicit upstream backend contract; never claim a lexical graph boost is HippoRAG. |
| Self-RAG | Native bounded self-reflection is separately named; trained reflection-token Self-RAG requires its actual upstream model/backend. |
| CRAG | Native bounded evidence evaluation/refinement/retrieval; paper-specific CRAG backend is separately identified. |
| Adaptive | Select an available strategy with a validated router; record chosen strategy and reason. |
| Agentic | Bounded plan/search/evaluate/refine rounds with observable stopping reason and original citations. |
| Multimodal | Existing image-caption indexing stays labelled; native multimodal retrieval requires an actual aligned encoder/backend. |
| Structured | Typed row/metadata predicates and safe aggregations; no caller-supplied SQL. |
| Federated | Operator-allowlisted source IDs, tenant-specific credentials, bounded requests, RRF and source-qualified provenance. |
| Temporal / real-time | Document validity, versioned as-of reads and explicit revision updates; automatic streaming connectors remain separately reported unless implemented. |

Registry metadata must distinguish available, requires_provider, requires_index, and requires_backend, including the reason. Preserve legacy mode names where their documented mechanisms remain valid; changed model requirements are explicit compatibility changes.

Query extensions: filters (typed equality/source/row predicates), operator source IDs, as_of/validity bounds, iteration limits. Return requested/executed strategy, engine/profile, trace, evidence and usage. No request chooses an arbitrary URL or tenant.

## 4. Security and correctness invariants

- Authenticate and derive tenant before every lookup, including parent, graph, archive, memory/federation and upstream evidence mapping.
- Generated text is untrusted; structured outputs are validated and bounded.
- Citations resolve to authorized source records; generated summaries/hypotheses never impersonate source text.
- Provider failures do not destroy an existing index.
- Backend adapters are operator configured and read-only at query time. Caller-selected URLs, shell and raw SQL are prohibited.
- Secrets and source bodies are not recorded in telemetry.

## 5. Evaluation and acceptance

RAG-05: expose per-capability availability and prove distinct execution paths; unavailable dependencies fail explicitly.
RAG-06: exact structured-query and time-travel fixtures; version update/delete/reindex and cross-tenant regression coverage.
RAG-07: actual graph traversal/hierarchy lineage and numeric masked MaxSim fixtures; a real encoder/index run is required before claiming a verified ColBERT integration.
RAG-08: bounded model-loop tests prove control flow only; live model quality is a separate evidence class.
RAG-09: a labelled synthetic evaluation corpus with at least thirty questions covers lexical, semantic, multi-hop, structured, temporal and no-answer cases. Report applicable Recall@K, MRR, nDCG, answer EM/F1, citation validity, refusal behavior, p50/p95, usage and estimated cost. Unsupported metrics are null with reason, never invented zeros.
RAG-10: run real HTTP ingest/query/reindex/delete plus the independent Agent API/worker and MCP protocol. Tests use disposable stores and no private data.

No universal quality score is fabricated from the illustrative corpus. Record measured baselines and method-specific coverage. Paid providers and pretrained research backends require configured dependencies and actual execution before verification claims.

## 6. Implementation order and review

1. Commit this specification before feature code.
2. Implement token/chunk/profile modules, then storage and retrieval integrations.
3. Implement native mechanisms and explicit external-backend contracts.
4. Update evaluation, HTTP/MCP examples, Docker dependencies and CI.
5. Self-review against requirement IDs; fix findings; rerun only affected checks and required integration gates.
6. Record evidence and remaining external requirements in docs/IMPLEMENTATION_REVIEW.md and README.

## References

- https://github.com/openai/tiktoken
- https://www.anthropic.com/research/contextual-retrieval
- https://github.com/stanford-futuredata/ColBERT
- https://github.com/parthsarthi03/raptor
- https://microsoft.github.io/graphrag/
- https://github.com/HKUDS/LightRAG
- https://github.com/OSU-NLP-Group/HippoRAG
- https://github.com/AkariAsai/self-rag
- https://github.com/HuskyInSalt/CRAG
