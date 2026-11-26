# Collection-oriented retrieval contract (v0.10)

Status: implementation contract, committed before code. Baseline:
`936530bf9cdd1aa6380b2f5f168a51b55559d54d`.

## Goal and sources

Bring Chroma-inspired collection/record ergonomics, flexible search and dataset
forks to RAGs while retaining PostgreSQL/pgvector, authenticated tenants, optional
features and the existing evidence/answering pipeline. This is our API, not a
drop-in Chroma protocol implementation or a claim of Chroma Cloud scale.

Official references checked on 2026-11-26:

- https://www.trychroma.com/products/chromadb
- https://docs.trychroma.com/docs/collections/manage-collections
- https://docs.trychroma.com/cloud/search-api/overview
- https://docs.trychroma.com/docs/querying-collections/metadata-filtering
- https://docs.trychroma.com/docs/querying-collections/full-text-search
- https://docs.trychroma.com/cloud/features/collection-forking

Chroma currently documents the unified Search API and instant block-level
copy-on-write forks as Cloud features. Our forks share immutable record/vector
payloads but copy O(record-count) mapping rows. No constant-time fork, HNSW/SPANN,
object-storage tiering, hosted autoscaling, live connector sync, or performance
parity is claimed. Existing image retrieval and research engines retain their
separate capabilities and evidence requirements.

## Storage and consistency

Collections are tenant-scoped, with an opaque immutable ID, tenant-unique name,
metadata, immutable embedding configuration, integer generation and timestamps.
Optional parent ID/generation records fork lineage. Names are 1..128 ASCII
letters/digits/dot/underscore/dash, start with a letter/digit, and cannot be an IP.

Use separate `rag_collections`, `rag_collection_versions` and
`rag_collection_records` tables. Legacy documents/chunks and their queries do not
gain collection records implicitly. A record maps `(tenant, collection_id, id)`
to one immutable version containing nullable text, metadata, one dense vector,
optional sparse vector and timestamps. PostgreSQL uses a native pgvector column;
SQLite stores the same float32 values. Records are not automatically chunked.
Long-document chunking remains available through the existing document API.

All inputs in a batch are validated before any provider call. Provider preparation
precedes the write transaction. Mutation checks the observed collection generation
under a database lock, applies the entire batch, and advances generation once.
`add` conflicts on any existing ID; `update` fails on any missing ID; `upsert`
supports both. Duplicate request IDs are invalid. A preparation or SQL failure
leaves records, vectors, generation and audit unchanged. Optional
`expected_generation` is checked before providers and again at commit.

Omitted update fields are preserved. New records default to null document and
empty metadata, and must have a valid vector. Changed non-null text is embedded
again unless an explicit matching vector is supplied. Metadata-only updates and
unchanged text reuse vectors. A null document can clear text while retaining its
vector. Sparse null clears sparse data; omitted sparse data is preserved.

Forks atomically snapshot mappings and share payloads. Updates replace only their
own mappings. Delete removes only selected mappings; physical versions are purged
only after the last reference disappears. Shared-version mutation locks prevent
concurrent reference/cleanup races. Read/filter/ranking runs in one repeatable
snapshot, including native pgvector distance computation, so concurrent cleanup
cannot remove data midway through a search. No provider runs inside a write lock.

SQLite-to-PostgreSQL import includes the complete optional collection table group,
exact vectors, references, configuration and generations. It remains provider-free
and rejects partial/inconsistent schemas or a nonempty destination.

## Embedding and record contracts

Collection creation accepts `embedding=null` for the operator-configured provider,
or `{mode: external, model: ..., space_id: ..., dimensions: N}`. Configured mode
pins the effective model, endpoint and embedding-space identity; dimensions bind
on the first successful write. External mode always requires caller vectors.
Credentials and provider endpoints cannot be overridden by record/search bodies.
Precomputed vectors require `embedding_space_id` matching the collection binding.
Configured text-query/document encoders retain their distinct provider methods.

Dense vectors are 1..4096 finite, non-boolean float32 values, nonzero after rounding.
Normalize magnitude safely for cosine storage/search, preserving direction; reject
invalid/overflow inputs before mutation. Distance is cosine only in this version.
Sparse vectors have sorted unique integer `indices` (0..2147483647), positive finite
float32 `values` (<=1000000), at most 2048 entries, and an explicit `space_id` matching
the collection's optional immutable `sparse_space_id`. No learned SPLADE encoder is
implied by accepting precomputed sparse vectors.

Metadata permits scalar string/finite-number/bool/null and homogeneous scalar
arrays (no nested objects); bool is distinct from numbers. Bounds: 64 keys,
128-character keys, 4096-character strings, 100 array entries, 16 KiB encoded.
Text is at most 20000 characters and is never silently truncated. Configured
embedding token windows are checked before any batch encoding.

## HTTP application contract

All routes use existing authentication and explicit operation scopes. Tenant is
derived solely from the credential. Bodies use strict, side-effect-free Pydantic
models with unknown fields rejected. Default-on feature flags remain individually
disableable: `collections`, `collections.fork`, `collections.regex`,
`collections.sparse_vectors`, `collections.batch`, `collections.grouping`.
Existing ingestion/documents/query/deletion and strategy/provider flags also apply.

| Route | Operation |
| --- | --- |
| `POST /collections` | `{name, metadata?, embedding?, sparse_space_id?, get_or_create?}` |
| `GET /collections` | Paginated list, optional exact `name` lookup |
| `GET /collections/{id}` | Collection information |
| `PATCH /collections/{id}` | `{name?, metadata?, expected_generation?}` |
| `DELETE /collections/{id}` | Delete with optional expected-generation query |
| `GET /collections/{id}/count` | Committed record count |
| `GET /collections/{id}/indexing-status` | Transactional ready state/counts/profile/generation |
| `POST /collections/{id}/fork` | `{name, metadata?, expected_generation?}` |
| `POST /collections/{id}/records/{add,update,upsert}` | Parallel arrays `ids`, `documents?`, `metadatas?`, `embeddings?`, `sparse_embeddings?`; `embedding_space_id?`, `expected_generation?` |
| `POST /collections/{id}/records/get` | `{ids?, where?, where_document?, limit?, offset?, include?}` |
| `POST /collections/{id}/records/delete` | Explicit nonempty IDs or filters; no unqualified wipe |
| `POST /collections/{id}/search` | `{searches: [SearchQuery, ...]}` |
| `POST /collections/{id}/ask` | One text question plus supported collection search/filter options and `generate` |

Collection names are resolved to immutable IDs by SDK/CLI. Get-or-create returns
the existing collection without silently changing its configuration. Write/delete
responses include collection ID, generation and changed/deleted counts. Get returns
`{collection_id,generation,items,total,limit,offset}`. Search returns
`{collection_id,generation,results:[{items,mode,score_kind,matched_count,
candidate_count,has_more,limit,offset}],evidence_status:ranked_unassessed,usage}`.
Each item always has stable `id` and immutable `version`; selected optional fields
are document, metadata, embedding, sparse_embedding, score and distance.

## Unified bounded search

`SearchQuery` fields: `query_text?`, `query_embedding?`, `embedding_space_id?`,
`query_sparse?`, `mode` (dense/lexical/sparse/hybrid), `where?`, `where_document?`,
`limit=10`, `offset=0`, `candidate_limit=100`, `dense_weight=1`, `sparse_weight=1`,
`rrf_k=60`, `select`, `group_by?={field,max_per_group}`.

- Dense: configured query encoder or matching caller vector; native PostgreSQL
  cosine distance, equivalent float32 cosine reference on SQLite.
- Lexical: real bounded BM25 (`k1=1.2,b=0.75`, Unicode word tokens, casefolded)
  over the filtered collection snapshot, with no embedding/model call.
- Sparse: exact dot product of matching caller sparse vectors.
- Hybrid: weighted positive RRF, `weight/(rrf_k+one_based_rank)`, dense plus
  lexical BM25 or supplied sparse vectors. Zero-weight legs make no provider call.
  All-zero/negative/nonfinite weights are invalid. Text plus precomputed dense
  vector is allowed for hybrid lexical+dense search.
- Scores are descending with stable record-ID ties; cosine distance is separate.
  Existing `/query` rankings and score meanings remain unchanged.
- Grouping takes the first `max_per_group` per typed metadata key after ranking,
  before pagination; missing/null/bool/number groups are distinct. It is result
  diversification, not SQL numeric aggregation.
- The entire batch validates before providers, shares one data generation, and
  fails as a whole. Selection changes payload only, not authorization/ranking.
- Default upper bounds: 10000 records/collection, 100 records/write, 16 searches,
  100 ranked candidates, 100 returned records, offset+limit<=candidate_limit,
  16 MiB scanned materialized data and 2 MiB response. Bounds fail explicitly;
  no silently partial corpus. Operator configuration may lower these limits.

## Filters and bounded regex

Metadata: equality shorthand; `$eq/$ne/$gt/$gte/$lt/$lte/$in/$nin`, array
`$contains/$not_contains`; `$and/$or`. Missing does not equal null. Missing fields
do not match any field operator, including negative ones (explicit conservative
deviation from Chroma's missing-key `$nin` behavior). Numeric comparisons never
coerce booleans or strings. Document predicates use full nullable record text,
case-sensitive `$contains/$not_contains/$regex/$not_regex`, and `$and/$or`.
Null document never matches, including negations.

All trees/regex branches compile eagerly, depth<=8, nodes<=64, list<=100,
pattern<=256. Use the timeout-capable `regex` library with one shared 1-second
filter execution deadline per batch. Invalid expression or timeout produces a
fixed safe error and no partial result. Both backends use the same evaluator,
after tenant+collection SQL scoping and before ranking.

## Answering, SDK and CLI

Collection `ask` retrieves within one collection and reuses existing evidence
admission, complete-source budget packing, generation permissions and citation
format. Citations add collection/record IDs and retain immutable version-derived
chunk IDs with exact source text/offsets. No arbitrary client evidence is accepted.
Raw search is explicitly unassessed. No research/federated/as_of semantics are
silently inferred for the new collection ask contract.

`rags.client.HttpClient` and authenticated `PersistentClient` share models and the
same application service. Imports create no app/database/network/model side effect.
Local client uses an explicit scoped credential and existing roles; ambient remote
model configuration is loaded only through an explicit settings/from_env choice.
HTTP client disables redirects/proxies, validates endpoint/timeouts, bounds replies,
and returns sanitized typed errors. Collection handles retain immutable IDs.
Methods: create/get/get_or_create/list/delete collections; add/upsert/update/get/
count/delete/query/search/ask/fork/modify/indexing_status. CLI exposes collection,
records and search operations through that SDK with bounded JSON files and no
credential output. README contains executable offline and HTTP examples.

## Required verification

Unit/integration tests cover contract validation, every filter operator and bounds,
regex timeout, exact BM25/RRF/dot-product ordering, profiles/dimensions, zero-weight
provider avoidance, whole-batch validation and failures, authorization/features,
stable IDs/projection/grouping, reference cleanup and immutable fork isolation.

Actual SQLite and PostgreSQL/pgvector tests cover SQL rollback, concurrent CAS,
fork/delete races, same-snapshot rank under concurrent writes, no cross-tenant
access, restart persistence and provider-free SQLite-to-PG import. Run real HTTP
SDK/CLI lifecycle and installed-wheel/import smoke. Benchmark labelled synthetic
write/query/fork workloads with workload/hardware parameters and p50/p95, without
Chroma comparison claims. Preserve all frozen Qwen artifacts and existing research
tests. Keep complete-package coverage floors (80% lines,72% branches), CI gates and
failure evidence intact. Verify the exact PR head and merged-master CI.
