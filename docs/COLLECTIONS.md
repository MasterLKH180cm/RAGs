# Collection SDK, search and dataset forks

RAGs provides collection-oriented vector records, an authenticated Python client,
an HTTP API and CLI commands. A collection has an immutable ID, a tenant-unique
name, metadata, a fixed embedding configuration and a generation counter. Each
record has a caller-supplied ID, nullable document text, metadata, one dense vector
and an optional sparse vector.

This API takes inspiration from Chroma's collection ergonomics. It uses RAGs'
own request and response contracts; installing `chromadb` or a Chroma server is
unnecessary. PostgreSQL/pgvector and SQLite remain the storage backends. The
[implementation contract](COLLECTIONS_SPEC.md) records the exact bounds and
consistency rules.

Collection records are **one vector per record**. They are not automatically
chunked or added to the existing document API. Use the document ingestion API for
long-document chunking and the existing research retrieval pipelines.

## Start locally

Install this repository's package as described in the [installation guide](../README.md#installable-cli-no-clone-needed).
The package provides both the `rags` command and `rags.client`; it is not currently
published on PyPI. In a source checkout, use `python -m pip install -e .` inside
your Python environment, or the repository's documented `uv sync` workflow.

Generate an explicit in-process credential for file-owner development:

```python
import os
import secrets
from rags.client import PersistentClient

local_key = secrets.token_urlsafe(32)
with PersistentClient("data/collections.db", api_key=local_key) as client:
    notes = client.get_or_create_collection("team-notes", metadata={"owner": "platform"})
    notes.upsert(
        ids=["retention", "recovery"],
        documents=[
            "The retention period is seven days.",
            "Recovery uses daily snapshots.",
        ],
        metadatas=[
            {"team": "platform", "tags": ["policy", "backup"]},
            {"team": "platform", "tags": ["runbook", "backup"]},
        ],
    )
    result = notes.search(query_texts=["retention period"], mode="hybrid", limit=5)
    print(result.rows())
    print(notes.count())
    print(notes.indexing_status())
```

The default local client explicitly creates a local-tenant, admin-role
configuration using the supplied key, deterministic local hash embeddings and
extractive answering. It does not read ambient authentication, remote model
credentials or model endpoints. This is a file-owner development mode: the key
mapping exists in the client configuration, and does not encrypt the database or
replace filesystem access control. Local hash embeddings demonstrate the data
flow; they are not a trained semantic encoder or a retrieval-quality benchmark.

To use an operator's existing credentials, tenant mappings, feature flags and
provider settings, pass an explicit `Settings` object:

```python
from rags.config import Settings

settings = Settings.from_env()
with PersistentClient(settings.db_path, os.environ["RAGS_API_KEY"], settings=settings) as client:
    print(client.list_collections_page())

# Equivalent explicit environment-loading factory:
with PersistentClient.from_env(os.environ["RAGS_API_KEY"]) as client:
    print([collection.name for collection in client.list_collections()])
```

Authentication happens before the database opens, and again before every local
operation. Tenant identity comes from the credential's operator configuration;
there is no caller-selected tenant argument. HTTP and local operations share the
same application service, scopes, feature gates and request validation. A local
client still has the privileges of its operating-system process, so use HTTP to
separate untrusted callers from storage and operator credentials.

Importing `rags.client` does not create an application, open a database, load a
model or make a network request. Both clients support `with` and `close()`.

## Connect through HTTP

For a new local HTTP deployment, bootstrap in an empty working directory:

```bash
rags init --profile offline
rags serve --env-file .env
```

In another terminal in that directory, the CLI can load the generated private
environment file directly:

```bash
rags collection create team-notes --env-file .env
rags collection list --env-file .env
```

For Python, supply the generated `RAGS_API_KEY` through the process environment
or your application's secret manager. Supply `RAGS_URL` for a nondefault server:

```python
import os
from rags.client import HttpClient

with HttpClient(
    os.environ.get("RAGS_URL", "http://127.0.0.1:8000"),
    api_key=os.environ["RAGS_API_KEY"],
    timeout=30.0,
) as client:
    notes = client.get_or_create_collection("team-notes")
    notes.upsert(ids=["retention"], documents=["The retention period is seven days."])
    print(notes.search(query_texts=["retention"], mode="lexical").rows())
```

The HTTP client sends Bearer authentication. HTTPS is required outside loopback
unless `allow_insecure_http=True` is explicitly supplied. Redirects, ambient HTTP
proxies and compressed responses are disabled. URLs cannot contain credentials,
queries or fragments. The network exchange has a total deadline, including body
streaming, and rejects responses exceeding 2 MiB. The timeout can be set between
zero and 360 seconds, excluding zero; the default is 30 seconds.

## Collection and record operations

The following snippets assume an open client and a `notes` collection handle.

```python
notes = client.get_collection("team-notes")
same = client.get_collection(collection_id=notes.id)
page = client.list_collections_page(limit=25, offset=0)
handles = client.list_collections(limit=25, offset=0)

info = notes.refresh().to_dict()
notes.modify(name="team-notes-v2", metadata={"owner": "security"},
             expected_generation=info["generation"])
```

`list_collections()` returns handles. `list_collections_page()` returns
`{items,total,limit,offset}`. Handles retain the immutable ID after a rename;
`refresh()` reloads their cached collection information. `get_or_create_collection`
returns an existing collection without silently changing its metadata or embedding
configuration. Record writes return their new generation; call `refresh()` when
you need current collection metadata or generation on the handle.

| Method | Behavior |
| --- | --- |
| `add(ids=..., ...)` | Every ID must be new; one conflict rejects the entire batch. |
| `update(ids=..., ...)` | Every ID must exist; one missing record rejects the entire batch. |
| `upsert(ids=..., ...)` | Add missing records and update existing records atomically. |
| `get(ids=..., where=..., where_document=..., include=...)` | Return a filtered, paginated record envelope without similarity ranking. |
| `delete(ids=..., where=..., where_document=...)` | Delete matching records; explicit IDs or a nonempty filter are required. |
| `count()` | Return the committed integer record count. |
| `client.delete_collection(...)` | Explicitly remove an entire collection. |

Writes use aligned arrays: `ids`, `documents`, `metadatas`, `embeddings` and
`sparse_embeddings`. Duplicate IDs, mismatched lengths and invalid values reject
the whole batch before embedding calls. New records need nonempty text that can
be encoded by the configured provider, or a valid explicit dense vector.

```python
generation = notes.refresh().to_dict()["generation"]
change = notes.update(ids=["retention"], metadatas=[{"team": "security"}],
                      expected_generation=generation)
row = notes.get(ids=["retention"], include=["document", "metadata", "embedding"])
notes.update(ids=["retention"], documents=[None],
             expected_generation=change["generation"])
```

Omitted update fields are preserved. Metadata replaces the complete metadata
object. A metadata-only update reuses the dense vector. Changed non-null text is
encoded again unless a matching explicit vector is supplied. A null document
clears text while retaining its vector. A sparse null clears that vector; omitted
sparse data is preserved. An unchanged write is a no-op and keeps the generation.

`expected_generation` enables compare-and-swap updates, modification, fork and
deletion. A stale value produces `collection_generation_conflict` with status
409. Refresh and reconcile the intended change before retrying; retrying a
write blindly can overwrite a newer caller's intent. Atomicity is per batch:
splitting an oversized request into multiple batches creates separate commits.

## Search, filtering and returned rows

Specify the mode explicitly when comparing retrieval behavior. The default is
`hybrid`.

| Mode | Inputs and ranking |
| --- | --- |
| `dense` | Text encoded by the bound provider, or a matching precomputed query vector; exact cosine ranking. |
| `lexical` | `query_text`; BM25 over the filtered snapshot, with no embedding or generation request. |
| `sparse` | An explicit, matching `query_sparse`; exact sparse dot product. |
| `hybrid` | Weighted reciprocal-rank fusion of dense and lexical BM25, or dense and explicit sparse results. |

```python
result = notes.search(
    query_texts=["retention policy", "snapshot recovery"],
    mode="hybrid",
    where={"$and": [
        {"team": {"$eq": "platform"}},
        {"tags": {"$contains": "backup"}},
    ]},
    where_document={"$not_contains": "deprecated"},
    select=["document", "metadata", "score", "distance"],
    group_by={"field": "team", "max_per_group": 2},
    limit=5,
)
first_query_rows = result.rows(0)
second_query_rows = result.rows(1)
print(result["generation"], result["results"][0]["score_kind"])
```

The response is a dictionary with `collection_id`, `generation`, `results`,
`evidence_status` and `usage`. `SearchResult.rows(index=0)` returns a copy of one
query's item list. Each item always includes stable `id` and immutable `version`.
`select` controls optional document, metadata, embedding, sparse embedding, score
and distance fields. Dense score, BM25 score, sparse score and RRF score have
different meanings; inspect `score_kind` before interpreting values. Scores sort
descending with stable record-ID ties. Cosine distance is supplied separately
when the dense leg ran; it is null for a purely lexical or sparse result.

Use `searches` for independent filters and ranking modes in the same batch:

```python
result = notes.search(searches=[
    {"mode": "lexical", "query_text": "retention", "select": ["document", "score"]},
    {"mode": "hybrid", "query_text": "recovery", "dense_weight": 0.0,
     "sparse_weight": 1.0, "where_document": {"$regex": "daily\\s+snapshots"}},
])
```

Do not combine `searches` with shortcut arrays or shared query options. A batch
uses one committed data generation and returns no partial results on failure.
The example's zero-weight dense leg makes no embedding request. For hybrid
ranking, `sparse_weight` controls either the BM25 leg or the explicit sparse leg.
Both weights cannot be zero. `query()` is an alias of this package's `search()`;
it does not translate Chroma's query protocol or its result-array format.

Metadata supports scalar strings, finite numbers, booleans, nulls and homogeneous
scalar arrays. Filters support equality shorthand, `$eq`, `$ne`, `$gt`, `$gte`,
`$lt`, `$lte`, `$in`, `$nin`, array `$contains`/`$not_contains`, and `$and`/`$or`.
Missing fields do not match any field operator, including negative operators;
missing and explicit null are distinct. Numbers are not coerced from booleans or
strings. Document predicates support case-sensitive `$contains`, `$not_contains`,
`$regex`, `$not_regex`, `$and` and `$or`; null documents match none of these.
Regular expressions are bounded and share a per-batch execution deadline.

Grouping keeps the first `max_per_group` ranked records per typed metadata value
before pagination. It diversifies results; it does not perform numeric SQL
aggregation. `limit`, `offset` and `candidate_limit` bound a ranked candidate pool,
with `offset + limit <= candidate_limit <= 100`. `has_more` describes additional
eligible rows in that bounded pool, not an unlimited collection cursor. Use
`get()` to enumerate records without ranked-search pagination.

## Bring your own embeddings

Create a collection with an immutable external embedding space:

```python
vectors = client.create_collection(
    "precomputed",
    embedding={"mode": "external", "model": "example-encoder-v1",
               "space_id": "example-dense-v1", "dimensions": 3},
    sparse_space_id="example-sparse-v1",
)
vectors.add(
    ids=["policy"], documents=["Retention lasts seven days."],
    embeddings=[[1.0, 0.0, 0.0]], embedding_space_id="example-dense-v1",
    sparse_embeddings=[{"indices": [12, 40], "values": [0.8, 0.3],
                        "space_id": "example-sparse-v1"}],
)
print(vectors.search(query_embeddings=[[1.0, 0.0, 0.0]], mode="dense",
                     embedding_space_id="example-dense-v1").rows())
print(vectors.search(mode="sparse", query_sparse={
    "indices": [12], "values": [0.8], "space_id": "example-sparse-v1",
}).rows())
```

Dense vectors must be finite, nonzero float32-compatible values. They are
normalized for cosine storage and comparison. Explicit vectors always need a
matching `embedding_space_id`; dimensions alone do not establish compatibility.
Sparse indices must be sorted and unique, with positive finite values and the
collection's `sparse_space_id`. Accepting sparse vectors does not imply a learned
SPLADE encoder is installed.

Configured collections pin the provider's effective model, endpoint and space
identity; dimensions bind on the first successful write. Their binding is visible
in `collection.refresh().to_dict()["configuration"]["embedding"]`. Provider
credentials and endpoints cannot be supplied in a record or search body. If the
operator changes the configured embedding profile, text encoding against an older
collection fails explicitly; matching precomputed vectors remain usable.

## Fork a dataset and inspect indexing state

```python
source = client.get_collection("team-notes")
generation = source.refresh().to_dict()["generation"]
branch = source.fork("team-notes-experiment", expected_generation=generation)
branch.update(ids=["retention"], documents=["The proposed retention period is thirty days."])
print(source.get(ids=["retention"])["items"])
print(branch.get(ids=["retention"])["items"])
print(branch.indexing_status())
```

Forking copies the collection's record mappings and shares immutable text,
metadata and vector payloads. Later writes replace only the target collection's
mappings. Deleting the source does not remove versions still referenced by the
fork. The fork starts at generation zero and records `fork_source_id` and
`fork_source_generation` for lineage.

Fork work is **O(record count)**. Chroma documents instant block-level copy-on-write
forking as a Cloud feature; this implementation does not claim those storage
internals, constant-time forks, HNSW/SPANN indexing, distributed autoscaling or
Chroma Cloud performance. See the
[official Chroma forking documentation](https://docs.trychroma.com/cloud/features/collection-forking).

Writes become queryable transactionally. `indexing_status()` reports `state=ready`,
`consistency=transactional`, committed counts and generation, `pending_operations=0`,
the embedding binding, whether the configured text encoder still matches, and
`float32_cosine_exact` or `pgvector_exact` indexing. These fields describe the
synchronous implementation; there is no fabricated background indexing queue.
The authenticated `/capabilities` response also contains a `collections` section
with enabled operations, feature flags and configured bounds.

## Ask questions with evidence

```python
answer = notes.ask("What is the retention period?", mode="lexical", top_k=5, generate=False)
print(answer["citations"])
```

`ask()` retrieves only within the selected collection and reuses RAGs' evidence
admission, complete-source context packing and citation pipeline. Citations carry
collection and record IDs plus immutable record-version references; changing a
record changes its citation identity. Raw `search()` is explicitly
`ranked_unassessed`, not an evidence-sufficiency decision. `generate=False`
retrieves and assesses evidence without an external answer-generation call.
With `generate=True`, the configured answer mode applies; external generation
additionally requires the generation scope and model feature permissions.

## CLI file inputs

Collection, record and search commands use the HTTP client. Credentials come from
`RAGS_API_KEY` or the existing `--env-file` loader, never a token command-line
argument. `RAGS_URL` defaults to `http://127.0.0.1:8000`; each leaf command also
accepts `--url`, `--timeout` and `--allow-insecure-http`. Success returns JSON on
stdout. Errors return a safe JSON code and status on stderr with exit status 1.

Create `records.json`:

```json
{
  "ids": ["retention", "recovery"],
  "documents": ["Retention lasts seven days.", "Recovery uses daily snapshots."],
  "metadatas": [{"team": "platform"}, {"team": "platform"}]
}
```

Create `query.txt` containing `retention policy`, then run:

```bash
rags collection create team-notes --env-file .env
rags records upsert team-notes --file records.json --env-file .env
rags records get team-notes --ids retention --env-file .env
rags search team-notes --query-file query.txt --mode lexical --env-file .env
rags collection count team-notes --env-file .env
rags collection status team-notes --env-file .env
rags collection fork team-notes --name team-notes-experiment --env-file .env
rags records delete team-notes-experiment --ids recovery --env-file .env
rags collection delete team-notes-experiment --env-file .env
```

`collection modify` accepts `--name`, `--metadata-file` and
`--expected-generation`. Creation accepts `--metadata-file`, `--embedding-file`,
`--sparse-space-id` and `--get-or-create`. Record `add`, `update` and `upsert`
accept the same JSON fields as their Python methods; include
`"expected_generation": N` in that file for conditional writes. Record `get` and
`delete` accept either `--ids` or a complete selection `--file`. Delete requires
an explicit selection.

`search --file searches.json` accepts `{"searches": [...]}` for the complete search
contract, including sparse vectors, grouping and batches. The text-file shortcut
also supports `--where-file`, `--where-document-file`, `--limit`, `--offset` and
`--select`. Do not mix a complete search file with shortcut options. Input files
must be bounded regular files; symlinks, duplicate JSON keys and nonfinite JSON
numbers are rejected.

## Bounds, feature gates and errors

| Bound | Default maximum |
| --- | --- |
| SDK/API encoded request, SDK/API response | 2 MiB each |
| Records per collection | 10,000 |
| Records per write/delete batch | 100 |
| Queries per search batch | 16 |
| Ranked candidates, returned records per query/page | 100 |
| Dense dimensions | 4,096 |
| Sparse entries | 2,048 |
| Document text, query text | 20,000 and 2,000 characters |
| Metadata per record/collection | 64 keys, 16 KiB encoded |
| Materialized scanned data | 16 MiB |
| Filter deadline | 1 second shared per batch |

Individual maxima are not a promise that all fields at maximum size fit in one
request. For example, 100 large UTF-8 documents can exceed 2 MiB. Reduce batch
size, page size or selected response fields. Oversized SDK requests fail before
transport or local provider execution. Service limits can be lowered by the
operator; `/capabilities` reports the active configuration.
Set `RAGS_COLLECTION_LIMITS_JSON` to a JSON object; Compose forwards the same
override. `max_response_bytes` has a minimum of 32768 bytes. Complete control
responses and accumulated write usage are validated before mutation commits,
including metadata and Unicode profile values.

The default-on `collections`, `collections.fork`, `collections.regex`,
`collections.sparse_vectors`, `collections.batch` and `collections.grouping`
flags can each be disabled. Existing ingestion, documents, query, deletion,
strategy and provider permissions still apply. A feature flag cannot grant a
credential an absent scope. The existing reader role can inspect and query;
it cannot create, modify, fork or delete collections. Editor and admin permissions
follow the existing role definitions.

```python
from rags.client import ClientError

try:
    notes.update(ids=["retention"], documents=["A revised policy"], expected_generation=0)
except ClientError as error:
    print(error.code, error.status_code)
```

Errors expose bounded codes and status numbers, without raw provider output,
server bodies or credentials. Common codes include `invalid_token` (401),
`insufficient_scope`/`feature_disabled` (403), `collection_not_found` (404),
`collection_conflict`/`collection_generation_conflict`/`embedding_space_mismatch`
(409), invalid input or filter and collection bounds (422), and network deadline
`request_timeout` (504). Unknown upstream errors are sanitized to `request_failed`.

## Verify the public paths

```bash
python -m pytest -q \
  tests/test_collection_client.py tests/test_collection_cli.py \
  tests/test_collection_client_integration.py tests/test_collection_http_e2e.py \
  -m "not postgres" --require-no-skips
```

These tests cover transport limits, import safety, credential handling, local
persistence, tenant isolation, scopes, fork independence, generation conflicts,
external vectors, filtering, citation identity, and complete SDK/CLI lifecycles
against a real loopback Uvicorn process, including server restarts. They use local
hash embeddings and require no model credentials. Storage, filter, ranking,
service/API, PostgreSQL and migration suites verify the underlying layers; the
project's full-suite and coverage gates remain required.

### Synthetic performance measurements

The benchmark uses precomputed synthetic vectors and real collection operations:

```bash
python scripts/benchmark_collections.py \
  --records 1000 --dimensions 128 --batch-size 50 --iterations 20 --warmup 2 \
  --report reports/collections-sqlite.json

# Supply a provisioned PostgreSQL/pgvector test database through the environment.
# RAGS_TEST_POSTGRES_URL is also accepted as a fallback.
python scripts/benchmark_collections.py --postgres \
  --records 1000 --dimensions 128 --batch-size 50 --iterations 20 --warmup 2 \
  --report reports/collections-postgres.json
```

The PostgreSQL DSN comes from `RAGS_BENCHMARK_POSTGRES_URL`; it is never printed
or included in the report. SQLite defaults to a temporary database;
`--sqlite PATH` explicitly chooses an existing or new file. Each invocation creates
two unique synthetic tenants and removes only their collection and audit rows,
including after a workload failure.

Reports include record/dimension/batch counts, hardware and Python information,
seed, warmup count, per-operation raw samples and interpolated p50/p95 latency.
Measured work includes batched ingestion, dense cosine, actual lexical BM25,
sparse dot product, both hybrid RRF combinations, batched search, record lookup,
metadata updates and collection forks. Correctness checks verify deterministic
rankings, counts, tenant isolation, shared immutable versions, mutation isolation
and fork survival after source deletion. Timings include the authenticated local
SDK, application service and database; model encoding, HTTP network transport,
setup and post-operation assertions are excluded. Search warmups precede sampled
searches. Initial ingestion samples count actual batches; other operations have
`--iterations` samples. A small sample count yields correspondingly limited p95
evidence. The workload is synthetic and establishes neither model quality nor
performance parity with Chroma or another database.
