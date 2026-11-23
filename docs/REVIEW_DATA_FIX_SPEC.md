# Review data-integrity and resource-bound corrections

This specification addresses CR-01, CR-02, CR-03, CR-06 and the federation
portion of CR-04 from the 2026-11-23 whole-codebase review. The execution plane
and model gateway remain separate services. No provider credentials or model
inference are required by these regression fixtures.

Status: implemented. The [review report](REVIEW_LITELLM.md) maps these changes
to regression evidence; [local results](testing-results-review-litellm.json)
record the tested source snapshot and selected PostgreSQL checks.

## Source validity and processing revisions

An index rebuild cannot change a source's valid-time interval. A later source
update closes every earlier overlapping processing revision for the same tenant
and document, not only the current index revision. When selecting evidence,
choose the newest processing revision for each document/valid-from interval
before applying its end time. This also prevents an older processing copy from
resurfacing in databases written before this correction. Metadata filters apply
only after revision selection. Start boundaries remain inclusive and end
boundaries exclusive; historical source reads and tenant isolation remain intact.

## Chunk construction

The ingestion planner has explicit limits of 1,024 source index records and
8 MiB of retained chunk text, context and distinct embedding strings, counted
as UTF-8 bytes. These are construction limits, not a final list-length check.
Token windows are generated incrementally. The planner validates a bounded
offset plan before retaining draft strings or calling a contextual generator.
Parent records consume the same record and byte budgets. Contextual output is
checked before constructing its combined embedding string. Semantic sentence
units retain their separate existing 2,048-unit limit. All six strategies retain
exact source offsets, tokenizer limits and explicit model dependencies.

## SQLite to PostgreSQL migration

The read-only source snapshot includes native `agent_runs`, `agent_messages`
and `agent_observations` when present. The importer preserves IDs, fingerprints,
checkpoint JSON, owner/session identity, message sequences, evidence references,
observation expiry and timestamps exactly; it does not approve or resume runs.
Older stores with no native agent schema remain importable. A partial native
schema, unknown populated table, or unsupported columns fails before target
schema initialization. All destination application tables participate in the
empty-target check, exclusive locks, copy transaction and count verification.
Import failure rolls back all rows; schema bootstrap may remain. Source files
are never modified. Operators must quiesce both applications as before.

## Embedding identity

The runtime provider's resolved gateway endpoint becomes the persisted router
embedding identity; the nominal settings marker `router` is not an embedding
space. Canonical endpoint identity removes trailing slashes and normalizes
scheme/host/default ports without changing path case. The actual configured
model and dimensions remain in the profile. Optional operator-controlled
`RAGS_EMBEDDING_SPACE_ID` identifies a versioned deployment/embedding space
behind a stable URL and model alias. Changing endpoint or explicit space ID
invalidates unchanged-ingest reuse and dense query compatibility until reindex.
Credentials are never part of this public profile. Existing router profiles
without a resolved endpoint require reindex. Direct/offline profiles retain
compatibility when no space ID is configured.

## Federation wall deadline

Each authorized remote request uses the shared cancellable HTTP transport with
a ten-second whole-call deadline spanning connection, headers and body. The
existing two-million-byte response cap, no redirects/compression policy and
per-tenant credentials remain. Any failed branch rejects the entire federated
result. Slow header/body fixtures must prove timely return and closed transports;
periodic bytes must not reset the total deadline.

## Acceptance

Regression coverage includes SQLite and actual PostgreSQL temporal boundaries,
multiple index-only rebuilds and pre-correction data; early count/byte rejection
before embeddings/context generation; native-agent migration equality and
rollback/nonempty-target refusal; same-model/same-dimension gateway changes;
explicit embedding-space changes; and actual loopback slow-header federation.
Existing chunk/tokenizer, migration, storage, federation and retrieval contracts
must continue to pass. Synthetic HTTP timings establish transport behavior only.
