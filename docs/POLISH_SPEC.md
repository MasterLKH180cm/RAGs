# Request-boundary and setup polish

Status: specified before implementation on 2026-11-21.
Baseline: `c015d300ad183117b22f7329bd697ccd93287fcd`.

## Goal and scope

Correct reproduced failures in ordinary ingestion/error handling and make the
two-service setup examples executable together. Keep retrieval algorithms,
chunking/tokenizer behavior for valid input, source spans, tenant boundaries,
frozen admission artifacts and quality thresholds unchanged. This iteration
does not add research engines or imply that a failed quality profile passed.

## P-RAG-01: serializable, private validation errors

The baseline returns HTTP 500 for some invalid JSON requests even though
Pydantic correctly rejects them: nonfinite numbers and unpaired Unicode
surrogates are echoed into FastAPI's validation response and fail serialization.

- Invalid request data must receive HTTP 422 with a `detail` list of `type`,
  `loc` and `msg` fields. Do not echo rejected input, raw bodies, exception
  contexts or credentials. Ordinary validation remains useful to clients.
- Error paths/messages must also serialize if a user-controlled path segment
  contains an invalid Unicode scalar. Valid Chinese, emoji and other Unicode
  text must retain its original content and offsets.
- Add a scoped request-validation handler; do not convert unexpected server
  errors or provider failures into client errors.
- Reject invalid Unicode in nested document/filter metadata before ingestion:
  the baseline otherwise persists it and subsequently fails document reads.
  Preserve existing metadata limits and valid Unicode exactly.
- Exercise malformed raw JSON values through the real ASGI application and
  temporary SQLite. Prove rejection before inference/storage mutation and
  preserve an existing document/revision. Include valid Unicode ingestion and
  retrieval and an ordinary field-validation error.

## P-RAG-02: consistent upload ingestion errors

The baseline turns a valid UTF-8 upload that exceeds the configured chunk cap
into HTTP 500, whereas equivalent JSON ingestion reports HTTP 400.

- Apply the existing ingestion ValueError-to-400 contract to the upload handoff.
  Preserve typed `CapabilityError` status/code, including provider HTTP 502.
- Preserve file size, media type, UTF-8, PDF and extracted-text bounds.
- Reproduce the cap with the actual configured chunker and a real multipart
  upload; prove the failed operation leaves existing sources/indexes intact.
  Keep successful file ingestion covered.

## P-RAG-03: coherent configuration and evidence

- Replace misleading production-key snippets with explicit generated-secret
  instructions and coherent tenant mappings shared with Agent. All generated
  credentials must satisfy existing validation and requester/approver
  credentials must remain distinct.
- Explain token-to-tenant versus tenant-to-token maps, relevant process
  configuration and the unchanged local development defaults. Do not include
  real credentials or imply the development Compose file is production-ready.
- Validate the documented configuration with actual settings/authentication
  code in a disposable environment. Preserve useful README flowcharts and
  accurately refresh test evidence and links.

## Acceptance and review

Run the existing strict unit/integration selections and complete-package
coverage floors (80% statements, 72% branches), lint, relevant ingestion
regressions and the real companion HTTP/worker/MCP smoke. Retain actual encoder,
research and admission CI gates. Update measured counts/source hashes and a
review record; do not claim unexecuted browser, external-service or quality
checks as passed. Review the final diff and verify the exact published revision.
