# Embedding and tokenizer selection

## Goal and baseline

Continue from master `50436763b63f895054b315f9732eb1979901f80b`.
The eighteen confirmed findings in `CODE_SMELL_REVIEW.md` are already merged;
retain their regression coverage and repair additional reproduced defects.
Expose practical local and hosted embedding choices and independently selected
chunking/generation tokenizers without changing existing deployment defaults.

## Contracts

1. Keep local hash, FastEmbed, OpenAI-compatible and approved gateway embeddings.
   Add optional Sentence Transformers with explicit model/revision, no remote
   Python execution, bounded cache ownership and correct query/document roles.
   Support explicit loopback-only unauthenticated embedding endpoints; never
   send ambient hosted credentials to them. Hosted provider families continue
   through the existing authorized gateway rather than new implicit routes.
2. Validate complete embedding inputs against the configured model window,
   including query/document prefixes. Reject oversize inputs instead of silently
   truncating. Validate returned vectors, requested dimensions and role handling.
   Bind index/collection identity to settings that change the vector space.
3. Add optional local Hugging Face tokenizer.json and SentencePiece model
   backends. Load explicit local artifacts without network or remote code,
   identify them by content hash, disable native truncation/padding, preserve
   original Unicode source spans and label token counts accurately. Existing
   unicode, UTF-8 byte and tiktoken behavior remains compatible.
4. Provide a static, offline model/tokenizer catalog with friendly IDs,
   provider/model names, language/use notes, installation requirements and
   explicit limitations. Offer both interactive selection and scriptable CLI
   commands producing a private, no-clobber environment overlay. Advanced
   model identifiers and tokenizer artifacts remain configurable.
5. Selecting or listing never downloads a model, calls a provider, writes a
   database or replaces an index. Model/config changes require restart and
   explicit reindex or a new collection; existing mismatches fail closed.
   Operator settings and authorization retain precedence. Generated selection
   files contain no copied credentials.
6. Keep feature permissions, CLI diagnostics, Compose forwarding, API catalog
   visibility and user documentation aligned. Keep optional dependencies out of
   ordinary offline use. Existing frozen model/quality evidence is unchanged.

## Acceptance

- Regression tests reproduce each new repaired failure and retain prior cases.
- Real tiny local tokenizer artifacts verify normalization, Unicode, special
  tokens, source spans, input bounds and fingerprint changes without downloads.
- Synthetic provider transports verify roles, dimensions, malformed vectors,
  credential isolation, bounded failure behavior and index incompatibility.
- CLI list/show/wizard/configuration flows work without credentials or model
  runtimes, reject invalid combinations and protect existing files.
- Actual available trained encoder smoke and SQLite/PostgreSQL integration
  exercise source ingestion, query, collections and model-switch rejection.
  Credential-dependent hosted inference is reported separately from fixtures.
- Run repository lint, unit/integration and coverage gates; preserve external
  resource gates and describe any execution limits precisely.
