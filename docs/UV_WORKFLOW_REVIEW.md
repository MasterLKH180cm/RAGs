# uv workflow implementation review

Reviewed on 2026-11-22 against [the specification](UV_WORKFLOW_SPEC.md).

## Changes

- Generated a portable 99-package `uv.lock` from the existing dependency ranges.
  The Python 3.12 development pin does not change the package's Python 3.11+
  requirement. Optional model dependencies are resolved but not installed by
  ordinary development/production commands.
- CI uses verified setup-uv v10.3.3 and uv 0.13.9. Lock/requirements-aware caches
  have separate suffixes for core, encoders, each research engine, runtime
  admission and the manual benchmark. Existing jobs and coverage floors remain.
- Docker copies the pinned uv binary, caches dependencies before copying source,
  installs the application non-editably, and retains the unprivileged runtime
  user. Startup invokes installed commands without dependency resolution.
- Both isolated installers use uv. Every existing research/model version pin and
  the LightRAG wheel hash remain. The legacy global CPU extra-index option was
  replaced with `--torch-backend cpu`: only PyTorch packages use that index,
  avoiding uv first-index conflicts with unrelated packages such as tqdm.
- README and MCP examples use the prepared uv environment, explain exact sync and
  optional extras, distinguish native SQLite from PostgreSQL, and show read-only
  pgvector confirmation. The deployment smoke command is labelled as writing
  test records and restarting the service. FAISS is accurately limited to the
  isolated RAPTOR upstream dependency set, not the native vector store.

## Local validation

| Check | Observed result |
| --- | --- |
| New project environment, selected core extras | 50 packages installed; no model weights requested |
| `uv lock --check` | Passed; lock resolves 99 packages |
| Unit contracts | 411 passed, 326 deselected, no skips |
| Ordinary integration contracts | 300 passed, 437 deselected, no skips |
| Existing Ruff selections and shell syntax | Passed |
| `uv pip check` | All installed packages compatible |
| Dependency layer with only pyproject + lock | Actual fresh-environment sync passed without source or README |
| Final noneditable source layer | Actual sync passed; isolated Python imported the installed package and selected dependencies |
| Actual API process launched through uv | Health, document ingestion and cited hybrid retrieval passed; process cleaned up |
| RAPTOR, LightRAG and answerability manifests | uv dependency-resolution dry runs passed; not a claim of complete local model/engine execution |

The local runner has a SOCKS proxy. The clean core environment did not include
httpx's optional SOCKS dependency, so three local integration requests initially
failed at client construction. Adding `httpx[socks]==0.28.1` only to this private
test environment supplied socksio; those three tests and then the complete
integration selection passed. No proxy configuration or product dependency was
changed. CI installs the exact selected lock on its normal runner. The new
Starlette version emits an httpx TestClient deprecation warning; tests pass.

Three repeat synchronizations of the already populated core environment took
0.05131, 0.01894 and 0.01603 seconds (median **0.01894 seconds**) on this runner.
This is an observed warm-sync baseline, not a controlled pip comparison or an
application-latency claim. Local raw evidence is generated under
`reports/testing/uv-install-validation.json`, `uv-http-smoke.json`, and the uv
inventory/JUnit files; these generated files are ignored by git.

## Remaining resource validation

Docker and PostgreSQL executables are unavailable locally. The unchanged actual
PostgreSQL/pgvector, Compose restart, encoder, research lifecycle and frozen-model
CI jobs are the verification route for those resources on this revision. Their
results must be checked before merging. The historical Qwen quality profile is
unchanged and remains unapproved; this installation migration does not promote it.
