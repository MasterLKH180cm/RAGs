# Optional LiteLLM generation integration

RAGs uses its typed OpenAI-compatible router to call a separately deployed
LiteLLM Proxy. It does not add LiteLLM as an application dependency. The shared
deployment, restricted-key bootstrap, HTTPS/CA configuration, runbook and
acceptance procedure live in the companion repository:

<https://github.com/MasterLKH180cm/agentic-system/blob/master/docs/LITELLM.md>

For sibling source checkouts, read `../agentic-system/docs/LITELLM.md` and use
`../agentic-system/compose.litellm.yaml` with its full-stack base Compose file.
The gateway is optional; existing direct provider configurations remain valid.

## RAG profile

`rags init --profile litellm` creates a production-mode starter with generation
provider `router`, model alias `inkling`, conservative UTF-8 context accounting,
verified HTTPS, required service authentication and local baseline embeddings.
Router, native agent and memory retain their default-on policy. Configure the
gateway URL, private service-key file, optional CA file and both mappings
together before use:

```dotenv
RAGS_ROUTER_GATEWAY_URL=https://litellm:4000/v1
RAGS_ROUTER_API_KEY_FILE=/app/gateway/api.key
RAGS_ROUTER_CA_FILE=/app/gateway/ca.pem
RAGS_ROUTER_MODELS_JSON='{"inkling":"gateway/inkling-2-mini-iq4nl-v1"}'
RAGS_ROUTER_MODEL_OPTIONS_JSON='{"inkling":{"protocol":"openai"}}'
```

The unconfigured profile makes no model requests. A gateway service key never
replaces the application's tenant credential. Keep generation aliases immutable
when changing deployments, including for contextual/hierarchy derived indexes.
Only generation is routed through the supplied proxy profile; embeddings stay
on their own explicitly versioned space.

`RAGS_ROUTER_CA_FILE` snapshots a bounded nonsymlink PEM file and uses normal
certificate and hostname verification. Its content hash is included in native
agent approvals. A changed endpoint, alias or CA requires fresh approval.
Credential rotation for the same execution contract retains compatibility.

## Correctness upgrade

- Source updates close every processing copy of a superseded validity interval;
  reads also prevent older reindexed copies from becoming visible again.
- Chunk construction enforces 1,024 source-index records and 8 MiB retained
  UTF-8 materialization before expanding the full plan.
- SQLite-to-PostgreSQL migration includes native agent runs, complete history
  and observations. Unknown populated application tables fail explicitly.
- Router chat/embeddings and federation have cancellable whole-call deadlines.
- Router embedding profiles include the resolved gateway endpoint; optional
  `RAGS_EMBEDDING_SPACE_ID` records an immutable deployment version.
- Multipart ingestion runs in a bounded eight-slot pool; cancelled waits keep
  their slot until the synchronous work ends.
- A terminal native-agent checkpoint and its complete user/assistant turn
  commit atomically, with revision fencing against duplicate finalization.

Old router embedding profiles recorded the literal `router` endpoint and must
be reindexed after this upgrade. Offline/direct profiles with no explicit space
ID retain their existing identity. Review the detailed contracts in
[REVIEW_DATA_FIX_SPEC.md](REVIEW_DATA_FIX_SPEC.md) and
[REVIEW_RUNTIME_FIX_SPEC.md](REVIEW_RUNTIME_FIX_SPEC.md).

## Model performance evidence

The benchmark adds `--preset inkling-litellm`, `--ca-file` and
`compare --require-same-deployment`. The proxy preset requires a complete model
deployment manifest before reading credentials or making live requests.
Compare with the same direct deployment and keep cache, retry and fallback
disabled for the first measurement. The shared runbook contains exact commands.

Actual proxy tests use a deterministic local HTTP provider. They establish
protocol, authentication, persistence and timeout behavior. They do not measure
Inkling inference quality or throughput. The actual model still requires a host
with sufficient RAM/VRAM; the gateway itself does not load or accelerate weights.
