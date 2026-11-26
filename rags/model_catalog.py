"""Small curated discovery catalog with no model, filesystem or network access.

The catalog describes supported contracts, not benchmark rankings or downloaded
artifacts. Operators may configure other verified models through embedding
profiles. Every accessor returns fresh values so callers cannot alter defaults.
"""
from copy import deepcopy


def _embedding(identifier, label, provider, model, *, languages, dimensions,
               window, execution, extra, cost, description, limitations, source,
               env=None, policy="exact", revision=None):
    values = {"RAGS_EMBEDDING_PROVIDER": provider, "RAGS_EMBEDDING_MODEL": model,
              "RAGS_EMBEDDING_INPUT_POLICY": policy}
    values.update(env or {})
    return {"id": identifier, "label": label, "provider": provider, "model": model,
            "languages": languages, "dimensions": dimensions,
            "max_input_tokens": window, "input_policy": policy,
            "execution": execution, "dependency_extra": extra,
            "cost_category": cost, "description": description,
            "limitations": limitations, "source_url": source,
            "revision": revision, "env": values}


_EMBEDDINGS = (
    _embedding(
        "offline-hash", "Offline hashing demo", "local", "offline-hash-v1",
        languages=["any"], dimensions=384, window=None, execution="local", extra=None,
        cost="local_compute", description="Deterministic setup and integration checks without downloads.",
        limitations=["Not a trained semantic model; do not infer retrieval quality from this demo."],
        source="https://github.com/MasterLKH180cm/RAGs", policy="auto"),
    _embedding(
        "bge-small-en", "BGE small · English", "fastembed", "BAAI/bge-small-en-v1.5",
        languages=["en"], dimensions=384, window=512, execution="local", extra="encoders",
        cost="local_compute", description="Compact English semantic retrieval through FastEmbed/ONNX.",
        limitations=["First use downloads weights unless cached; the input window includes special tokens.",
                     "Language suitability is a model-card claim, not a benchmark on your corpus."],
        source="https://huggingface.co/BAAI/bge-small-en-v1.5"),
    _embedding(
        "bge-small-zh", "BGE small · Chinese", "fastembed", "BAAI/bge-small-zh-v1.5",
        languages=["zh"], dimensions=512, window=512, execution="local", extra="encoders",
        cost="local_compute", description="Compact Chinese semantic retrieval through FastEmbed/ONNX.",
        limitations=["First use downloads weights unless cached; validate Traditional Chinese on your corpus.",
                     "Different dimensions and model identity require a new vector index."],
        source="https://huggingface.co/BAAI/bge-small-zh-v1.5"),
    _embedding(
        "fastembed-minilm", "MiniLM · FastEmbed", "fastembed", "sentence-transformers/all-MiniLM-L6-v2",
        languages=["en"], dimensions=384, window=256, execution="local", extra="encoders",
        cost="local_compute", description="Small English sentence encoder with a shorter input window.",
        limitations=["First use downloads weights unless cached; use short chunks.",
                     "A different runtime is not assumed to produce the same vector space."],
        source="https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2"),
    _embedding(
        "st-e5-small", "Multilingual E5 small", "sentence_transformers", "intfloat/multilingual-e5-small",
        languages=["multilingual", "en", "zh"], dimensions=384, window=512,
        execution="local", extra="sentence-transformers", cost="local_compute",
        description="Multilingual encoder with explicit query and passage instructions.",
        limitations=["PyTorch runtime and first-use weight download require more disk than the base install.",
                     "Query/passage prefixes are part of the input budget and immutable vector identity."],
        source="https://huggingface.co/intfloat/multilingual-e5-small",
        revision="614241f622f53c4eeff9890bdc4f31cfecc418b3",
        env={"RAGS_EMBEDDING_REVISION": "614241f622f53c4eeff9890bdc4f31cfecc418b3",
             "RAGS_EMBEDDING_QUERY_PREFIX": "query: ",
             "RAGS_EMBEDDING_DOCUMENT_PREFIX": "passage: ",
             "RAGS_EMBEDDING_NORMALIZE": "true"}),
    _embedding(
        "st-minilm", "MiniLM · Sentence Transformers", "sentence_transformers",
        "sentence-transformers/all-MiniLM-L6-v2", languages=["en"], dimensions=384, window=256,
        execution="local", extra="sentence-transformers", cost="local_compute",
        description="Pinned English encoder using the optional Sentence Transformers runtime.",
        limitations=["PyTorch runtime and first-use weight download require more disk than the base install.",
                     "Use short chunks; a matching model name across runtimes does not promise identical vectors."],
        source="https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2",
        revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        env={"RAGS_EMBEDDING_REVISION": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
             "RAGS_EMBEDDING_NORMALIZE": "true"}),
    _embedding(
        "ollama-nomic", "Nomic Embed Text · Ollama", "ollama", "nomic-embed-text",
        languages=["en"], dimensions=768, window=None, execution="local_service", extra=None,
        cost="local_compute", description="Native Ollama embeddings with server-side overflow rejection.",
        limitations=["Run Ollama and pull the model separately; a mutable model tag needs an operator space ID.",
                     "No exact local token counter is claimed; truncate:false rejects oversized input at the server."],
        source="https://ollama.com/library/nomic-embed-text", policy="server_reject",
        env={"RAGS_EMBEDDING_BASE_URL": "http://127.0.0.1:11434",
             "RAGS_EMBEDDING_AUTH_MODE": "none",
             "RAGS_EMBEDDING_QUERY_PREFIX": "search_query: ",
             "RAGS_EMBEDDING_DOCUMENT_PREFIX": "search_document: "}),
    _embedding(
        "openai-small", "OpenAI text-embedding-3-small", "openai", "text-embedding-3-small",
        languages=["multilingual", "en", "zh"], dimensions=1536, window=8191,
        execution="hosted", extra=None, cost="provider_billed",
        description="Hosted embeddings using the known OpenAI input-token contract.",
        limitations=["Requires operator-supplied credentials and sends text to the configured endpoint.",
                     "Provider charges apply; check current pricing and evaluate retrieval quality."],
        source="https://platform.openai.com/docs/guides/embeddings",
        env={"RAGS_EMBEDDING_BASE_URL": "https://api.openai.com/v1",
             "RAGS_EMBEDDING_AUTH_MODE": "required"}),
)

_TOKENIZERS = (
    {"id": "auto", "label": "Automatic source counter", "backend": "auto",
     "count_kind": "resolved_from_embedding", "unit": "provider_dependent",
     "dependency_extra": None, "requires_artifact": False,
     "description": "Choose the encoder counter where available; otherwise conservative UTF-8 bytes.",
     "limitations": ["Source chunking only; this is resolved when configuration is generated."]},
    {"id": "embedding", "label": "Configured embedding counter", "backend": "embedding",
     "count_kind": "exact_encoding", "unit": "model_tokens",
     "dependency_extra": None, "requires_artifact": False,
     "description": "Count complete document inputs with the selected embedding encoder.",
     "limitations": ["Needs a supported exact encoder contract and its runtime; source chunking only."]},
    {"id": "unicode", "label": "Unicode characters", "backend": "unicode",
     "count_kind": "fallback_units", "unit": "unicode_codepoints",
     "dependency_extra": None, "requires_artifact": False,
     "description": "Simple deterministic character budget with no extra dependency.",
     "limitations": ["Character counts are not model-token counts."]},
    {"id": "utf8_bytes", "label": "Conservative UTF-8 bytes", "backend": "utf8_bytes",
     "count_kind": "conservative_bytes", "unit": "utf8_bytes",
     "dependency_extra": None, "requires_artifact": False,
     "description": "Conservative raw-text budget without loading a model tokenizer.",
     "limitations": ["Not exact model tokens or chat framing; embedding validation remains separate."]},
    {"id": "cl100k_base", "label": "Tiktoken cl100k_base", "backend": "tiktoken",
     "count_kind": "exact_encoding", "unit": "model_tokens",
     "dependency_extra": None, "requires_artifact": False,
     "description": "Exact raw-text encoding counts for compatible OpenAI model families.",
     "limitations": ["First use may fetch encoding data; not an exact chat-message or tool envelope budget."]},
    {"id": "o200k_base", "label": "Tiktoken o200k_base", "backend": "tiktoken",
     "count_kind": "exact_encoding", "unit": "model_tokens",
     "dependency_extra": None, "requires_artifact": False,
     "description": "Exact raw-text encoding counts for compatible newer OpenAI model families.",
     "limitations": ["Select the encoding your model uses; chat framing is counted separately."]},
    {"id": "huggingface", "label": "Registered Hugging Face tokenizer.json", "backend": "huggingface",
     "count_kind": "exact_encoding", "unit": "model_tokens",
     "dependency_extra": "tokenizers", "requires_artifact": True,
     "description": "Use an operator-registered, SHA-256-verified local tokenizer artifact.",
     "limitations": ["Select huggingface:ALIAS; no automatic download or request-supplied file path.",
                     "Raw-text counting disables padding, truncation and special-token insertion."]},
    {"id": "sentencepiece", "label": "Registered SentencePiece model", "backend": "sentencepiece",
     "count_kind": "exact_encoding", "unit": "model_tokens",
     "dependency_extra": "tokenizers", "requires_artifact": True,
     "description": "Use a registered local SentencePiece model with a verified SHA-256 digest.",
     "limitations": ["Select sentencepiece:ALIAS; sampling is disabled and no model is downloaded.",
                     "Raw encoding counts do not include a generation provider's chat framing."]},
)


def embedding_catalog() -> list[dict]:
    """Public metadata only; no endpoints, credentials or environment assignments."""
    return [{key: deepcopy(value) for key, value in item.items() if key != "env"}
            for item in _EMBEDDINGS]


def tokenizer_catalog() -> list[dict]:
    return deepcopy(list(_TOKENIZERS))


def get_embedding_preset(identifier: str) -> dict:
    for item in _EMBEDDINGS:
        if item["id"] == identifier:
            return deepcopy(item)
    raise ValueError("Unknown embedding preset; run rags models list")


def get_tokenizer_preset(identifier: str) -> dict:
    for item in _TOKENIZERS:
        if item["id"] == identifier:
            return deepcopy(item)
    raise ValueError("Unknown tokenizer preset; run rags tokenizers list")
