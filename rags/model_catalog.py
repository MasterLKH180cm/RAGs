"""Offline discovery of curated models; selecting a preset never loads weights."""
from dataclasses import dataclass
from importlib.util import find_spec


@dataclass(frozen=True)
class EmbeddingPreset:
    id: str
    label: str
    provider: str
    model: str
    languages: str
    dimensions: int | None
    description: str
    extra: str = ""
    source: str = ""
    query_prefix: str = ""
    document_prefix: str = ""

    def environment(self) -> dict[str, str]:
        values = {"RAGS_EMBEDDING_PROVIDER": self.provider, "RAGS_EMBEDDING_MODEL": self.model}
        if self.query_prefix:
            values["RAGS_EMBEDDING_QUERY_PREFIX"] = self.query_prefix
        if self.document_prefix:
            values["RAGS_EMBEDDING_DOCUMENT_PREFIX"] = self.document_prefix
        if self.provider == "openai":
            values["RAGS_EMBEDDING_BASE_URL"] = "https://api.openai.com/v1"
        if self.provider == "ollama":
            values.update(RAGS_EMBEDDING_BASE_URL="http://127.0.0.1:11434", RAGS_EMBEDDING_AUTH_MODE="none")
        return values

    def public(self) -> dict:
        module = {"encoders": "fastembed", "sentence-transformers": "sentence_transformers"}.get(self.extra)
        return {"id": self.id, "label": self.label, "provider": self.provider, "model": self.model,
                "languages": self.languages, "dimensions": self.dimensions,
                "description": self.description, "extra": self.extra,
                "runtime_installed": not module or find_spec(module) is not None,
                "source": self.source, "environment": self.environment(),
                "availability": "configuration_required" if self.provider in {"openai", "router", "ollama"}
                else "offline_demo" if self.provider == "local" else "weights_required"}


_HF = "https://huggingface.co/"
_FAST = "https://qdrant.github.io/fastembed/examples/Supported_Models/"


def _fast(id, model, languages, dimensions, description, query_prefix="", document_prefix=""):
    return EmbeddingPreset(id, model.split("/")[-1], "fastembed", model, languages, dimensions,
                           description, "encoders", _FAST, query_prefix, document_prefix)


def _st(id, model, languages, dimensions, description, query_prefix="", document_prefix=""):
    return EmbeddingPreset(id, model.split("/")[-1], "sentence_transformers", model, languages,
                           dimensions, description, "sentence-transformers", _HF + model,
                           query_prefix, document_prefix)


EMBEDDINGS = (
    EmbeddingPreset("offline", "Offline hash demo", "local", "offline-hash-v1", "demo", None,
                    "Deterministic testing without downloads; not a trained semantic model."),
    _fast("bge-small-en", "BAAI/bge-small-en-v1.5", "en", 384, "Compact English retrieval; CPU ONNX."),
    _fast("bge-base-en", "BAAI/bge-base-en-v1.5", "en", 768, "English retrieval; CPU ONNX."),
    _fast("bge-small-zh", "BAAI/bge-small-zh-v1.5", "zh", 512, "Compact Chinese retrieval; CPU ONNX."),
    _fast("multilingual-minilm", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
          "multilingual", 384, "Multilingual sentence similarity; evaluate on your retrieval data."),
    _fast("multilingual-e5-large", "intfloat/multilingual-e5-large", "multilingual", 1024,
          "Multilingual E5 retrieval with explicit query/document prefixes.", "query: ", "passage: "),
    _fast("jina-code", "jinaai/jina-embeddings-v2-base-code", "code,en", 768,
          "Code and natural-language retrieval using the ONNX runtime."),
    _fast("nomic-v1.5", "nomic-ai/nomic-embed-text-v1.5", "en", 768,
          "Nomic retrieval with explicit task prefixes.", "search_query: ", "search_document: "),
    _fast("qwen3-0.6b", "Qwen/Qwen3-Embedding-0.6B", "multilingual", 1024,
          "Qwen3 retrieval with an explicit query instruction; larger download than small BGE.",
          "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "),
    _st("st-minilm", "sentence-transformers/all-MiniLM-L6-v2", "en", 384,
        "Compact Sentence Transformers baseline."),
    _st("st-mpnet", "sentence-transformers/all-mpnet-base-v2", "en", 768,
        "MPNet sentence embeddings using Sentence Transformers."),
    _st("st-multilingual-e5-base", "intfloat/multilingual-e5-base", "multilingual", 768,
        "Multilingual retrieval; query and passage prefixes are supplied by this preset.", "query: ", "passage: "),
    _st("st-multilingual-e5-small", "intfloat/multilingual-e5-small", "multilingual", 384,
        "Smaller multilingual E5; explicit query and passage prefixes.", "query: ", "passage: "),
    _st("st-bge-m3", "BAAI/bge-m3", "multilingual", 1024,
        "BGE-M3 dense output only; sparse and multi-vector output are not selected here."),
    EmbeddingPreset("openai-small", "OpenAI embedding 3 small", "openai", "text-embedding-3-small",
                    "multilingual", 1536, "Hosted embeddings; API key required. Supports requested dimensions.",
                    source="https://platform.openai.com/docs/guides/embeddings"),
    EmbeddingPreset("openai-large", "OpenAI embedding 3 large", "openai", "text-embedding-3-large",
                    "multilingual", 3072, "Hosted embeddings; API key required. Supports requested dimensions.",
                    source="https://platform.openai.com/docs/guides/embeddings"),
    EmbeddingPreset("openai-ada", "OpenAI ada 002", "openai", "text-embedding-ada-002",
                    "multilingual", 1536, "Existing ada-002 deployments; fixed dimensions.",
                    source="https://platform.openai.com/docs/guides/embeddings"),
    *(EmbeddingPreset("ollama-" + id, label, "ollama", model, languages, dimensions,
                       "Local Ollama server; pull the model first. Oversize input is rejected by the server.",
                       source="https://docs.ollama.com/capabilities/embeddings",
                       query_prefix=query_prefix, document_prefix=document_prefix)
      for id, label, model, languages, dimensions, query_prefix, document_prefix in (
          ("nomic", "Ollama Nomic", "nomic-embed-text", "en", 768, "search_query: ", "search_document: "),
          ("mxbai", "Ollama mixedbread", "mxbai-embed-large", "en", 1024,
           "Represent this sentence for searching relevant passages: ", ""),
          ("bge-m3", "Ollama BGE-M3", "bge-m3", "multilingual", 1024, "", ""),
          ("embeddinggemma", "Ollama EmbeddingGemma", "embeddinggemma", "multilingual", 768,
           "task: search result | query: ", "title: none | text: "),
      )),
    EmbeddingPreset("gateway", "Approved LiteLLM gateway alias", "router", "embeddings", "model-dependent", None,
                    "Map the embeddings alias to an approved upstream. Configure its tokenizer/window and task roles.",
                    source="https://docs.litellm.ai/docs/embedding/supported_embedding"),
)

TOKENIZERS = (
    {"id": "unicode", "label": "Unicode code points", "backend": "unicode", "extra": "",
     "description": "Offline fallback units; these are not model tokens.", "config": {"backend": "unicode"}},
    {"id": "utf8-bytes", "label": "UTF-8 byte budget", "backend": "utf8_bytes", "extra": "",
     "description": "Conservative byte accounting, not an exact tokenizer for an arbitrary model.",
     "config": {"backend": "utf8_bytes"}},
    *({"id": name.replace("_", "-"), "label": "tiktoken " + name, "backend": "tiktoken", "extra": "",
       "description": "Exact text counts for this encoding; choose the encoding matching your model.",
       "config": {"backend": "tiktoken", "encoding": name}}
      for name in ("cl100k_base", "o200k_base", "p50k_base", "r50k_base")),
    {"id": "huggingface", "label": "Hugging Face tokenizer.json", "backend": "huggingface", "extra": "tokenizers",
     "description": "Explicit local tokenizer.json; no download or remote code. Requires --tokenizer-path.",
     "config": {"backend": "huggingface"}},
    {"id": "sentencepiece", "label": "SentencePiece model", "backend": "sentencepiece", "extra": "tokenizers",
     "description": "Explicit local .model artifact; no download. Requires --tokenizer-path.",
     "config": {"backend": "sentencepiece"}},
)


def embedding_preset(name: str) -> EmbeddingPreset:
    for preset in EMBEDDINGS:
        if preset.id == name:
            return preset
    raise ValueError("Unknown embedding preset; run 'rags models list' or use --provider and --model")


def tokenizer_preset(name: str, path: str | None = None) -> dict:
    for preset in TOKENIZERS:
        if preset["id"] == name:
            config = dict(preset["config"])
            if preset["backend"] in {"huggingface", "sentencepiece"}:
                if not path:
                    raise ValueError("This tokenizer requires an explicit local artifact path")
                config["path"] = path
            elif path is not None:
                raise ValueError("Only local artifact tokenizers accept a path")
            return config
    raise ValueError("Unknown tokenizer preset; run 'rags tokenizers list'")


def catalog(features=None) -> dict:
    models = [preset.public() for preset in EMBEDDINGS]
    tokenizers = [{**item, "config": dict(item["config"])} for item in TOKENIZERS]
    if features is not None:
        for item in models:
            item["enabled"] = features.enabled("embedding." + item["provider"])
        for item in tokenizers:
            item["enabled"] = features.enabled("tokenizer." + item["backend"])
    return {"schema": "model-catalog-v1", "embeddings": models, "tokenizers": tokenizers,
            "notes": ["Listing does not load or download models, test credentials or prove model availability.",
                      "Model IDs outside this curated catalog are supported through compatible configured providers.",
                      "Changing embedding identity requires reindexing documents or creating a new collection."]}
