"""Encoder-native complete-input counters shared by providers and research."""
import importlib.util
import os

from rags.registry import CapabilityError


def disable_encoder_telemetry() -> None:
    """Disable ONNX diagnostic traffic before constructing an inference session."""
    os.environ["ORT_DISABLE_TELEMETRY"] = "1"
    if importlib.util.find_spec("onnxruntime") is not None:
        import onnxruntime
        onnxruntime.disable_telemetry_events()


def fastembed_token_window(encoder):
    """Clone the exact native tokenizer without its truncation or batch padding.

    Restrict this adapter to the verified ordinary ONNX preprocessing family.
    Query/passage wrappers from other families need a separate explicit contract.
    """
    from fastembed.text.onnx_embedding import OnnxTextEmbedding
    from fastembed.text.text_embedding_base import TextEmbeddingBase
    from tokenizers import Tokenizer
    model = encoder.model
    tokenizer = getattr(model, "tokenizer", None)
    ordinary = type(model) is OnnxTextEmbedding
    if not ordinary:
        from fastembed.text.pooled_embedding import PooledEmbedding
        from fastembed.text.pooled_normalized_embedding import PooledNormalizedEmbedding
        ordinary = ((type(model) is PooledNormalizedEmbedding
                     and model.model_name == "sentence-transformers/all-MiniLM-L6-v2")
                    or (type(model) is PooledEmbedding
                        and model.model_name == "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"))
    if (not ordinary or tokenizer is None
            or getattr(model.query_embed, "__func__", None) is not TextEmbeddingBase.query_embed):
        raise CapabilityError("requires_provider", "This FastEmbed family needs an exact input tokenizer contract")
    limit = (tokenizer.truncation or {}).get("max_length")
    if type(limit) is not int or not 1 <= limit <= 100_000:
        raise CapabilityError("requires_provider", "Embedding model token window is unavailable")
    full = Tokenizer.from_str(tokenizer.to_str())
    full.no_truncation()
    full.no_padding()
    return lambda text, query=False: len(full.encode(text).ids), limit
