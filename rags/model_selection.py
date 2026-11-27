"""Credential-free model selection overlays for operators and the installed CLI."""
import argparse
import json
import os
from pathlib import Path
import shlex
import sys

from rags.model_catalog import EMBEDDINGS, TOKENIZERS, catalog, embedding_preset, tokenizer_preset
from rags.tokenization import TokenizerConfig


PROVIDERS = ("local", "fastembed", "sentence_transformers", "openai", "ollama", "router")
MODEL_ENV_KEYS = frozenset({
    "RAGS_EMBEDDING_PROVIDER", "RAGS_EMBEDDING_MODEL", "RAGS_EMBEDDING_BASE_URL", "RAGS_EMBEDDING_AUTH_MODE",
    "RAGS_EMBEDDING_REVISION", "RAGS_EMBEDDING_DIMENSIONS", "RAGS_EMBEDDING_QUERY_PREFIX",
    "RAGS_EMBEDDING_DOCUMENT_PREFIX", "RAGS_EMBEDDING_QUERY_INPUT_TYPE", "RAGS_EMBEDDING_DOCUMENT_INPUT_TYPE",
    "RAGS_EMBEDDING_TOKENIZER_JSON", "RAGS_EMBEDDING_MAX_TOKENS", "RAGS_EMBEDDING_SPACE_ID",
    "RAGS_TOKENIZER_JSON", "RAGS_GENERATION_TOKENIZER_JSON",
})


def add_commands(sub) -> None:
    models = sub.add_parser("models", help="Browse embeddings without loading models or contacting providers")
    actions = models.add_subparsers(dest="model_action", required=True)
    listing = actions.add_parser("list", help="List curated embedding choices")
    listing.add_argument("--provider", choices=PROVIDERS)
    listing.add_argument("--language", help="Filter en, zh, multilingual, code or demo")
    listing.add_argument("--json", action="store_true")
    show = actions.add_parser("show", help="Show a model preset and its setup requirements")
    show.add_argument("preset")
    show.add_argument("--json", action="store_true")
    tokenizer = sub.add_parser("tokenizers", help="Browse supported text counters")
    token_actions = tokenizer.add_subparsers(dest="tokenizer_action", required=True)
    token_list = token_actions.add_parser("list")
    token_list.add_argument("--json", action="store_true")
    select = sub.add_parser("configure", help="Choose models interactively or write a model selection overlay")
    mode = select.add_mutually_exclusive_group()
    mode.add_argument("--embedding", help="Preset ID from 'rags models list'")
    mode.add_argument("--provider", choices=PROVIDERS, help="Advanced: configure a provider directly")
    select.add_argument("--model", help="Advanced provider model ID or approved router alias")
    for name in ("tokenizer", "generation-tokenizer", "embedding-tokenizer"):
        select.add_argument("--" + name, choices=tuple(item["id"] for item in TOKENIZERS))
        select.add_argument("--" + name + "-path", type=Path, help="Local tokenizer artifact")
    select.add_argument("--embedding-max-tokens", type=int)
    select.add_argument("--endpoint")
    select.add_argument("--dimensions", type=int)
    select.add_argument("--revision")
    select.add_argument("--query-prefix")
    select.add_argument("--document-prefix")
    select.add_argument("--query-input-type")
    select.add_argument("--document-input-type")
    select.add_argument("--space-id")
    select.add_argument("--auth-mode", choices=("required", "none"))
    select.add_argument("--output", type=Path, default=Path("rags.models.env"))


def _pick(items, title: str, default: str) -> str:
    print(title)
    for number, item in enumerate(items, 1):
        print(f"  {number:2d}. {item['id']:<28} {item.get('description', '')}")
    answer = input("Choose a number or ID [" + default + "]: ").strip() or default
    if answer.isdecimal() and 1 <= int(answer) <= len(items):
        return items[int(answer) - 1]["id"]
    if answer not in {item["id"] for item in items}:
        raise ValueError("Select a listed number or ID")
    return answer


def selection_environment(args) -> dict[str, str]:
    if args.embedding:
        if args.model is not None:
            raise ValueError("Use --provider with --model for a custom model; presets already select a model")
        values = embedding_preset(args.embedding).environment()
    else:
        if not args.provider or not args.model:
            raise ValueError("Choose --embedding, or supply both --provider and --model")
        values = {"RAGS_EMBEDDING_PROVIDER": args.provider, "RAGS_EMBEDDING_MODEL": args.model}
    # Reset optional vector-space settings when switching overlays. Secrets and
    # feature policy are deliberately absent and are never copied or overridden.
    values = {**{key: "" for key in MODEL_ENV_KEYS if key.startswith("RAGS_EMBEDDING_")},
              "RAGS_EMBEDDING_AUTH_MODE": "required", **values}
    fields = {"endpoint": "BASE_URL", "dimensions": "DIMENSIONS", "revision": "REVISION",
              "query_prefix": "QUERY_PREFIX", "document_prefix": "DOCUMENT_PREFIX",
              "query_input_type": "QUERY_INPUT_TYPE", "document_input_type": "DOCUMENT_INPUT_TYPE",
              "space_id": "SPACE_ID", "auth_mode": "AUTH_MODE"}
    for field, suffix in fields.items():
        value = getattr(args, field)
        if value is not None:
            values["RAGS_EMBEDDING_" + suffix] = str(value)
    prefix_keys = {"RAGS_EMBEDDING_QUERY_PREFIX", "RAGS_EMBEDDING_DOCUMENT_PREFIX"}
    if any("\0" in value or "\r" in value or ("\n" in value and key not in prefix_keys)
           for key, value in values.items()):
        raise ValueError("Only embedding prefixes may contain newlines; configuration cannot contain NUL or CR")
    if not 1 <= len(values["RAGS_EMBEDDING_MODEL"]) <= 160:
        raise ValueError("Model ID must contain 1..160 characters")
    if args.dimensions is not None and not 1 <= args.dimensions <= 16384:
        raise ValueError("Dimensions must be 1..16384")
    provider = values["RAGS_EMBEDDING_PROVIDER"]
    if provider == "local" and values["RAGS_EMBEDDING_MODEL"] != "offline-hash-v1":
        raise ValueError("The local demo provider uses offline-hash-v1; select another provider for a trained model")
    if args.dimensions is not None and provider not in {"openai", "sentence_transformers", "router", "ollama"}:
        raise ValueError("Requested dimensions are unavailable for this provider")
    if args.revision and provider != "sentence_transformers":
        raise ValueError("Revision pins currently require sentence_transformers")
    if values["RAGS_EMBEDDING_BASE_URL"]:
        from rags.config import endpoint
        values["RAGS_EMBEDDING_BASE_URL"] = endpoint(values["RAGS_EMBEDDING_BASE_URL"], "Embedding endpoint")
    if values["RAGS_EMBEDDING_AUTH_MODE"] == "none":
        from rags.model_profiles import validate_local_no_auth_endpoint
        if provider not in {"openai", "ollama"}:
            raise ValueError("No-auth mode requires a supported loopback embedding endpoint")
        validate_local_no_auth_endpoint(values["RAGS_EMBEDDING_BASE_URL"])
    if bool(args.embedding_tokenizer) != (args.embedding_max_tokens is not None):
        raise ValueError("Supply --embedding-tokenizer and --embedding-max-tokens together")
    if args.embedding_max_tokens is not None:
        if not 1 <= args.embedding_max_tokens <= 100000:
            raise ValueError("Embedding token window must be 1..100000")
        if args.embedding_tokenizer == "unicode":
            raise ValueError("Unicode code points cannot establish a model embedding window")
        values["RAGS_EMBEDDING_MAX_TOKENS"] = str(args.embedding_max_tokens)
    for name, env in (("tokenizer", "RAGS_TOKENIZER_JSON"),
                      ("generation_tokenizer", "RAGS_GENERATION_TOKENIZER_JSON"),
                      ("embedding_tokenizer", "RAGS_EMBEDDING_TOKENIZER_JSON")):
        choice, path = getattr(args, name), getattr(args, name + "_path")
        if path is not None and not choice:
            raise ValueError("Select a tokenizer when supplying its artifact path")
        if not choice:
            continue
        config = tokenizer_preset(choice, str(path.absolute()) if path is not None else None)
        TokenizerConfig(**config)
        if path is not None and (path.is_symlink() or not path.is_file()):
            raise ValueError("Tokenizer artifact must be an existing regular, non-symlink file")
        values[env] = json.dumps(config, ensure_ascii=False, separators=(",", ":"))
    from rags.config import Settings
    optional = {}
    for suffix in ("REVISION", "QUERY_INPUT_TYPE", "DOCUMENT_INPUT_TYPE"):
        optional["embedding_" + suffix.lower()] = values["RAGS_EMBEDDING_" + suffix] or None
    for suffix in ("QUERY_PREFIX", "DOCUMENT_PREFIX", "SPACE_ID"):
        optional["embedding_" + suffix.lower()] = values["RAGS_EMBEDDING_" + suffix]
    for suffix in ("DIMENSIONS", "MAX_TOKENS"):
        raw = values["RAGS_EMBEDDING_" + suffix]
        optional["embedding_" + suffix.lower()] = int(raw) if raw else None
    raw_tokenizer = values["RAGS_EMBEDDING_TOKENIZER_JSON"]
    optional["embedding_tokenizer_config"] = json.loads(raw_tokenizer) if raw_tokenizer else None
    # Reuse the runtime's pure contract validation, with no credentials or
    # ambient environment. This does not instantiate providers or read weights.
    Settings(db_path=":memory:", keys={}, embedding_provider=provider, model_provider="extractive",
             openai_key="", openai_base_url="https://api.openai.com/v1",
             embedding_model=values["RAGS_EMBEDDING_MODEL"], chat_model="unused",
             anthropic_key="", anthropic_model="unused",
             embedding_base_url=values["RAGS_EMBEDDING_BASE_URL"] or None,
             embedding_auth_mode=values["RAGS_EMBEDDING_AUTH_MODE"], **optional)
    return values


def write_selection(path: Path, values: dict[str, str]) -> None:
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise ValueError("Output directory must exist and must not be a symlink")
    if set(values) - MODEL_ENV_KEYS:
        raise ValueError("Model overlays may contain only model and tokenizer settings")
    body = ("# RAGs model selection. No credentials are stored in this file.\n"
            "# Existing process environment takes precedence. Restart after changes.\n"
            "# Reindex documents or create a new collection after changing embedding identity.\n"
            + "".join(name + "=" + shlex.quote(value) + "\n" for name, value in sorted(values.items())))
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write(body)
    except BaseException:
        path.unlink()
        raise


def run(args: argparse.Namespace) -> int:
    if args.command == "models":
        if args.model_action == "show":
            value = embedding_preset(args.preset).public()
            print(json.dumps(value, ensure_ascii=False, indent=2))
            return 0
        values = catalog()["embeddings"]
        if args.provider:
            values = [item for item in values if item["provider"] == args.provider]
        if args.language:
            values = [item for item in values if args.language in item["languages"].split(",")]
        if args.json:
            print(json.dumps(values, ensure_ascii=False, indent=2))
        else:
            print(f"{'ID':<28} {'PROVIDER':<23} {'LANGUAGES':<16} MODEL")
            for item in values:
                print(f"{item['id']:<28} {item['provider']:<23} {item['languages']:<16} {item['model']}")
            print("Details: rags models show <ID>\nChoose: rags configure --embedding <ID>")
        return 0
    if args.command == "tokenizers":
        if args.json:
            print(json.dumps(catalog()["tokenizers"], ensure_ascii=False, indent=2))
        else:
            for item in TOKENIZERS:
                print(f"{item['id']:<18} {item['description']}")
        return 0
    if not args.embedding and not args.provider:
        if not sys.stdin.isatty():
            raise ValueError("Use --embedding <preset> or --provider <provider> --model <ID>; interactive selection requires a terminal")
        args.embedding = _pick([item.public() for item in EMBEDDINGS], "Embedding model", "bge-small-en")
        if not args.tokenizer:
            args.tokenizer = _pick(TOKENIZERS, "Chunking tokenizer (independent of embedding tokenizer)", "unicode")
        if args.tokenizer in {"huggingface", "sentencepiece"} and args.tokenizer_path is None:
            args.tokenizer_path = Path(input("Local tokenizer artifact path: ").strip())
    values = selection_environment(args)
    write_selection(args.output, values)
    print("Created " + str(args.output))
    if args.embedding:
        preset = embedding_preset(args.embedding)
        if preset.extra:
            print("Install runtime: uv pip install 'rags-platform[" + preset.extra + "]'")
        if preset.provider == "ollama":
            print("Prepare model: ollama pull " + preset.model)
    print("Check: rags doctor --env-file .env --model-config " + shlex.quote(str(args.output)) + " --no-db")
    print("Start: rags serve --env-file .env --model-config " + shlex.quote(str(args.output)))
    print("Existing environment and operator feature permissions still apply. Reindex documents or create a new collection after switching models.")
    return 0
