"""Discovery and explicit, non-destructive model/tokenizer configuration tools."""
from __future__ import annotations

import json
import os
import shlex
import sys
from importlib import resources
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from rags.file_io import read_regular_text
from rags.model_catalog import (
    embedding_catalog,
    get_embedding_preset,
    tokenizer_catalog,
)


class SelectionError(ValueError):
    """Messages contain only deliberate public guidance, never input contents."""


_EMBEDDING_FIELDS = {
    "provider": "PROVIDER", "model": "MODEL", "revision": "REVISION",
    "base_url": "BASE_URL", "auth_mode": "AUTH_MODE", "dimensions": "DIMENSIONS",
    "query_prefix": "QUERY_PREFIX", "document_prefix": "DOCUMENT_PREFIX",
    "normalize": "NORMALIZE", "input_policy": "INPUT_POLICY", "tokenizer": "TOKENIZER_JSON",
    "token_limit": "TOKEN_LIMIT", "special_tokens": "SPECIAL_TOKENS", "space_id": "SPACE_ID",
}
_RUNTIMES = {"fastembed": ("fastembed", "encoders"),
             "sentence_transformers": ("sentence-transformers", "sentence-transformers"),
             "huggingface": ("tokenizers", "tokenizers"),
             "sentencepiece": ("sentencepiece", "tokenizers"), "tiktoken": ("tiktoken", None)}


def add_selection_arguments(parser) -> None:
    choices = parser.add_mutually_exclusive_group()
    choices.add_argument("--embedding", metavar="PRESET", help="Embedding preset from rags models list")
    choices.add_argument("--embedding-config", type=Path, metavar="JSON_FILE",
                         help="Bounded operator embedding contract; no inline credentials")
    parser.add_argument("--source-tokenizer", metavar="ID", help="auto, preset ID or backend:registered-alias")
    parser.add_argument("--generation-tokenizer", metavar="ID", help="Independent prompt-text counter")
    parser.add_argument("--tokenizer-registry", type=Path, metavar="JSON_FILE",
                        help="Operator registry of local tokenizer aliases, paths and SHA-256 digests")
    parser.add_argument("--chunk-tokens", type=int, help="Explicit source chunk budget")
    parser.add_argument("--chunk-overlap", type=int, help="Explicit source overlap budget")


def add_commands(subparsers) -> None:
    for name, label in (("models", "Embedding models"), ("tokenizers", "Text counters")):
        group = subparsers.add_parser(name, help=label + ": pure offline discovery")
        commands = group.add_subparsers(dest="discovery_command", required=True)
        listing = commands.add_parser("list", help="List curated choices without loading anything")
        detail = commands.add_parser("show", help="Explain one choice and its limitations")
        detail.add_argument("id")
        for command in (listing, detail):
            command.add_argument("--json", action="store_true")
        if name == "models":
            listing.add_argument("--kind", choices=("embedding",), default="embedding")
            listing.add_argument("--language", help="Filter model-card language tags, e.g. en or zh")
            listing.add_argument("--runtime", help="Filter provider, e.g. fastembed or sentence_transformers")
    config = subparsers.add_parser("config", help="Preview a change or write a new private candidate")
    actions = config.add_subparsers(dest="config_command", required=True)
    for name in ("plan", "write"):
        command = actions.add_parser(name)
        command.add_argument("--env-file", type=Path, required=True)
        command.add_argument("--json", action="store_true")
        add_selection_arguments(command)
        if name == "write":
            command.add_argument("--output", type=Path, required=True)


def update_env_text(raw: str, updates: dict[str, str | None]) -> str:
    """Replace selected assignments while preserving all other bytes and comments."""
    from rags.cli import ENV_NAME, MAX_ENV_SIZE
    if any(not ENV_NAME.fullmatch(name) or (value is not None and (
            not isinstance(value, str) or any(c in value for c in "\r\n\x00")))
           for name, value in updates.items()):
        raise SelectionError("Selected settings require single-line shell-quoted values")
    remaining = dict(updates)
    result = []
    for line in raw.splitlines(keepends=True):
        name = line.partition("=")[0]
        if name not in remaining:
            result.append(line)
            continue
        value = remaining.pop(name)
        if value is not None:
            result.append(name + "=" + shlex.quote(value) + "\n")
    additions = [name + "=" + shlex.quote(value) + "\n"
                 for name, value in remaining.items() if value is not None]
    if additions and result and not result[-1].endswith("\n"):
        result.append("\n")
    rendered = "".join(result + additions)
    try:
        size = len(rendered.encode("utf-8"))
    except UnicodeError as exc:
        raise SelectionError("Selected settings must contain valid UTF-8 text") from exc
    if size > MAX_ENV_SIZE:
        raise SelectionError("Candidate environment file exceeds 64 KiB; reduce its settings before writing")
    return rendered


def _json(value: str) -> dict:
    from rags.model_profiles import strict_json
    try:
        result = strict_json(value)
    except (ValueError, TypeError) as exc:
        raise SelectionError("Configuration JSON must contain a valid object with distinct fields") from exc
    if not isinstance(result, dict):
        raise SelectionError("Configuration JSON must be an object")
    return result


def _custom_embedding(path: Path) -> dict:
    from rags.embedding_profiles import EmbeddingProfile
    try:
        config = _json(read_regular_text(path, max_bytes=65_536))
        profile = EmbeddingProfile(**config)
    except (OSError, ValueError, TypeError) as exc:
        raise SelectionError("Invalid embedding contract file; see the model selection guide") from exc
    if profile.api_key_env:
        raise SelectionError("Direct selection uses RAGS_EMBEDDING_API_KEY; api_key_env belongs in approved profiles")
    env = {}
    for key, suffix in _EMBEDDING_FIELDS.items():
        if key not in config:
            continue
        value = config[key]
        if value is None:
            continue
        env["RAGS_EMBEDDING_" + suffix] = (json.dumps(value, separators=(",", ":"))
                                         if isinstance(value, (bool, dict)) else str(value))
    return {"id": "custom", **profile.public(), "env": env,
            "max_input_tokens": profile.token_limit,
            "dependency_extra": _RUNTIMES.get(profile.provider, (None, None))[1]}


def _exact_embedding(env: dict) -> bool:
    provider = env.get("RAGS_EMBEDDING_PROVIDER", "local")
    model = env.get("RAGS_EMBEDDING_MODEL") or env.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    return (provider in {"fastembed", "sentence_transformers"}
            or bool(env.get("RAGS_EMBEDDING_TOKENIZER_JSON"))
            or (provider == "openai" and model in {
                "text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"}))


def _known_window(env: dict) -> int | None:
    if env.get("RAGS_EMBEDDING_TOKEN_LIMIT"):
        try:
            return int(env["RAGS_EMBEDDING_TOKEN_LIMIT"])
        except (TypeError, ValueError) as exc:
            raise SelectionError("Configured embedding token limit must be an integer") from exc
    provider = env.get("RAGS_EMBEDDING_PROVIDER", "local")
    model = (env.get("RAGS_EMBEDDING_MODEL") or
             (env.get("RAGS_FASTEMBED_MODEL", "BAAI/bge-small-en-v1.5") if provider == "fastembed"
              else env.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")))
    for entry in embedding_catalog():
        if (entry["provider"] == provider and entry["model"] == model
                and (not entry["revision"] or entry["revision"] == env.get("RAGS_EMBEDDING_REVISION"))):
            return entry["max_input_tokens"]
    if provider == "openai" and model in {"text-embedding-3-large", "text-embedding-ada-002"}:
        return 8191
    return None


def _tokenizer(selector: str, env: dict, *, source: bool) -> dict:
    from rags.tokenization import TokenizerConfig
    from rags.tokenizer_artifacts import parse_tokenizer_artifacts
    if selector == "auto":
        if not source:
            raise SelectionError("Automatic selection is for source chunking; choose a generation counter explicitly")
        selector = "embedding" if _exact_embedding(env) else "utf8_bytes"
    if selector == "embedding":
        if not source or not _exact_embedding(env):
            raise SelectionError("The embedding source counter requires a configured exact encoder contract")
        return {"backend": "embedding"}
    if selector in {"unicode", "utf8_bytes"}:
        return {"backend": selector}
    if selector in {"cl100k_base", "o200k_base"}:
        return {"backend": "tiktoken", "encoding": selector}
    backend, separator, alias = selector.partition(":")
    try:
        config = {"backend": backend, "model": alias}
        TokenizerConfig(**config)
        registry = parse_tokenizer_artifacts(env.get("RAGS_TOKENIZER_ARTIFACTS_JSON") or "{}")
        if not separator or backend not in {"huggingface", "sentencepiece"} or alias not in registry:
            raise ValueError("missing registration")
        if registry[alias]["backend"] != backend:
            raise ValueError("backend differs")
    except (ValueError, TypeError) as exc:
        raise SelectionError("Choose a tokenizer preset or a matching registered backend:ALIAS") from exc
    return config


def selection_updates(args, existing: dict) -> tuple[dict, dict | None, list[str]]:
    """Build explicit deltas only; never load weights or resolve credentials."""
    from rags.chunking import ChunkConfig
    selected = None
    updates = {}
    warnings = []
    if args.tokenizer_registry:
        from rags.tokenizer_artifacts import parse_tokenizer_artifacts
        try:
            registry = parse_tokenizer_artifacts(read_regular_text(args.tokenizer_registry, max_bytes=65_536))
        except (OSError, ValueError) as exc:
            raise SelectionError("Invalid tokenizer registry file; use aliases, absolute paths and SHA-256 digests") from exc
        updates["RAGS_TOKENIZER_ARTIFACTS_JSON"] = json.dumps(registry, separators=(",", ":"))
    if args.embedding or args.embedding_config:
        try:
            selected = (get_embedding_preset(args.embedding) if args.embedding
                        else _custom_embedding(args.embedding_config))
        except ValueError as exc:
            raise SelectionError("Unknown or invalid embedding selection; run rags models list/show") from exc
        updates.update({"RAGS_EMBEDDING_" + suffix: None for suffix in _EMBEDDING_FIELDS.values()})
        updates.update(selected["env"])
    effective = {**existing, **{key: value for key, value in updates.items() if value is not None}}
    for key, value in updates.items():
        if value is None:
            effective.pop(key, None)
    source = args.source_tokenizer or ("auto" if selected else None)
    for selector, key, is_source in ((source, "RAGS_TOKENIZER_JSON", True),
                                     (args.generation_tokenizer, "RAGS_GENERATION_TOKENIZER_JSON", False)):
        if selector:
            updates[key] = json.dumps(_tokenizer(selector, effective, source=is_source), separators=(",", ":"))
    if selected or source in {"auto", "embedding"} or args.chunk_tokens is not None or args.chunk_overlap is not None:
        window = _known_window(effective)
        recommended = min(384, max(1, window - 32)) if window else 384
        chunk = _json(existing["RAGS_CHUNKING_JSON"]) if existing.get("RAGS_CHUNKING_JSON") else {
            "strategy": "structure", "max_tokens": recommended, "overlap_tokens": min(64, recommended // 8)}
        if args.chunk_tokens is not None:
            chunk["max_tokens"] = args.chunk_tokens
        if args.chunk_overlap is not None:
            chunk["overlap_tokens"] = args.chunk_overlap
        try:
            ChunkConfig(**chunk)
        except (TypeError, ValueError) as exc:
            raise SelectionError("Invalid chunk budget; overlap must be smaller than a positive chunk size") from exc
        updates["RAGS_CHUNKING_JSON"] = json.dumps(chunk, separators=(",", ":"))
        active_source = _json(updates.get("RAGS_TOKENIZER_JSON") or effective.get("RAGS_TOKENIZER_JSON")
                              or '{"backend":"unicode"}')
        if window and active_source["backend"] == "embedding" and chunk["max_tokens"] > window:
            raise SelectionError("Chunk budget exceeds the selected encoder window; set --chunk-tokens and "
                                 "--chunk-overlap explicitly")
        if window and chunk["max_tokens"] > recommended:
            warnings.append("Existing or explicit chunk budget exceeds the suggested encoder budget; "
                            f"set --chunk-tokens {recommended} and --chunk-overlap {min(64, recommended // 8)}")
    if selected and not _exact_embedding(effective):
        warnings.append("Automatic source counting uses conservative UTF-8 bytes; the embedding provider still "
                        "validates complete input independently")
    return updates, selected, warnings


def _artifact_counter_changed(config: dict | None, before: dict, after: dict) -> bool:
    """Only the selected artifact's backend/digest affects its counting identity."""
    from rags.tokenizer_artifacts import parse_tokenizer_artifacts
    if not config or config.get("backend") not in {"huggingface", "sentencepiece"}:
        return False
    try:
        old = parse_tokenizer_artifacts(before.get("RAGS_TOKENIZER_ARTIFACTS_JSON") or "{}")
        new = parse_tokenizer_artifacts(after.get("RAGS_TOKENIZER_ARTIFACTS_JSON") or "{}")
    except ValueError as exc:
        raise SelectionError("Existing tokenizer registry is invalid; original configuration preserved") from exc
    alias = config.get("model")
    def identity(entry):
        return (entry.get("backend"), entry.get("sha256")) if entry else None
    return identity(old.get(alias)) != identity(new.get(alias))


def _requirements(selected: dict | None) -> list[str]:
    if not selected:
        return []
    commands = []
    if selected.get("dependency_extra"):
        commands.append("python -m pip install -e " + shlex.quote(".[" + selected["dependency_extra"] + "]"))
    if selected["provider"] == "ollama":
        commands.append("ollama pull " + shlex.quote(selected["model"]))
    return commands


def _interactive(args) -> None:
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        raise SelectionError("Interactive setup requires a terminal; use --embedding and tokenizer flags in scripts")
    if args.embedding or args.embedding_config:
        raise SelectionError("Choose --interactive or an explicit embedding selection")
    choices = embedding_catalog()
    print("Choose an embedding model. No model will be downloaded during setup.")
    for index, item in enumerate(choices, 1):
        print(f" {index}. {item['label']} ({item['execution']}; {item['cost_category']})")
    try:
        answer = input("Model number (blank cancels): ").strip()
        if not answer:
            raise SelectionError("Setup cancelled; no files were created")
        index = int(answer) - 1
        if not 0 <= index < len(choices):
            raise ValueError("out of range")
        args.embedding = choices[index]["id"]
        chosen = choices[index]
        print(chosen["description"])
        for limitation in chosen["limitations"]:
            print(" - " + limitation)
        if input("Create configuration with this selection? [y/N]: ").strip().lower() not in {"y", "yes"}:
            raise SelectionError("Setup cancelled; no files were created")
    except (EOFError, KeyboardInterrupt) as exc:
        raise SelectionError("Setup cancelled; no files were created") from exc
    except ValueError as exc:
        if isinstance(exc, SelectionError):
            raise
        raise SelectionError("Choose a displayed model number; no files were created") from exc


def run_init(args, create) -> int:
    from rags.cli import parse_env_text
    if args.interactive:
        _interactive(args)
    template = resources.files("rags").joinpath("templates", args.profile + ".env").read_text(encoding="utf-8")
    updates, selected, warnings = selection_updates(args, parse_env_text(template))
    paths = create(args.directory, args.profile, env_overrides=updates or None)
    for path in paths:
        print("Created " + str(path))
    for message in warnings:
        print("Note: " + message)
    for command in _requirements(selected):
        print("Setup (from the matching source checkout): " + command)
    if selected:
        print("Selection configured; runtime/model validation has not run.")
    print("Check with: rags doctor --env-file " + shlex.quote(str(paths[0])) + " --no-db")
    print("Start with: rags serve --env-file " + shlex.quote(str(paths[0])))
    return 0


def _feature_blockers(env: dict, updates: dict) -> list[str]:
    from rags.features import _parse
    flags = {}
    try:
        if env.get("RAGS_FEATURES_FILE"):
            flags.update(_parse(read_regular_text(Path(env["RAGS_FEATURES_FILE"]), max_bytes=65_536)))
        flags.update(_parse(env.get("RAGS_FEATURES_JSON") or "{}"))
    except (OSError, ValueError) as exc:
        raise SelectionError("Cannot validate existing feature restrictions; original configuration preserved") from exc
    required = []
    if updates.get("RAGS_EMBEDDING_PROVIDER"):
        required.append("embedding." + updates["RAGS_EMBEDDING_PROVIDER"])
    for name in ("RAGS_TOKENIZER_JSON", "RAGS_GENERATION_TOKENIZER_JSON"):
        if updates.get(name):
            required.append("tokenizer." + _json(updates[name])["backend"])
    return sorted(name for name in required if flags.get(name) is False)


def configuration_plan(args) -> tuple[dict, str]:
    from rags.cli import read_env_file
    from rags.tokenization import TokenizerConfig
    raw, existing = read_env_file(args.env_file)
    updates, selected, warnings = selection_updates(args, existing)
    if not updates:
        raise SelectionError("Select an embedding, tokenizer or chunk budget to plan")
    candidate = update_env_text(raw, updates)
    effective = {key: value for key, value in {**existing, **updates}.items() if value is not None}
    blockers = _feature_blockers(effective, updates)
    feature_env = {name: os.environ[name] for name in ("RAGS_FEATURES_FILE", "RAGS_FEATURES_JSON")
                   if name in os.environ}
    if feature_env:
        blockers = sorted(set(blockers) | set(_feature_blockers({**effective, **feature_env}, updates)))
    if effective.get("RAGS_EMBEDDING_AUTH_MODE") == "none" and any(effective.get(name) for name in (
            "RAGS_EMBEDDING_API_KEY", "RAGS_EMBEDDING_API_KEY_FILE")):
        blockers.append("Existing embedding credentials conflict with selected no-auth policy")
    changed = sorted(key for key, value in updates.items() if existing.get(key) != value)
    embedding_changed = any(name.startswith("RAGS_EMBEDDING_") for name in changed)
    source_changed = bool(set(changed) & {"RAGS_TOKENIZER_JSON", "RAGS_CHUNKING_JSON"})
    source_tokenizer = _json(effective.get("RAGS_TOKENIZER_JSON", '{"backend":"unicode"}'))
    generation_tokenizer = (_json(effective["RAGS_GENERATION_TOKENIZER_JSON"])
                            if effective.get("RAGS_GENERATION_TOKENIZER_JSON") else None)
    try:
        TokenizerConfig(**source_tokenizer)
        if generation_tokenizer is not None:
            TokenizerConfig(**generation_tokenizer)
    except (TypeError, ValueError) as exc:
        raise SelectionError("Existing tokenizer settings are invalid; original configuration preserved") from exc
    if "RAGS_TOKENIZER_ARTIFACTS_JSON" in changed:
        source_changed = source_changed or _artifact_counter_changed(source_tokenizer, existing, effective)
        embedding_counter = (_json(effective["RAGS_EMBEDDING_TOKENIZER_JSON"])
                             if effective.get("RAGS_EMBEDDING_TOKENIZER_JSON") else None)
        embedding_changed = embedding_changed or _artifact_counter_changed(embedding_counter, existing, effective)
    report = {
        "status": "blocked" if blockers else "planned", "validation": "structural_only",
        "runtime_validation": "not_run", "changed_settings": changed,
        "embedding": ({key: selected[key] for key in ("id", "provider", "model")} if selected else None),
        "source_tokenizer": source_tokenizer,
        "generation_tokenizer": generation_tokenizer or {"backend": "provider_default"},
        "blocked_by": blockers, "warnings": warnings,
        "process_overrides": sorted(name for name in set(updates) | set(feature_env) if name in os.environ
                                    and os.environ[name] != effective.get(name)),
        "index_actions": (["Re-embed source text into a new collection; a fork preserves the old vector space.",
                           "Reindex documents with an explicit processing selection after restart."] if embedding_changed
                          else ["Reindex documents explicitly to adopt the new source chunking profile."] if source_changed
                          else ["Generation counting changes prompt budgets only; stored vectors are unchanged."]),
        "setup_commands": _requirements(selected),
        "next_steps": ["Review this candidate and any preserved feature restrictions.",
                       "Run any pip setup command from the matching RAGs source checkout.",
                       "Run rags doctor --env-file CANDIDATE --no-db; it does not run inference.",
                       "Restart using the reviewed candidate, then run a representative ingest/query validation."],
    }
    return report, candidate


def _write_candidate(path: Path, body: str) -> None:
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise SelectionError("Candidate directory must exist and must not be a symlink")
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    except FileExistsError as exc:
        raise SelectionError("Candidate already exists; choose a new output path") from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write(body)
    except BaseException:
        path.unlink()
        raise


def _show(entry: dict) -> None:
    print(entry["id"] + " — " + entry["label"])
    print(entry["description"])
    for key in ("provider", "model", "revision", "languages", "dimensions", "max_input_tokens",
                "input_policy", "execution", "cost_category", "backend", "count_kind", "unit"):
        if entry.get(key) is not None:
            value = ", ".join(entry[key]) if isinstance(entry[key], list) else str(entry[key])
            print(f"{key}: {value}")
    if entry.get("dependency_extra"):
        print("Install from matching source checkout: python -m pip install -e '.[" + entry["dependency_extra"] + "]'")
    for limitation in entry["limitations"]:
        print(" - " + limitation)
    if entry.get("source_url"):
        print("Source: " + entry["source_url"])


def run(args) -> int:
    if args.command in {"models", "tokenizers"}:
        entries = embedding_catalog() if args.command == "models" else tokenizer_catalog()
        if args.discovery_command == "show":
            entries = [entry for entry in entries if entry["id"] == args.id]
            if not entries:
                raise SelectionError("Unknown catalog ID; run the matching list command")
        elif args.command == "models":
            if args.language:
                entries = [entry for entry in entries if args.language.lower() in entry["languages"]
                           or "any" in entry["languages"]]
            if args.runtime:
                entries = [entry for entry in entries if entry["provider"] == args.runtime]
        if args.json:
            print(json.dumps(entries[0] if args.discovery_command == "show" else entries, indent=2))
        elif args.discovery_command == "show":
            _show(entries[0])
        else:
            for entry in entries:
                runtime = entry.get("provider", entry.get("backend"))
                print(f"{entry['id']:22s} {entry['label']} [{runtime}]")
            print("Use show ID for input limits, dependencies and model-card sources; listing loads no models.")
        return 0
    report, body = configuration_plan(args)
    if args.config_command == "write" and not report["blocked_by"]:
        _write_candidate(args.output, body)
        report["status"] = "written"
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("Configuration " + report["status"] + "; runtime/model validation has not run.")
        print("Changed settings: " + (", ".join(report["changed_settings"]) or "none"))
        for key in ("blocked_by", "warnings", "index_actions", "setup_commands", "next_steps"):
            for item in report[key]:
                print(" - " + item)
        if report["process_overrides"]:
            print("Process environment overrides candidate values for: " + ", ".join(report["process_overrides"]))
        if report["status"] == "written":
            print("Created " + str(args.output) + " with private permissions; original files were preserved.")
    return 1 if report["blocked_by"] else 0


def readiness_checks(settings) -> list[dict]:
    """Inspect metadata and artifact presence without importing optional runtimes."""
    checks = []
    roles = [("embedding_runtime", settings.embedding_provider, None),
             ("source_tokenizer", settings.tokenizer_config["backend"], settings.tokenizer_config),
             ("generation_tokenizer", settings.generation_tokenizer_config["backend"],
              settings.generation_tokenizer_config)]
    for name, backend, config in roles:
        status = "ok"
        detail = backend + "; configured; inference/tokenizer validation not run"
        permission = ("embedding." if name == "embedding_runtime" else "tokenizer.") + backend
        if not settings.features.enabled(permission):
            status = "warning"
            detail += "; feature disabled"
        if backend in _RUNTIMES:
            package, extra = _RUNTIMES[backend]
            try:
                installed = version(package)
                detail += "; installed=" + installed
            except PackageNotFoundError:
                status = "warning"
                detail += "; dependency missing; install " + ("rags-platform[" + extra + "]" if extra else package)
        if backend in {"huggingface", "sentencepiece"}:
            entry = settings.tokenizer_artifacts.get(config.get("model"))
            if entry:
                path = Path(entry["path"])
                present = path.is_file() and not path.is_symlink()
                detail += "; artifact=" + ("present (digest not verified)" if present else "unavailable")
                if not present:
                    status = "warning"
            else:
                status = "error"
                detail += "; artifact alias is not registered"
        if backend == "embedding":
            profile = settings.embedding_profile
            effective = {"RAGS_EMBEDDING_PROVIDER": profile.provider, "RAGS_EMBEDDING_MODEL": profile.model,
                         "RAGS_EMBEDDING_TOKENIZER_JSON": bool(profile.tokenizer)}
            if not _exact_embedding(effective):
                status = "error"
                detail += "; exact encoder contract unavailable; select utf8_bytes"
        checks.append({"name": name, "status": status, "detail": detail, "validation": "not_run"})
    return checks
