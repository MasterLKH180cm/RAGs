"""Selection UX without optional runtimes, downloads, or live provider access."""
import json
import os
import stat
from pathlib import Path

import pytest

from rags import cli, model_cli
from rags.config import Settings
from rags.model_catalog import embedding_catalog, get_embedding_preset, tokenizer_catalog

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def clean_environment():
    prefixes = ("RAGS_", "OPENAI_", "ANTHROPIC_")
    original = {name: value for name, value in os.environ.items() if name.startswith(prefixes)}
    for name in original:
        os.environ.pop(name)
    try:
        yield
    finally:
        for name in tuple(os.environ):
            if name.startswith(prefixes):
                os.environ.pop(name)
        os.environ.update(original)


def test_discovery_is_pure_and_defensive(monkeypatch, capsys):
    monkeypatch.setenv("RAGS_KEYS_JSON", "private-invalid-value")
    monkeypatch.setattr(Settings, "from_env", lambda: pytest.fail("Discovery must not validate deployment settings"))
    monkeypatch.setattr(cli, "version", lambda _: "0.9.0")  # Installed application metadata is independent.
    monkeypatch.setattr(Path, "open", lambda *a, **k: pytest.fail("Catalog must not read files"))
    monkeypatch.setattr(model_cli, "version", lambda *a: pytest.fail("Catalog does not probe runtime state"))
    assert cli.main(["models", "list", "--language", "zh", "--runtime", "fastembed", "--json"]) == 0
    entries = json.loads(capsys.readouterr().out)
    assert [item["id"] for item in entries] == ["bge-small-zh"]
    assert "env" not in entries[0]
    preset = get_embedding_preset("bge-small-zh")
    preset["languages"].clear()
    preset["env"]["RAGS_EMBEDDING_MODEL"] = "changed"
    assert get_embedding_preset("bge-small-zh")["languages"] == ["zh"]
    assert cli.main(["tokenizers", "show", "huggingface", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["requires_artifact"] is True


@pytest.mark.parametrize("identifier", [item["id"] for item in embedding_catalog()])
def test_init_selected_model_matches_feature_permissions_and_safe_source_budget(tmp_path, identifier, capsys):
    assert cli.main(["init", "--directory", str(tmp_path), "--embedding", identifier]) == 0
    _, env = cli.read_env_file(tmp_path / ".env")
    preset = get_embedding_preset(identifier)
    assert env["RAGS_EMBEDDING_PROVIDER"] == preset["provider"]
    flags = json.loads((tmp_path / "rags.features.json").read_text())
    assert flags["embedding." + preset["provider"]] is True
    assert all(value is False for name, value in flags.items()
               if name.startswith("embedding.") and name != "embedding." + preset["provider"])
    # Feature files use False to disable; selected provider must be allowed.
    cli.load_env_file(tmp_path / ".env")
    settings = Settings.from_env()
    assert settings.embedding_provider == preset["provider"]
    assert settings.features.enabled("embedding." + preset["provider"])
    if preset["max_input_tokens"]:
        assert settings.chunking_config["max_tokens"] <= preset["max_input_tokens"]


def test_init_chinese_encoder_configuration_is_loadable_and_does_not_download(tmp_path, monkeypatch, capsys):
    assert cli.main(["init", "--directory", str(tmp_path), "--embedding", "bge-small-zh"]) == 0
    cli.load_env_file(tmp_path / ".env")
    settings = Settings.from_env()
    assert settings.features.enabled("embedding.fastembed")
    assert not settings.features.enabled("embedding.openai")
    assert settings.tokenizer_config == {"backend": "embedding"}
    assert settings.chunking_config["max_tokens"] == 384
    assert settings.chunking_config["overlap_tokens"] == 48
    assert "runtime/model validation has not run" in capsys.readouterr().out


def test_base_profile_generation_and_explicit_tokenizer_selection_are_orthogonal(tmp_path):
    assert cli.main(["init", "--profile", "openai", "--directory", str(tmp_path),
                     "--embedding", "st-e5-small", "--generation-tokenizer", "unicode"]) == 0
    _, env = cli.read_env_file(tmp_path / ".env")
    assert env["RAGS_MODEL_PROVIDER"] == "openai"
    assert env["RAGS_EMBEDDING_QUERY_PREFIX"] == "query: "
    assert env["RAGS_EMBEDDING_DOCUMENT_PREFIX"] == "passage: "
    assert json.loads(env["RAGS_GENERATION_TOKENIZER_JSON"]) == {"backend": "unicode"}


def test_init_auto_uses_base_profile_provider_without_embedding_override(tmp_path):
    assert cli.main(["init", "--profile", "openai", "--directory", str(tmp_path),
                     "--source-tokenizer", "auto"]) == 0
    _, env = cli.read_env_file(tmp_path / ".env")
    assert json.loads(env["RAGS_TOKENIZER_JSON"]) == {"backend": "embedding"}


@pytest.mark.parametrize("answer", ["", "2", "invalid", "100"])
def test_interactive_cancellation_or_bad_input_creates_no_files(tmp_path, monkeypatch, answer):
    monkeypatch.setattr(model_cli.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(model_cli.sys.stdout, "isatty", lambda: True)
    replies = iter([answer, "no"])
    monkeypatch.setattr("builtins.input", lambda _: next(replies))
    assert cli.main(["init", "--directory", str(tmp_path), "--interactive"]) == 1
    assert list(tmp_path.iterdir()) == []


def test_interactive_requires_explicit_terminal_and_can_confirm(tmp_path, monkeypatch):
    monkeypatch.setattr(model_cli.sys.stdin, "isatty", lambda: False)
    assert cli.main(["init", "--directory", str(tmp_path), "--interactive"]) == 1
    assert not (tmp_path / ".env").exists()
    monkeypatch.setattr(model_cli.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(model_cli.sys.stdout, "isatty", lambda: True)
    replies = iter(["3", "yes"])
    monkeypatch.setattr("builtins.input", lambda _: next(replies))
    assert cli.main(["init", "--directory", str(tmp_path), "--interactive"]) == 0
    assert cli.read_env_file(tmp_path / ".env")[1]["RAGS_EMBEDDING_MODEL"] == "BAAI/bge-small-zh-v1.5"


def test_plan_preserves_existing_feature_restrictions_and_never_writes(tmp_path, capsys):
    env_path, flags_path = cli.init_config(tmp_path, "offline")
    original = env_path.read_bytes(), flags_path.read_bytes()
    output = tmp_path / ".env.next"
    assert cli.main(["config", "write", "--env-file", str(env_path), "--embedding", "bge-small-en",
                     "--output", str(output), "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["blocked_by"] == ["embedding.fastembed"]
    assert not output.exists()
    assert original == (env_path.read_bytes(), flags_path.read_bytes())
    assert not (tmp_path / "data").exists()


def test_plan_preserves_ambient_restrictions_and_environment_without_disclosing_raw_invalid_config(tmp_path, capsys, monkeypatch):
    env_path, flags_path = cli.init_config(tmp_path, "offline")
    flags_path.write_text("{}")
    monkeypatch.setenv("RAGS_FEATURES_JSON", '{"embedding.fastembed":false}')
    before = dict(os.environ)
    assert cli.main(["config", "plan", "--env-file", str(env_path), "--embedding", "bge-small-en", "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["blocked_by"] == ["embedding.fastembed"]
    assert report["process_overrides"] == ["RAGS_FEATURES_JSON"]
    assert dict(os.environ) == before
    with env_path.open("a") as stream:
        stream.write('RAGS_GENERATION_TOKENIZER_JSON=\'{"backend":"unicode","api_key":"PRIVATE"}\'\n')
    assert cli.main(["config", "plan", "--env-file", str(env_path), "--source-tokenizer", "utf8_bytes", "--json"]) == 1
    output = capsys.readouterr()
    assert "PRIVATE" not in output.out + output.err
    assert dict(os.environ) == before


def test_candidate_is_private_no_clobber_preserves_secrets_and_surfaces_process_precedence(tmp_path, capsys, monkeypatch):
    env_path, _ = cli.init_config(tmp_path, "offline")
    with env_path.open("a") as stream:
        stream.write("CUSTOM_VALUE='preserve my value'\nRAGS_GENERATION_API_KEY='secret-credential'\n")
    original = env_path.read_bytes()
    _, values = cli.read_env_file(env_path)
    monkeypatch.setenv("RAGS_GENERATION_TOKENIZER_JSON", '{"backend":"unicode"}')
    output = tmp_path / ".env.next"
    command = ["config", "write", "--env-file", str(env_path), "--generation-tokenizer", "utf8_bytes",
               "--output", str(output), "--json"]
    assert cli.main(command) == 0
    text = capsys.readouterr().out
    report = json.loads(text)
    assert report["runtime_validation"] == "not_run"
    assert report["process_overrides"] == ["RAGS_GENERATION_TOKENIZER_JSON"]
    assert "secret-credential" not in text and values["RAGS_API_KEY"] not in text
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    candidate = cli.read_env_file(output)[1]
    assert candidate["CUSTOM_VALUE"] == "preserve my value"
    assert candidate["RAGS_GENERATION_API_KEY"] == "secret-credential"
    assert candidate["RAGS_API_KEY"] == values["RAGS_API_KEY"]
    assert env_path.read_bytes() == original
    saved = output.read_bytes()
    assert cli.main(command) == 1
    assert output.read_bytes() == saved


def test_candidate_refuses_symlink_and_removes_partial_write(tmp_path, monkeypatch):
    existing = tmp_path / "keep"
    existing.write_text("unchanged")
    output = tmp_path / "candidate"
    output.symlink_to(existing)
    with pytest.raises(model_cli.SelectionError, match="already exists"):
        model_cli._write_candidate(output, "private")
    assert existing.read_text() == "unchanged"
    output.unlink()
    monkeypatch.setattr(os, "fchmod", lambda *a: (_ for _ in ()).throw(OSError("write failure")))
    with pytest.raises(OSError):
        model_cli._write_candidate(output, "private")
    assert not output.exists()


@pytest.mark.parametrize("remaining,accepted", [(3, True), (0, False)])
def test_candidate_size_limit_applies_to_rendered_output_before_creation(tmp_path, capsys, remaining, accepted):
    env_path, _ = cli.init_config(tmp_path, "offline")
    raw = env_path.read_text() + 'RAGS_GENERATION_TOKENIZER_JSON=\'{"backend":"unicode"}\'\n'
    padding = cli.MAX_ENV_SIZE - remaining - len(raw.encode("utf-8"))
    raw += "#" + "x" * (padding - 2) + "\n"
    env_path.write_text(raw)
    assert env_path.stat().st_size == cli.MAX_ENV_SIZE - remaining
    target = tmp_path / "candidate"
    before = dict(os.environ)
    result = cli.main(["config", "write", "--env-file", str(env_path), "--generation-tokenizer", "utf8_bytes",
                       "--output", str(target), "--json"])
    assert result == (0 if accepted else 1)
    assert env_path.read_text() == raw and dict(os.environ) == before
    if accepted:
        assert target.stat().st_size == cli.MAX_ENV_SIZE
        assert json.loads(cli.read_env_file(target)[1]["RAGS_GENERATION_TOKENIZER_JSON"]) == {"backend": "utf8_bytes"}
    else:
        assert not target.exists()
        assert "exceeds 64 KiB" in capsys.readouterr().err


def test_oversized_or_invalid_utf8_init_selection_creates_neither_file(tmp_path):
    for value in ("x" * cli.MAX_ENV_SIZE, "\ud800"):
        with pytest.raises(model_cli.SelectionError):
            cli.init_config(tmp_path, "offline", env_overrides={"RAGS_EMBEDDING_DOCUMENT_PREFIX": value})
        assert list(tmp_path.iterdir()) == []


def test_existing_explicit_oversized_chunks_require_an_explicit_adjustment(tmp_path, capsys):
    env_path, _ = cli.init_config(tmp_path, "offline")
    with env_path.open("a") as stream:
        stream.write('RAGS_CHUNKING_JSON=\'{"max_tokens":850,"overlap_tokens":120}\'\n')
    original = env_path.read_bytes()
    assert cli.main(["config", "plan", "--env-file", str(env_path), "--embedding", "st-minilm"]) == 1
    assert "Chunk budget exceeds" in capsys.readouterr().err
    assert env_path.read_bytes() == original


def test_auto_source_with_existing_fastembed_uses_the_encoders_window(tmp_path, capsys):
    env_path, _ = cli.init_config(tmp_path, "offline", env_overrides={
        "RAGS_EMBEDDING_PROVIDER": "fastembed", "RAGS_EMBEDDING_MODEL": "BAAI/bge-small-en-v1.5"})
    assert cli.main(["config", "write", "--env-file", str(env_path), "--source-tokenizer", "auto",
                     "--output", str(tmp_path / "candidate")]) == 0
    values = cli.read_env_file(tmp_path / "candidate")[1]
    assert json.loads(values["RAGS_CHUNKING_JSON"])["max_tokens"] == 384
    assert json.loads(values["RAGS_TOKENIZER_JSON"])["backend"] == "embedding"


@pytest.mark.parametrize("embedding_counter", [False, True])
def test_registry_rebinding_reports_reindex_or_reembedding_requirement(tmp_path, capsys, embedding_counter):
    old = {"counter": {"backend": "huggingface", "path": str(tmp_path / "old.json"), "sha256": "a" * 64}}
    new = {"counter": {"backend": "huggingface", "path": str(tmp_path / "new.json"), "sha256": "b" * 64}}
    counter = json.dumps({"backend": "huggingface", "model": "counter"})
    updates = {"RAGS_TOKENIZER_ARTIFACTS_JSON": json.dumps(old), "RAGS_TOKENIZER_JSON": counter}
    if embedding_counter:
        updates.update({"RAGS_EMBEDDING_PROVIDER": "openai", "RAGS_EMBEDDING_MODEL": "custom",
                        "RAGS_EMBEDDING_TOKENIZER_JSON": counter, "RAGS_EMBEDDING_TOKEN_LIMIT": "512",
                        "RAGS_EMBEDDING_SPECIAL_TOKENS": "0"})
    env_path, _ = cli.init_config(tmp_path, "offline", env_overrides=updates)
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(new))
    assert cli.main(["config", "plan", "--env-file", str(env_path),
                     "--tokenizer-registry", str(registry_path), "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    expected = "Re-embed" if embedding_counter else "Reindex"
    assert report["index_actions"][0].startswith(expected)
    assert str(tmp_path) not in json.dumps(report)


def test_registered_alias_selection_does_not_open_or_disclose_artifact(tmp_path, capsys):
    artifact_path = tmp_path / "operator-private.model"
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"zh-count": {"backend": "sentencepiece", "path": str(artifact_path),
                                                "sha256": "a" * 64}}))
    assert cli.main(["init", "--directory", str(tmp_path), "--source-tokenizer", "sentencepiece:zh-count",
                     "--tokenizer-registry", str(registry)]) == 0
    env_path = tmp_path / ".env"
    assert not artifact_path.exists()
    capsys.readouterr()
    assert cli.main(["config", "plan", "--env-file", str(env_path),
                     "--generation-tokenizer", "sentencepiece:zh-count", "--json"]) == 0
    report = capsys.readouterr().out
    assert str(artifact_path) not in report
    assert json.loads(report)["generation_tokenizer"] == {"backend": "sentencepiece", "model": "zh-count"}


@pytest.mark.parametrize("selector", ["embedding", "auto", "huggingface:/etc/passwd", "sentencepiece:missing"])
def test_invalid_generation_or_unregistered_artifact_selection_has_no_side_effects(tmp_path, selector):
    assert cli.main(["init", "--directory", str(tmp_path), "--generation-tokenizer", selector]) == 1
    assert not (tmp_path / ".env").exists()


def test_custom_model_contract_is_validated_and_old_identity_options_are_cleared(tmp_path, capsys):
    model = tmp_path / "model.json"
    model.write_text(json.dumps({"provider": "openai", "model": "custom-encoder", "dimensions": 128,
                                 "base_url": "https://models.example/v1", "input_policy": "exact",
                                 "tokenizer": {"backend": "tiktoken", "encoding": "cl100k_base"},
                                 "token_limit": 512, "special_tokens": 0}))
    assert cli.main(["init", "--directory", str(tmp_path), "--embedding-config", str(model)]) == 0
    values = cli.read_env_file(tmp_path / ".env")[1]
    assert values["RAGS_EMBEDDING_MODEL"] == "custom-encoder"
    assert json.loads(values["RAGS_TOKENIZER_JSON"]) == {"backend": "embedding"}
    assert values["RAGS_EMBEDDING_TOKEN_LIMIT"] == "512"
    assert str(model) not in capsys.readouterr().out


def test_doctor_reports_router_alias_and_missing_optional_runtime_truthfully(tmp_path, monkeypatch):
    env_path, _ = cli.init_config(tmp_path, "inkling-local")
    cli.load_env_file(env_path)
    monkeypatch.setenv("RAGS_MODEL_PROVIDER", "router")
    monkeypatch.setenv("RAGS_GENERATION_MODEL", "inkling")
    monkeypatch.setenv("RAGS_GENERATION_AUTH_MODE", "required")
    result = cli.diagnose(check_db=False)
    generation = next(item for item in result["checks"] if item["name"] == "generation")
    assert "model=inkling" in generation["detail"] and "model=extractive" not in generation["detail"]
    monkeypatch.setenv("RAGS_EMBEDDING_PROVIDER", "sentence_transformers")
    monkeypatch.setenv("RAGS_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    monkeypatch.setenv("RAGS_EMBEDDING_REVISION", "1110a243fdf4706b3f48f1d95db1a4f5529b4d41")
    monkeypatch.setattr(model_cli, "version", lambda _: (_ for _ in ()).throw(model_cli.PackageNotFoundError()))
    report = cli.diagnose(check_db=False)
    runtime = next(item for item in report["checks"] if item["name"] == "embedding_runtime")
    assert runtime["status"] == "warning" and runtime["validation"] == "not_run"
    assert "dependency missing" in runtime["detail"]


def test_catalog_details_and_empty_filters_are_useful(capsys):
    for command, identifier in (("models", "st-e5-small"), ("tokenizers", "utf8_bytes")):
        assert cli.main([command, "show", identifier]) == 0
        assert identifier in capsys.readouterr().out
    assert cli.main(["models", "show", "missing"]) == 1
    assert cli.main(["models", "list", "--runtime", "unknown", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == []
    assert {item["backend"] for item in tokenizer_catalog()} >= {"huggingface", "sentencepiece", "embedding"}


@pytest.mark.parametrize("backend", ["unicode", "utf8_bytes", "tiktoken", "huggingface", "sentencepiece", "embedding"])
def test_discovery_and_catalog_counter_vocabulary_matches_runtime_profiles(backend):
    from types import SimpleNamespace

    from rags.features import FeatureFlags
    from rags.model_discovery import tokenizer_selection
    from rags.tokenization import (
        ExactEncodingTokenizer,
        TiktokenTokenizer,
        UnicodeTokenizer,
        UTF8ByteTokenizer,
    )

    runtime = {"unicode": UnicodeTokenizer, "utf8_bytes": UTF8ByteTokenizer,
               "tiktoken": TiktokenTokenizer}.get(backend, ExactEncodingTokenizer)
    settings = SimpleNamespace(features=FeatureFlags(), tokenizer_artifacts={})
    config = {"backend": backend}
    if backend == "tiktoken":
        config["encoding"] = "cl100k_base"
    elif backend in {"huggingface", "sentencepiece"}:
        config["model"] = "registered-counter"
    selection = tokenizer_selection(config, settings)
    entries = [entry for entry in tokenizer_catalog() if entry["backend"] == backend]
    assert entries
    for entry in [selection, *entries]:
        assert entry["unit"] == runtime.unit
        assert entry["count_kind"] == runtime.count_kind
    assert selection["runtime_validation"] == "not_run"
