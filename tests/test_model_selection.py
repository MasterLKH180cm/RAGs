"""Offline model discovery, operator selection and environment-overlay contracts."""
from __future__ import annotations

import builtins
import json
import os
import stat
import subprocess
import sys

import pytest

from rags import cli, model_catalog, model_selection
from rags.config import Settings
from rags.features import FeatureFlags


@pytest.fixture(autouse=True)
def isolated_model_environment():
    prefixes = ("RAGS_", "OPENAI_", "ANTHROPIC_")
    original = {key: value for key, value in os.environ.items() if key.startswith(prefixes)}
    for key in original:
        os.environ.pop(key)
    try:
        yield
    finally:
        for key in tuple(os.environ):
            if key.startswith(prefixes):
                os.environ.pop(key)
        os.environ.update(original)


@pytest.mark.unit
@pytest.mark.parametrize("command", [
    ["models", "list", "--json"], ["models", "show", "bge-small-en", "--json"],
    ["tokenizers", "list", "--json"],
])
def test_discovery_is_offline_and_independent_of_credentials(command, monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RAGS_ENV", "production")
    monkeypatch.setenv("RAGS_KEYS_JSON", "invalid-on-purpose")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-printed")
    monkeypatch.setattr(Settings, "from_env", lambda: pytest.fail("Discovery must not load deployment settings"))
    monkeypatch.setattr(cli, "database_check", lambda *_: pytest.fail("Discovery must not inspect databases"))
    original = builtins.__import__

    def no_model_imports(name, *args, **kwargs):
        if name.split(".")[0] in {"fastembed", "sentence_transformers", "tokenizers", "sentencepiece", "tiktoken"}:
            pytest.fail("Discovery must not load tokenizer/encoder runtimes")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_model_imports)
    assert cli.main(command) == 0
    result = capsys.readouterr()
    assert json.loads(result.out)
    assert "must-not-be-printed" not in result.out + result.err
    assert list(tmp_path.iterdir()) == []


@pytest.mark.unit
def test_catalog_exposes_curated_models_with_requirements_and_limits():
    value = model_catalog.catalog()
    assert value["schema"] == "model-catalog-v1"
    models = value["embeddings"]
    assert len(models) == 22
    assert len({item["id"] for item in models}) == len(models)
    assert {item["provider"] for item in models} == set(model_selection.PROVIDERS)
    by_id = {item["id"]: item for item in models}
    assert by_id["offline"]["availability"] == "offline_demo"
    assert "not a trained" in by_id["offline"]["description"]
    assert by_id["bge-small-zh"]["languages"] == "zh"
    assert by_id["jina-code"]["languages"] == "code,en"
    assert by_id["st-bge-m3"]["extra"] == "sentence-transformers"
    assert "dense output only" in by_id["st-bge-m3"]["description"]
    assert by_id["openai-small"]["availability"] == "configuration_required"
    assert by_id["gateway"]["provider"] == "router"
    assert by_id["gateway"]["dimensions"] is None
    assert by_id["ollama-nomic"]["environment"]["RAGS_EMBEDDING_AUTH_MODE"] == "none"
    assert len(value["tokenizers"]) == 8
    assert {item["backend"] for item in value["tokenizers"]} == {
        "unicode", "utf8_bytes", "tiktoken", "huggingface", "sentencepiece"}


@pytest.mark.unit
def test_catalog_reports_operator_permissions_without_mutating_presets():
    flags = FeatureFlags((("embedding.fastembed", False), ("tokenizer.huggingface", False)))
    restricted = model_catalog.catalog(flags)
    assert all(not item["enabled"] for item in restricted["embeddings"] if item["provider"] == "fastembed")
    assert all(item["enabled"] for item in restricted["embeddings"] if item["provider"] != "fastembed")
    assert not next(item for item in restricted["tokenizers"] if item["id"] == "huggingface")["enabled"]
    restricted["tokenizers"][0]["config"]["backend"] = "changed"
    assert model_catalog.catalog()["tokenizers"][0]["config"] == {"backend": "unicode"}
    assert all("enabled" not in item for item in model_catalog.catalog()["embeddings"])


@pytest.mark.unit
@pytest.mark.parametrize("preset,query,document", [
    ("ollama-nomic", "search_query: ", "search_document: "),
    ("ollama-mxbai", "Represent this sentence for searching relevant passages: ", ""),
    ("ollama-embeddinggemma", "task: search result | query: ", "title: none | text: "),
    ("ollama-bge-m3", "", ""),
])
def test_ollama_presets_supply_the_models_retrieval_roles(preset, query, document):
    values = model_selection.selection_environment(_arguments("--embedding", preset))
    assert values["RAGS_EMBEDDING_QUERY_PREFIX"] == query
    assert values["RAGS_EMBEDDING_DOCUMENT_PREFIX"] == document


@pytest.mark.unit
@pytest.mark.parametrize("command,expected", [
    (["models", "list", "--provider", "openai", "--json"], {"openai-small", "openai-large", "openai-ada"}),
    (["models", "list", "--language", "code", "--json"], {"jina-code"}),
    (["models", "list", "--provider", "ollama", "--language", "multilingual", "--json"],
     {"ollama-bge-m3", "ollama-embeddinggemma"}),
])
def test_offline_model_filters(command, expected, capsys):
    assert cli.main(command) == 0
    assert {item["id"] for item in json.loads(capsys.readouterr().out)} == expected


@pytest.mark.unit
def test_discovery_human_output_and_unknown_preset(capsys):
    assert cli.main(["models", "list"]) == 0
    output = capsys.readouterr().out
    assert "PROVIDER" in output and "rags models show" in output
    assert cli.main(["tokenizers", "list"]) == 0
    output = capsys.readouterr().out
    assert "huggingface" in output and "tokenizer.json" in output and "sentencepiece" in output
    assert cli.main(["models", "show", "not-a-preset"]) == 2
    assert "rags models list" in capsys.readouterr().err


def _arguments(*args):
    return cli.build_parser().parse_args(["configure", *args])


@pytest.mark.unit
def test_custom_gateway_model_and_independent_tokenizer_settings():
    args = _arguments("--provider", "router", "--model", "approved-cohere-alias",
                      "--query-input-type", "search_query", "--document-input-type", "search_document",
                      "--tokenizer", "cl100k-base", "--generation-tokenizer", "o200k-base",
                      "--embedding-tokenizer", "utf8-bytes", "--embedding-max-tokens", "2048",
                      "--space-id", "cohere:deployment@v2")
    values = model_selection.selection_environment(args)
    assert values["RAGS_EMBEDDING_PROVIDER"] == "router"
    assert values["RAGS_EMBEDDING_MODEL"] == "approved-cohere-alias"
    assert values["RAGS_EMBEDDING_QUERY_INPUT_TYPE"] == "search_query"
    assert values["RAGS_EMBEDDING_DOCUMENT_INPUT_TYPE"] == "search_document"
    assert json.loads(values["RAGS_TOKENIZER_JSON"]) == {"backend": "tiktoken", "encoding": "cl100k_base"}
    assert json.loads(values["RAGS_GENERATION_TOKENIZER_JSON"]) == {"backend": "tiktoken", "encoding": "o200k_base"}
    assert json.loads(values["RAGS_EMBEDDING_TOKENIZER_JSON"]) == {"backend": "utf8_bytes"}
    assert values["RAGS_EMBEDDING_MAX_TOKENS"] == "2048"
    assert values["RAGS_EMBEDDING_SPACE_ID"] == "cohere:deployment@v2"


@pytest.mark.unit
def test_selection_resets_old_embedding_settings_without_overwriting_credentials():
    values = model_selection.selection_environment(_arguments("--embedding", "bge-small-en"))
    assert values["RAGS_EMBEDDING_REVISION"] == ""
    assert values["RAGS_EMBEDDING_DIMENSIONS"] == ""
    assert values["RAGS_EMBEDDING_QUERY_PREFIX"] == ""
    assert values["RAGS_EMBEDDING_TOKENIZER_JSON"] == ""
    assert values["RAGS_EMBEDDING_MAX_TOKENS"] == ""
    assert values["RAGS_EMBEDDING_AUTH_MODE"] == "required"
    assert set(values) <= model_selection.MODEL_ENV_KEYS
    assert not any("KEY" in key or "FEATURE" in key or "DATABASE" in key for key in values)


@pytest.mark.integration
@pytest.mark.parametrize("preset", [item.id for item in model_catalog.EMBEDDINGS])
def test_every_curated_overlay_roundtrips_into_validated_settings(preset, tmp_path):
    path = tmp_path / "selection.env"
    assert cli.main(["configure", "--embedding", preset, "--output", str(path)]) == 0
    base = tmp_path / "base.env"
    base.write_text("RAGS_ENV=test\n")
    cli.load_configuration(base, path)
    config = Settings.from_env()
    expected = model_catalog.embedding_preset(preset)
    assert config.embedding_provider == expected.provider
    assert config.embedding_model == expected.model
    assert config.embedding_query_prefix == expected.query_prefix
    assert config.embedding_document_prefix == expected.document_prefix
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert config.features == FeatureFlags()


@pytest.mark.integration
def test_roundtrip_preserves_literal_shell_text_and_never_copies_credentials(tmp_path, monkeypatch):
    marker = tmp_path / "must-not-execute"
    prefix = "query: '$HOME' `touch " + str(marker) + "` $(touch " + str(marker) + ") "
    monkeypatch.setenv("OPENAI_API_KEY", "never-copy-this-private-key")
    output = tmp_path / "models.env"
    assert cli.main(["configure", "--embedding", "st-minilm", "--query-prefix", prefix,
                     "--output", str(output)]) == 0
    body = output.read_text()
    assert "never-copy-this-private-key" not in body
    values = cli.read_env_file(output)
    assert values["RAGS_EMBEDDING_QUERY_PREFIX"] == prefix
    assert not marker.exists()
    assert "OPENAI_API_KEY" not in values


@pytest.mark.integration
def test_qwen_multiline_instruction_roundtrips_without_normalization(tmp_path):
    output = tmp_path / "qwen.env"
    assert cli.main(["configure", "--embedding", "qwen3-0.6b", "--output", str(output)]) == 0
    expected = model_catalog.embedding_preset("qwen3-0.6b").query_prefix
    assert "\nQuery: " in expected
    values = cli.read_env_file(output)
    assert values["RAGS_EMBEDDING_QUERY_PREFIX"] == expected
    base = tmp_path / "base.env"
    base.write_text("RAGS_ENV=test\n")
    cli.load_configuration(base, output)
    settings = Settings.from_env()
    assert settings.embedding_query_prefix == expected
    assert settings.embedding_document_prefix == ""


@pytest.mark.integration
@pytest.mark.parametrize("separator", ["\u2028", "\u2029", "\x85", "\v", "\f"])
def test_unicode_prefix_separators_roundtrip_without_becoming_newlines(separator, tmp_path):
    prefix = "查詢" + separator + "Query: "
    output = tmp_path / "unicode-prefix.env"
    assert cli.main(["configure", "--embedding", "st-minilm", "--query-prefix", prefix,
                     "--output", str(output)]) == 0
    assert cli.read_env_file(output)["RAGS_EMBEDDING_QUERY_PREFIX"] == prefix
    base = tmp_path / "base.env"
    base.write_text("RAGS_ENV=test\n")
    cli.load_configuration(base, output)
    assert Settings.from_env().embedding_query_prefix == prefix


@pytest.mark.integration
def test_multiline_prefix_assignment_and_comment_lines_remain_literal_text(tmp_path):
    prefix = ("Instruct: retrieve supporting evidence\n"
              "RAGS_AUTH_MODE=oauth2_introspection\n"
              "RAGS_FEATURES_JSON='{}'\n"
              "OPENAI_API_KEY=must-stay-literal\n"
              "# This is source prefix text, not an environment comment\n"
              "Query: ")
    output = tmp_path / "multiline.env"
    assert cli.main(["configure", "--embedding", "st-minilm", "--query-prefix", prefix,
                     "--output", str(output)]) == 0
    values = cli.read_env_file(output)
    assert values["RAGS_EMBEDDING_QUERY_PREFIX"] == prefix
    assert not {"RAGS_AUTH_MODE", "RAGS_FEATURES_JSON", "OPENAI_API_KEY"} & values.keys()
    base = tmp_path / "base.env"
    base.write_text("RAGS_ENV=test\n")
    cli.load_configuration(base, output)
    settings = Settings.from_env()
    assert settings.embedding_query_prefix == prefix
    assert settings.auth_mode == "api_key" and settings.openai_key == ""
    assert settings.features == FeatureFlags()


@pytest.mark.unit
def test_unterminated_multiline_assignment_is_rejected_atomically(tmp_path):
    base = tmp_path / "base.env"
    base.write_text("RAGS_ENV=test\nOPENAI_API_KEY=base-secret\n")
    output = tmp_path / "incomplete.env"
    output.write_text("RAGS_EMBEDDING_PROVIDER=fastembed\n"
                      "RAGS_EMBEDDING_QUERY_PREFIX='Instruct: retrieve evidence\n"
                      "# literal unfinished instruction\nQuery: \n")
    with pytest.raises(ValueError, match="shell quoting"):
        cli.load_configuration(base, output)
    assert "RAGS_ENV" not in os.environ
    assert "OPENAI_API_KEY" not in os.environ
    assert "RAGS_EMBEDDING_PROVIDER" not in os.environ


@pytest.mark.integration
def test_local_artifact_selection_roundtrips_real_tokenizer(tmp_path):
    from tokenizers import Tokenizer, models

    from rags.tokenization import resolve_tokenizer

    artifact = tmp_path / "tokenizer.json"
    Tokenizer(models.WordLevel({"[UNK]": 0, "hello": 1}, unk_token="[UNK]")).save(str(artifact))
    output = tmp_path / "models.env"
    assert cli.main(["configure", "--embedding", "offline", "--tokenizer", "huggingface",
                     "--tokenizer-path", str(artifact), "--generation-tokenizer", "utf8-bytes",
                     "--output", str(output)]) == 0
    base = tmp_path / "base.env"
    base.write_text("RAGS_ENV=test\n")
    cli.load_configuration(base, output)
    settings = Settings.from_env()
    assert settings.tokenizer_config == {"backend": "huggingface", "path": str(artifact.absolute())}
    assert settings.generation_tokenizer_config == {"backend": "utf8_bytes"}
    assert resolve_tokenizer(settings.tokenizer_config).encode("hello") == [1]


@pytest.mark.unit
@pytest.mark.parametrize("options", [
    ["--embedding", "offline", "--model", "arbitrary"],
    ["--provider", "openai"], ["--model", "missing-provider"],
    ["--embedding", "not-a-preset"],
    ["--embedding", "offline", "--dimensions", "10"],
    ["--embedding", "openai-small", "--dimensions", "0"],
    ["--embedding", "openai-small", "--dimensions", "16385"],
    ["--embedding", "bge-small-en", "--revision", "abc123"],
    ["--embedding", "openai-small", "--auth-mode", "none"],
    ["--embedding", "openai-small", "--endpoint", "https://example.test", "--auth-mode", "none"],
    ["--embedding", "openai-small", "--endpoint", "https://user:password@example.test"],
    ["--embedding", "openai-small", "--endpoint", "https://example.test?token=private"],
    ["--embedding", "bge-small-en", "--auth-mode", "none"],
    ["--embedding", "offline", "--embedding-tokenizer", "utf8-bytes"],
    ["--embedding", "offline", "--embedding-max-tokens", "1000"],
    ["--embedding", "offline", "--embedding-tokenizer", "unicode", "--embedding-max-tokens", "1000"],
    ["--embedding", "offline", "--embedding-tokenizer", "utf8-bytes", "--embedding-max-tokens", "0"],
    ["--embedding", "offline", "--embedding-tokenizer", "utf8-bytes", "--embedding-max-tokens", "100001"],
    ["--embedding", "offline", "--tokenizer", "huggingface"],
    ["--embedding", "offline", "--tokenizer-path", "unused.json"],
    ["--embedding", "offline", "--tokenizer", "unicode", "--tokenizer-path", "unused.json"],
    ["--embedding", "offline", "--query-prefix", "bad\rline"],
    ["--embedding", "offline", "--query-prefix", "bad\0line"],
    ["--provider", "openai", "--model", ""],
    ["--provider", "openai", "--model", "x" * 161],
])
def test_invalid_selection_has_no_file_side_effect(options, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(model_selection.sys.stdin, "isatty", lambda: False)
    output = tmp_path / "models.env"
    assert cli.main(["configure", *options, "--output", str(output)]) == 2
    assert not output.exists()
    assert capsys.readouterr().err.startswith("rags:")


@pytest.mark.unit
@pytest.mark.parametrize("options", [
    ["--embedding", "bge-small-en", "--query-input-type", "search_query"],
    ["--embedding", "st-minilm", "--document-input-type", "search_document"],
    ["--embedding", "ollama-nomic", "--query-input-type", "search_query"],
    ["--embedding", "st-minilm", "--revision", "bad revision"],
    ["--embedding", "offline", "--space-id", "bad space"],
    ["--embedding", "st-minilm", "--query-prefix", "x" * 1025],
])
def test_configure_rejects_settings_invalid_values_before_writing(options, tmp_path, capsys):
    output = tmp_path / "models.env"
    assert cli.main(["configure", *options, "--output", str(output)]) == 2
    assert not output.exists()
    assert capsys.readouterr().err.startswith("rags:")


@pytest.mark.unit
@pytest.mark.parametrize("options", [
    ["--embedding", "private-configuration-marker"],
    ["--embedding", "openai-small", "--endpoint", "https://private-configuration-marker@example.test"],
    ["--embedding", "offline", "--tokenizer", "huggingface", "--tokenizer-path", "/private-configuration-marker/missing.json"],
    ["--embedding", "offline", "--query-prefix", "private-configuration-marker\rline"],
])
def test_selection_errors_do_not_echo_sensitive_invalid_values(options, tmp_path, capsys):
    output = tmp_path / "models.env"
    assert cli.main(["configure", *options, "--output", str(output)]) == 2
    result = capsys.readouterr()
    assert "private-configuration-marker" not in result.out + result.err
    assert not output.exists()


@pytest.mark.integration
def test_selection_does_not_clobber_existing_file_or_symlink(tmp_path, capsys):
    existing = tmp_path / "keep.env"
    existing.write_text("keep original")
    link = tmp_path / "linked.env"
    link.symlink_to(existing)
    for output in (existing, link):
        assert cli.main(["configure", "--embedding", "offline", "--output", str(output)]) == 2
        assert "already exists" in capsys.readouterr().err
        assert existing.read_text() == "keep original"
    assert link.is_symlink()


@pytest.mark.integration
def test_selection_rejects_symlink_directory(tmp_path, capsys):
    destination = tmp_path / "actual"
    destination.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(destination, target_is_directory=True)
    assert cli.main(["configure", "--embedding", "offline", "--output", str(alias / "models.env")]) == 2
    assert "directory" in capsys.readouterr().err
    assert not (destination / "models.env").exists()


@pytest.mark.unit
@pytest.mark.parametrize("name", ["RAGS_KEYS_JSON", "RAGS_API_KEY", "RAGS_AUTH_MODE", "RAGS_KEY_ROLES_JSON",
                                 "RAGS_FEATURES_JSON", "RAGS_FEATURES_FILE", "RAGS_DATABASE_URL", "RAGS_DB_PATH",
                                 "OPENAI_API_KEY", "RAGS_EMBEDDING_API_KEY", "RAGS_EMBEDDING_API_KEY_FILE"])
def test_model_overlay_cannot_inject_auth_features_database_or_secrets(name, tmp_path):
    base = tmp_path / "base.env"
    base.write_text("RAGS_ENV=test\n")
    model = tmp_path / "model.env"
    model.write_text(name + "='injected'\n")
    with pytest.raises(ValueError, match="only model and tokenizer"):
        cli.load_configuration(base, model)
    assert "RAGS_ENV" not in os.environ
    assert name not in os.environ
    with pytest.raises(ValueError, match="only model and tokenizer"):
        model_selection.write_selection(tmp_path / "output.env", {name: "injected"})
    assert not (tmp_path / "output.env").exists()


@pytest.mark.integration
def test_overlay_precedence_preserves_process_environment_and_operator_policy(tmp_path, monkeypatch):
    base = tmp_path / "base.env"
    base.write_text("RAGS_ENV=test\nRAGS_EMBEDDING_PROVIDER=local\nRAGS_EMBEDDING_MODEL=from-base\n"
                    "RAGS_EMBEDDING_DIMENSIONS=128\nRAGS_EMBEDDING_QUERY_PREFIX='old prefix'\n"
                    "RAGS_FEATURES_JSON='{" + '"embedding.fastembed":false' + "}'\n"
                    "OPENAI_API_KEY=base-secret\n")
    overlay = tmp_path / "models.env"
    assert cli.main(["configure", "--embedding", "bge-small-en", "--output", str(overlay)]) == 0
    monkeypatch.setenv("RAGS_EMBEDDING_MODEL", "operator-selected-model")
    monkeypatch.setenv("OPENAI_API_KEY", "environment-secret")
    cli.load_configuration(base, overlay)
    settings = Settings.from_env()
    assert settings.embedding_provider == "fastembed"
    assert settings.fastembed_model == "operator-selected-model"
    assert settings.embedding_dimensions is None
    assert settings.embedding_query_prefix == ""
    assert not settings.features.enabled("embedding.fastembed")
    assert settings.openai_key == "environment-secret"
    assert os.environ["RAGS_EMBEDDING_MODEL"] == "operator-selected-model"


@pytest.mark.unit
@pytest.mark.parametrize("malformed", [
    "RAGS_EMBEDDING_PROVIDER=local\nRAGS_EMBEDDING_MODEL='unterminated\n",
    "RAGS_EMBEDDING_PROVIDER=local\nRAGS_EMBEDDING_PROVIDER=openai\n",
    "RAGS_EMBEDDING_PROVIDER=local\nBAD NAME=value\n",
    "RAGS_EMBEDDING_PROVIDER=local\nRAGS_EMBEDDING_MODEL=two values\n",
])
def test_both_files_are_parsed_before_environment_changes(tmp_path, malformed):
    base = tmp_path / "base.env"
    base.write_text("RAGS_ENV=test\nOPENAI_API_KEY=base-secret\n")
    overlay = tmp_path / "models.env"
    overlay.write_text(malformed)
    with pytest.raises(ValueError):
        cli.load_configuration(base, overlay)
    assert "RAGS_ENV" not in os.environ
    assert "OPENAI_API_KEY" not in os.environ
    assert "RAGS_EMBEDDING_PROVIDER" not in os.environ


@pytest.mark.unit
@pytest.mark.parametrize("location", ["single", "base", "overlay"])
def test_manual_nul_values_are_rejected_before_any_environment_mutation(tmp_path, location):
    base = tmp_path / "base.env"
    overlay = tmp_path / "models.env"
    base.write_text("RAGS_ENV=test\nOPENAI_API_KEY=base-secret\n")
    overlay.write_text("RAGS_EMBEDDING_PROVIDER=openai\nRAGS_EMBEDDING_MODEL=custom-model\n")
    target = overlay if location == "overlay" else base
    with target.open("a", encoding="utf-8") as stream:
        stream.write("RAGS_EMBEDDING_QUERY_PREFIX='literal\0value'\n")
    before = dict(os.environ)
    with pytest.raises(ValueError, match="NUL"):
        if location == "single":
            cli.load_env_file(base)
        else:
            cli.load_configuration(base, overlay)
    assert dict(os.environ) == before


@pytest.mark.unit
def test_interactive_wizard_has_scriptable_noninteractive_error(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(model_selection.sys.stdin, "isatty", lambda: False)
    output = tmp_path / "models.env"
    assert cli.main(["configure", "--output", str(output)]) == 2
    error = capsys.readouterr().err
    assert "--embedding" in error and "terminal" in error
    assert not output.exists()


@pytest.mark.integration
def test_interactive_wizard_defaults_and_cancel(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(model_selection.sys.stdin, "isatty", lambda: True)
    choices = iter(["", ""])
    monkeypatch.setattr(builtins, "input", lambda _: next(choices))
    output = tmp_path / "models.env"
    assert cli.main(["configure", "--output", str(output)]) == 0
    values = cli.read_env_file(output)
    assert values["RAGS_EMBEDDING_MODEL"] == "BAAI/bge-small-en-v1.5"
    assert json.loads(values["RAGS_TOKENIZER_JSON"]) == {"backend": "unicode"}
    assert "Reindex" in capsys.readouterr().out

    def cancelled(_):
        raise EOFError

    monkeypatch.setattr(builtins, "input", cancelled)
    cancelled_path = tmp_path / "cancelled.env"
    assert cli.main(["configure", "--output", str(cancelled_path)]) == 2
    assert "cancelled" in capsys.readouterr().err
    assert not cancelled_path.exists()


@pytest.mark.integration
def test_interactive_wizard_under_real_terminal(tmp_path):
    import pty

    output = tmp_path / "interactive.env"
    master, slave = pty.openpty()
    try:
        with subprocess.Popen([sys.executable, "-m", "rags.cli", "configure", "--output", str(output)],
                              stdin=slave, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as process:
            os.close(slave)
            slave = None
            os.write(master, b"offline\nunicode\n")
            stdout, stderr = process.communicate(timeout=15)
            assert process.returncode == 0, stderr
            assert "Embedding model" in stdout and "Chunking tokenizer" in stdout
            assert "Created" in stdout
    finally:
        os.close(master)
        if slave is not None:
            os.close(slave)
    values = cli.read_env_file(output)
    assert values["RAGS_EMBEDDING_MODEL"] == "offline-hash-v1"
    assert json.loads(values["RAGS_TOKENIZER_JSON"]) == {"backend": "unicode"}
    assert stat.S_IMODE(output.stat().st_mode) == 0o600


@pytest.mark.unit
def test_wizard_rejects_unknown_selection_without_writing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(model_selection.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(builtins, "input", lambda _: "9999")
    output = tmp_path / "models.env"
    assert cli.main(["configure", "--output", str(output)]) == 2
    assert "listed number or ID" in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.integration
def test_doctor_and_serve_parse_model_selection_overlay(tmp_path, monkeypatch, capsys):
    base = tmp_path / "base.env"
    base.write_text("RAGS_ENV=test\nRAGS_EMBEDDING_PROVIDER=local\n")
    selection = tmp_path / "selection.env"
    selection.write_text("RAGS_EMBEDDING_PROVIDER=ollama\nRAGS_EMBEDDING_MODEL=nomic-embed-text\n"
                         "RAGS_EMBEDDING_BASE_URL=http://127.0.0.1:11434\nRAGS_EMBEDDING_AUTH_MODE=none\n")
    assert cli.main(["doctor", "--env-file", str(base), "--model-config", str(selection), "--no-db", "--json"]) == 0
    checks = json.loads(capsys.readouterr().out)["checks"]
    embedding = next(check for check in checks if check["name"] == "embedding")
    assert embedding["status"] == "ok" and "nomic-embed-text" in embedding["detail"]
    observed = []
    monkeypatch.setattr(cli, "_run_serve", lambda args: observed.append(Settings.from_env()) or 0)
    assert cli.main(["serve", "--env-file", str(base), "--model-config", str(selection)]) == 0
    assert observed[0].embedding_provider == "ollama"
    assert observed[0].embedding_auth_mode == "none"


@pytest.mark.unit
@pytest.mark.parametrize("credential,expected", [("", "warning"), ("REPLACE_ME", "warning"),
                                               ("private-ollama-key", "ok")])
def test_doctor_recognizes_authenticated_ollama_without_disclosing_credentials(monkeypatch, credential, expected):
    monkeypatch.setenv("RAGS_ENV", "test")
    monkeypatch.setenv("RAGS_EMBEDDING_PROVIDER", "ollama")
    monkeypatch.setenv("RAGS_EMBEDDING_MODEL", "nomic-embed-text")
    monkeypatch.setenv("RAGS_EMBEDDING_BASE_URL", "https://ollama.example")
    monkeypatch.setenv("RAGS_EMBEDDING_AUTH_MODE", "required")
    monkeypatch.setenv("RAGS_EMBEDDING_API_KEY", credential)
    monkeypatch.setattr(cli, "database_check", lambda *_: pytest.fail("No database access expected"))
    report = cli.diagnose(check_db=False)
    embedding = next(check for check in report["checks"] if check["name"] == "embedding")
    assert embedding["status"] == expected
    assert ("credentials missing" in embedding["detail"]) == (expected == "warning")
    assert "private-ollama-key" not in json.dumps(report)
