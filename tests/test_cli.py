"""CLI, installation profile and diagnostic contracts with no live side effects."""
import json
import os
import sqlite3
import stat

import pytest

from rags import cli
from rags.config import Settings
from rags.features import FeatureFlags


pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def clean_rag_env():
    """The CLI writes process env without monkeypatch, so restore it explicitly."""
    prefixes = ("RAGS_", "OPENAI_", "ANTHROPIC_")
    original = {key: val for key, val in os.environ.items() if key.startswith(prefixes)}
    for key in original:
        os.environ.pop(key)
    try:
        yield
    finally:
        for key in tuple(os.environ):
            if key.startswith(prefixes):
                os.environ.pop(key)
        os.environ.update(original)


def test_cli_parser_exposes_installable_commands():
    parser = cli.build_parser()
    for command in ("init", "doctor", "serve", "benchmark"):
        assert parser.parse_args([command]).command == command


@pytest.mark.parametrize("profile", cli.PROFILES)
def test_init_profile_creates_private_credentials_and_feature_flags(tmp_path, profile, monkeypatch):
    env_path, features_path = cli.init_config(tmp_path, profile)
    assert stat.S_IMODE(env_path.stat().st_mode) == 0o600
    assert json.loads(features_path.read_text()) == {
        name: False for name in cli.PROFILE_DISABLED[profile]
    }
    assert stat.S_IMODE(features_path.stat().st_mode) == 0o644
    cli.load_env_file(env_path)
    config = Settings.from_env()
    token = os.environ["RAGS_API_KEY"]
    assert len(token) >= 40 and config.keys == {token: "local"}
    assert config.features == FeatureFlags.from_env()
    assert config.features.enabled("query")
    assert config.features.enabled("agents")
    assert config.features.enabled("model_generation")
    from rags.agents.config import AgentPlatformConfig
    agent = AgentPlatformConfig.from_env()
    assert agent.router_enabled and agent.agent_enabled and agent.memory_enabled
    assert agent.router_ready is (profile == "inkling-local")
    assert config.db_path.startswith(("postgresql://", "postgres://")) == (profile == "postgres")
    assert ("REPLACE_ME" in config.db_path) == (profile == "postgres")
    assert config.embedding_provider == ("openai" if profile in {"openai", "ollama"} else "local")
    with pytest.raises(FileExistsError):
        cli.init_config(tmp_path, profile)


def test_init_rejects_symlink_and_preserves_existing_files(tmp_path):
    other = tmp_path / "destination"
    other.write_text("keep")
    (tmp_path / ".env").symlink_to(other)
    with pytest.raises(FileExistsError):
        cli.init_config(tmp_path, "offline")
    assert other.read_text() == "keep" and not (tmp_path / "rags.features.json").exists()


def test_init_rolls_back_new_secret_when_flag_write_fails(tmp_path, monkeypatch):
    original = cli.os.open

    def fail_second(path, flags, mode):
        if str(path).endswith("rags.features.json"):
            raise OSError("injected failure")
        return original(path, flags, mode)

    monkeypatch.setattr(cli.os, "open", fail_second)
    with pytest.raises(OSError, match="injected failure"):
        cli.init_config(tmp_path, "offline")
    assert not (tmp_path / ".env").exists()
    assert not (tmp_path / "rags.features.json").exists()


@pytest.mark.parametrize("body", [
    "BAD NAME=some-value\n", "RAGS_ENV=dev\nRAGS_ENV=test\n", "no_equals\n",
    "RAGS_ENV='unclosed\n", "RAGS_ENV=foo bar\n",
])
def test_env_loader_rejects_invalid_names_repeated_values_and_shell_commands(tmp_path, body):
    source = tmp_path / "input.env"
    source.write_text(body)
    with pytest.raises(ValueError):
        cli.load_env_file(source)


def test_env_loader_does_not_execute_shell_or_override_existing_secrets(tmp_path, monkeypatch):
    source = tmp_path / "profile.env"
    source.write_text("RAGS_ENV=development\nRAGS_API_KEY='from-file'\n"
                      "RAGS_MODEL_PROVIDER='$(touch /tmp/must-not-execute)'\n")
    monkeypatch.setenv("RAGS_API_KEY", "external-secret")
    cli.load_env_file(source)
    assert os.environ["RAGS_API_KEY"] == "external-secret"
    assert os.environ["RAGS_MODEL_PROVIDER"] == "$(touch /tmp/must-not-execute)"


def test_doctor_is_nonsecret_and_no_db_is_network_free(tmp_path, capsys, monkeypatch):
    env_path, _ = cli.init_config(tmp_path, "offline")
    cli.load_env_file(env_path)
    monkeypatch.setattr(cli, "database_check", lambda *_: pytest.fail("No DB checks permitted"))
    assert cli.main(["doctor", "--no-db", "--json"]) == 0
    output = capsys.readouterr().out
    assert json.loads(output)["status"] == "ok"
    assert os.environ["RAGS_API_KEY"] not in output
    assert "RAGS_KEYS_JSON" not in output


def test_doctor_fails_closed_and_does_not_serialize_raw_invalid_values(monkeypatch):
    secret = "user-private-invalid-provider"
    monkeypatch.setenv("RAGS_ENV", "production")
    monkeypatch.setenv("RAGS_KEYS_JSON", secret)
    response = cli.diagnose()
    assert response["status"] == "error"
    assert secret not in json.dumps(response)


def test_doctor_read_only_sqlite_missing_and_existing(tmp_path, monkeypatch):
    path = tmp_path / "test.db"
    env_path, _ = cli.init_config(tmp_path, "offline")
    cli.load_env_file(env_path)
    monkeypatch.setenv("RAGS_DB_PATH", str(path))
    settings = Settings.from_env()
    assert cli.database_check(settings)["status"] == "warning"
    assert not path.exists()
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE documents(id text)")
        conn.execute("CREATE TABLE chunks(id text)")
    assert cli.database_check(settings)["status"] == "ok"
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT count(*) FROM sqlite_master WHERE type='table'").fetchone()[0] == 2


def test_doctor_postgres_placeholder_does_not_attempt_connection(tmp_path):
    env_path, _ = cli.init_config(tmp_path, "postgres")
    cli.load_env_file(env_path)
    assert cli.database_check(Settings.from_env())["status"] == "warning"


def test_serve_rejects_nonloopback_without_explicit_optin(tmp_path):
    env_path, _ = cli.init_config(tmp_path, "offline")
    assert cli.main(["serve", "--env-file", str(env_path), "--host", "0.0.0.0"]) == 1


def test_benchmark_parameter_guard_does_not_access_postgres():
    from rags.benchmark import benchmark
    for parameters in [
        {"documents": 9}, {"documents": 2001}, {"chunks_per_document": 0},
        {"chunks_per_document": 17}, {"documents": 2000, "chunks_per_document": 16},
        {"iterations": 2}, {"iterations": 101}
    ]:
        with pytest.raises(ValueError):
            benchmark("postgresql://invalid", **parameters)
    with pytest.raises(ValueError, match="PostgreSQL"):
        benchmark("local.db")


@pytest.mark.parametrize("profile", ("offline", "openai", "ollama", "anthropic"))
def test_all_nonpostgres_profiles_have_consistent_model_settings(tmp_path, profile):
    cli.load_env_file(cli.init_config(tmp_path, profile)[0])
    config = Settings.from_env()
    assert config.keys and config.db_path.endswith("rags.db")
    assert config.model_provider == ("extractive" if profile == "offline" else
                                     "anthropic" if profile == "anthropic" else "openai")
    assert config.features.enabled("query")

def test_malformed_environment_file_does_not_partially_modify_process(tmp_path):
    source = tmp_path / "partial.env"
    source.write_text("RAGS_ENV=development\nDUP=ok\nDUP=again\n")
    with pytest.raises(ValueError):
        cli.load_env_file(source)
    assert "RAGS_ENV" not in os.environ and "DUP" not in os.environ

def test_agent_workflow_cli_options():
    parser = cli.build_parser()
    assert parser.parse_args(['workflow', 'list']).workflow_command == 'list'
    assert parser.parse_args(['workflow', 'run', '--question-file', 'goal.txt']).workflow_command == 'run'
    assert parser.parse_args(['workflow', 'review', '--run-id', '123', '--revision', '1', '--approve']).approve
    assert parser.parse_args(['workflow', 'recover', '--run-id', '123', '--revision', '2']).revision == 2
