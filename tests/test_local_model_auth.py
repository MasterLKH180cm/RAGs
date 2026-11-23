"""Explicit no-key local inference through production adapters; no model claims."""
from contextlib import contextmanager
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import re
from threading import Thread

from fastapi.testclient import TestClient
import pytest

from rags import cli
from rags.agents.config import AgentPlatformConfig
from rags.agents.router import ModelRouter
from rags.api import create_app
from rags.config import Settings
from rags.model_profiles import INKLING_MODEL, TINKER_BASE_URL
from rags.providers import Providers
from rags.registry import CapabilityError
from rags.vision import describe_image


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    original = {name: value for name, value in os.environ.items()
                if name.startswith(("RAGS_", "OPENAI_", "ANTHROPIC_", "TINKER_"))}
    for name in tuple(os.environ):
        if name.startswith(("RAGS_", "OPENAI_", "ANTHROPIC_", "TINKER_")):
            monkeypatch.delenv(name)
    monkeypatch.setenv("RAGS_ENV", "test")
    yield
    # CLI profile loading writes process env directly; remove its additions too.
    for name in tuple(os.environ):
        if name.startswith(("RAGS_", "OPENAI_", "ANTHROPIC_", "TINKER_")):
            os.environ.pop(name)
    os.environ.update(original)


def local_settings(**changes):
    values = {"model_provider": "openai", "chat_model": INKLING_MODEL,
              "generation_base_url": "http://127.0.0.1:8080/v1", "generation_protocol": "openai",
              "generation_auth_mode": "none", "generation_tokenizer_config": {"backend": "utf8_bytes"}}
    values.update(changes)
    return replace(Settings.from_env(), **values)


def local_router_config(**changes):
    values = {"gateway_url": "http://127.0.0.1:8080/v1", "gateway_auth_mode": "none",
              "allow_http": True, "models": {"inkling": INKLING_MODEL},
              "model_options": {"inkling": {"protocol": "openai"}}}
    values.update(changes)
    return AgentPlatformConfig(**values)


def local_environment(monkeypatch):
    for name, value in {
        "RAGS_MODEL_PROVIDER": "openai", "RAGS_GENERATION_MODEL": INKLING_MODEL,
        "RAGS_GENERATION_BASE_URL": "http://127.0.0.1:8080/v1",
        "RAGS_GENERATION_AUTH_MODE": "none", "RAGS_GENERATION_PROTOCOL": "openai",
        "RAGS_GENERATION_TOKENIZER_JSON": '{"backend":"utf8_bytes"}',
        "RAGS_ROUTER_GATEWAY_URL": "http://127.0.0.1:8080/v1", "RAGS_ROUTER_AUTH_MODE": "none",
        "RAGS_ROUTER_ALLOW_HTTP": "true", "RAGS_ROUTER_MODELS_JSON": json.dumps({"inkling": INKLING_MODEL}),
        "RAGS_ROUTER_MODEL_OPTIONS_JSON": '{"inkling":{"protocol":"openai"}}',
    }.items():
        monkeypatch.setenv(name, value)


@pytest.mark.unit
class TestLocalAuthentication:
    @pytest.mark.parametrize("endpoint", [
        "", "http://localhost:8080/v1", "http://model:8080/v1", "http://host.docker.internal:8080/v1",
        "http://192.168.1.1:8080/v1", "https://example.com/v1", TINKER_BASE_URL,
        "http://0.0.0.0:8080/v1", "http://127.0.0.1.evil.invalid/v1", "http://127.1:8080/v1",
        "http://2130706433/v1", "http://[::ffff:127.0.0.1]/v1", "http://[::1%25eth0]/v1",
        "http://user:secret@127.0.0.1/v1", "http://@127.0.0.1/v1",
        "http://127.0.0.1/v1?", "http://127.0.0.1/v1#fragment", "file:///tmp/model",
        "http://127.0.0.1:0/v1", "http://127.0.0.1:65536/v1", " http://127.0.0.1/v1",
    ])
    def test_unsafe_no_auth_endpoints_fail_in_both_production_configs(self, endpoint):
        with pytest.raises(ValueError):
            local_settings(generation_base_url=endpoint)
        with pytest.raises(ValueError):
            local_router_config(gateway_url=endpoint)

    @pytest.mark.parametrize("endpoint", ["http://127.0.0.1:8080/v1", "http://[::1]:8080/v1",
                                          "https://127.0.0.1/v1"])
    def test_literal_loopback_is_valid_without_a_placeholder_key(self, endpoint):
        config = local_settings(generation_base_url=endpoint)
        assert config.generation_credential == "" and Providers(config).has_generation
        router = local_router_config(gateway_url=endpoint)
        assert router.router_ready and router.gateway_key == ""

    @pytest.mark.parametrize("mode", [None, True, "", "optional", "NONE"])
    def test_invalid_auth_modes_are_not_coerced(self, mode):
        with pytest.raises(ValueError):
            local_settings(generation_auth_mode=mode)
        with pytest.raises(ValueError):
            local_router_config(gateway_auth_mode=mode)

    @pytest.mark.parametrize("protocol", ["auto", "tinker"])
    def test_no_auth_does_not_implicitly_choose_tinker_fields(self, protocol):
        with pytest.raises(ValueError):
            local_settings(generation_protocol=protocol)
        with pytest.raises(ValueError):
            local_router_config(model_options={"inkling": {"protocol": protocol}})

    def test_no_auth_rejects_reasoning_controls_missing_alias_policy_and_plaintext_without_optin(self):
        with pytest.raises(ValueError):
            local_settings(reasoning_effort="low")
        with pytest.raises(ValueError):
            local_router_config(model_options={})
        with pytest.raises(ValueError):
            local_router_config(model_options={"inkling": {"protocol": "openai", "reasoning_effort": "low"}})
        with pytest.raises(ValueError):
            local_router_config(allow_http=False)
        with pytest.raises(ValueError):
            local_settings(model_provider="anthropic")

    @pytest.mark.parametrize("field", ["RAGS_GENERATION_API_KEY", "RAGS_GENERATION_API_KEY_FILE",
                                       "RAGS_ROUTER_API_KEY", "RAGS_ROUTER_API_KEY_FILE"])
    def test_explicit_credential_conflict_rejects_before_secret_file_access(self, monkeypatch, field):
        local_environment(monkeypatch)
        monkeypatch.setenv(field, "/missing/secret-file" if field.endswith("_FILE") else "contradictory-key")
        operation = Settings.from_env if "GENERATION" in field else AgentPlatformConfig.from_env
        with pytest.raises(ValueError, match="cannot configure"):
            operation()

    def test_programmatic_no_auth_config_rejects_credentials(self):
        with pytest.raises(ValueError, match="cannot configure"):
            local_settings(generation_api_key="explicit-key")
        with pytest.raises(ValueError, match="cannot configure"):
            local_router_config(gateway_key="explicit-key")

    def test_no_auth_does_not_read_ambient_hosted_secrets_or_their_files(self, monkeypatch):
        local_environment(monkeypatch)
        for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "TINKER_API_KEY", "RAGS_EMBEDDING_API_KEY"):
            monkeypatch.setenv(name, "ambient-key-that-must-not-be-read")
            monkeypatch.setenv(name + "_FILE", "/missing/unused-secret-file")
        settings = Settings.from_env()
        assert settings.generation_credential == settings.openai_key == settings.anthropic_key == ""
        assert settings.embedding_api_key is None and Providers(settings).has_generation
        assert AgentPlatformConfig.from_env().router_ready

    @pytest.mark.parametrize("dedicated", [True, False])
    def test_independent_embedding_provider_retains_its_credential(self, monkeypatch, dedicated):
        local_environment(monkeypatch)
        monkeypatch.setenv("RAGS_EMBEDDING_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://embedding-fixture.invalid/v1")
        if dedicated:
            monkeypatch.setenv("RAGS_EMBEDDING_API_KEY", "embedding-key")
            monkeypatch.setenv("OPENAI_API_KEY_FILE", "/missing/unused-legacy-secret")
        else:
            monkeypatch.setenv("OPENAI_API_KEY", "embedding-key")
        settings = Settings.from_env()
        assert settings.embedding_credential == "embedding-key"
        assert settings.embedding_endpoint == "https://embedding-fixture.invalid/v1"
        assert settings.generation_credential == ""

    def test_hosted_defaults_still_require_credentials(self):
        settings = local_settings(generation_auth_mode="required")
        assert not Providers(settings).has_generation
        with pytest.raises(ValueError, match="credentials"):
            local_router_config(gateway_auth_mode="required")

    def test_local_profile_configures_both_interfaces_and_doctor_performs_no_network(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr("rags.provider_transport.post", lambda *a, **kw: pytest.fail("Doctor made a model request"))
        env, _ = cli.init_config(tmp_path, "inkling-local")
        assert cli.main(["doctor", "--env-file", str(env), "--no-db", "--json"]) == 0
        report = json.loads(capsys.readouterr().out)
        assert report["status"] == "ok"
        assert Settings.from_env().context_budget == 4096
        assert "auth=none" in next(row["detail"] for row in report["checks"] if row["name"] == "generation")
        assert cli.main(["agent", "doctor", "--env-file", str(env), "--json"]) == 0
        agent = json.loads(capsys.readouterr().out)
        assert agent["gateway_configured"] and agent["gateway_auth_mode"] == "none"


@contextmanager
def local_model_fixture():
    calls = []
    state = {"finish_reason": "stop"}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append({"body": payload, "authorization": self.headers.get("Authorization")})
            if payload.get("tools"):
                message = {"content": None, "reasoning_content": "private continuation fixture",
                           "tool_calls": [{"id": "lookup_1", "type": "function", "function": {
                               "name": "lookup", "arguments": '{"query":"retention"}'}}]}
                reason = "tool_calls" if state["finish_reason"] == "stop" else state["finish_reason"]
            else:
                identifiers = re.findall(r"\[([a-f0-9]{32})\]", json.dumps(payload))
                citation = " [" + identifiers[0] + "]" if identifiers else ""
                message = {"content": "Local fixture answer" + citation}
                reason = state["finish_reason"]
            raw = json.dumps({"choices": [{"finish_reason": reason, "message": message}],
                              "usage": {"prompt_tokens": 8, "completion_tokens": 4}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", calls, state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.integration
class TestActualLocalModelTransport:
    def test_direct_router_and_vision_ignore_ambient_proxies_and_send_no_authorization(self, monkeypatch):
        for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
            monkeypatch.setenv(name, "http://unreachable-proxy.invalid:99")
        monkeypatch.setenv("NO_PROXY", "")
        monkeypatch.setenv("no_proxy", "")
        with local_model_fixture() as (url, calls, _):
            settings = local_settings(generation_base_url=url, openai_key="unused-hosted-key")
            assert Providers(settings).complete("system", "user") == "Local fixture answer"
            assert describe_image(b"pixels", "image/png", settings) == "Local fixture answer"
            router = ModelRouter(local_router_config(gateway_url=url))
            tools = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}]
            reply = router.complete(model="inkling", messages=[{"role": "user", "content": "look up policy"}], tools=tools)
            assert reply.tool_calls[0].arguments == {"query": "retention"}
            assert reply.reasoning_content == "private continuation fixture"
            assert "private continuation" not in reply.model_dump_json()
            assert len(calls) == 3
            assert all(call["authorization"] is None for call in calls)
            for call in calls:
                assert "max_tokens" in call["body"]
                assert "max_completion_tokens" not in call["body"] and "reasoning_effort" not in call["body"]
                assert call["body"]["model"] == INKLING_MODEL

    def test_local_truncation_is_still_a_generation_and_tool_failure(self):
        with local_model_fixture() as (url, _, state):
            state["finish_reason"] = "length"
            with pytest.raises(CapabilityError) as error:
                Providers(local_settings(generation_base_url=url)).complete("system", "user")
            assert error.value.code == "invalid_model_output"
            router = ModelRouter(local_router_config(gateway_url=url))
            with pytest.raises(CapabilityError) as error:
                router.complete(model="inkling", messages=[{"role": "user", "content": "lookup"}],
                                tools=[{"type": "function", "function": {"name": "lookup"}}])
            assert error.value.code == "invalid_model_output"

    def test_tenant_authentication_still_precedes_local_generation(self, tmp_path):
        with local_model_fixture() as (url, calls, _):
            settings = local_settings(db_path=str(tmp_path / "local-rag.db"), generation_base_url=url,
                                      keys={"owner-key": "owner", "other-key": "other"})
            with TestClient(create_app(settings)) as client:
                assert client.post("/query", json={"question": "private policy", "mode": "sparse"}).status_code == 401
                record = client.post("/documents", headers={"Authorization": "Bearer owner-key"},
                                     json={"title": "Policy", "text": "Private retention policy."})
                assert record.status_code == 201
                result = client.post("/query", headers={"Authorization": "Bearer other-key"},
                                     json={"question": "retention policy", "mode": "sparse"})
                assert result.status_code == 200 and result.json()["citations"] == []
                assert calls == []
                authorized = client.post("/query", headers={"Authorization": "Bearer owner-key"},
                                         json={"question": "retention policy", "mode": "sparse"})
                assert authorized.status_code == 200 and authorized.json()["citations"]
                assert authorized.json()["citations"][0]["document_id"] == record.json()["id"]
            assert len(calls) == 1 and calls[0]["authorization"] is None
