"""Image feature permissions precede decoders, encoders and tenant data access."""
import io

from fastapi import FastAPI, Header, HTTPException
from fastapi.testclient import TestClient
from PIL import Image
import pytest

from rags.features import FeatureFlags
from rags.multimodal import MultimodalService, install_multimodal_routes
from rags.registry import CapabilityError


pytestmark = pytest.mark.integration
AUTH = {"Authorization": "Bearer asset-feature-key"}


@pytest.fixture(autouse=True)
def isolated_multimodal_configuration(monkeypatch):
    for name in ("RAGS_FEATURES_JSON", "RAGS_FEATURES_FILE", "RAGS_MULTIMODAL_MODEL", "RAGS_MODEL_CACHE"):
        monkeypatch.delenv(name, raising=False)


class CountingEncoder:
    id = "fixture:asset-flags-v1"
    dimensions = 3
    profile = {"implementation": "protocol_fixture", "dimensions": 3}

    def __init__(self):
        self.calls = 0

    def embed_image(self, image):
        self.calls += 1
        return list(image.getpixel((0, 0)))

    def embed_text(self, query):
        self.calls += 1
        return [1, 0, 0]


def red_image():
    with Image.new("RGB", (2, 2), "red") as image:
        stream = io.BytesIO()
        image.save(stream, format="PNG")
        return stream.getvalue()


def disabled(name):
    return FeatureFlags(((name, False),))


@pytest.mark.parametrize("operation,feature", [
    ("ingest", "ingestion"), ("ingest", "multimodal"), ("ingest", "embedding.fastembed"),
    ("force_ingest", "reindex"),
    ("reindex", "reindex"), ("reindex", "ingestion"), ("reindex", "multimodal"),
    ("reindex", "embedding.fastembed"),
    ("search_text", "query"), ("search_text", "multimodal"),
    ("search_text", "strategy.multimodal_native"), ("search_text", "embedding.fastembed"),
    ("search_image", "query"), ("search_image", "multimodal"),
    ("search_image", "strategy.multimodal_native"), ("search_image", "embedding.fastembed"),
    ("get", "documents"), ("get", "multimodal"), ("original", "original_images"),
    ("list", "documents"), ("list", "multimodal"),
    ("delete", "deletion"), ("delete", "multimodal"),
])
def test_direct_image_permissions_preserve_original_before_any_work(tmp_path, monkeypatch, operation, feature):
    path = str(tmp_path / "assets.db")
    original = MultimodalService(path, encoder=CountingEncoder())
    original.ingest("tenant", red_image(), asset_id="asset")
    before = original.store.get("tenant", "asset", original=True)
    encoder = CountingEncoder()
    service = MultimodalService(path, encoder=encoder, features=disabled(feature))

    def unexpected(*args, **kwargs):
        pytest.fail("Disabled image operation accessed a decoder, encoder loader or stored data")

    monkeypatch.setattr("rags.multimodal.validated_image", unexpected)
    monkeypatch.setattr("rags.multimodal.clip_encoder", unexpected)
    for name in ("get", "list", "delete", "save", "vectors", "dense"):
        monkeypatch.setattr(service.store, name, unexpected)
    operations = {
        "ingest": lambda: service.ingest("tenant", b"invalid", asset_id="asset"),
        "force_ingest": lambda: service.ingest("tenant", b"invalid", asset_id="asset", force=True),
        "reindex": lambda: service.reindex("tenant", "asset"),
        "search_text": lambda: service.search_text("tenant", "red"),
        "search_image": lambda: service.search_image("tenant", b"invalid"),
        "get": lambda: service.get("tenant", "asset"),
        "original": lambda: service.get("tenant", "asset", original=True),
        "list": lambda: service.list("tenant"),
        "delete": lambda: service.delete("tenant", "asset"),
    }
    with pytest.raises(CapabilityError) as error:
        operations[operation]()
    assert error.value.code == "feature_disabled" and error.value.status_code == 403
    assert feature in error.value.reason and encoder.calls == 0
    assert original.store.get("tenant", "asset", original=True) == before


@pytest.mark.parametrize("feature", ["multimodal", "query", "strategy.multimodal_native", "embedding.fastembed"])
def test_capability_distinguishes_disabled_permission_from_configured_checkpoint(tmp_path, monkeypatch, feature):
    monkeypatch.setattr("rags.multimodal.clip_encoder", lambda *_: pytest.fail("Capability must not load a model"))
    path = str(tmp_path / "assets.db")
    selected = MultimodalService(path, model_name="clip-vit-b-32", features=disabled(feature)).capability()
    assert selected["status"] == "feature_disabled" and selected["enabled"] is False
    assert selected["disabled_by"] == [feature] and selected["model_loaded"] is False
    configured = MultimodalService(path, model_name="clip-vit-b-32").capability()
    assert configured["status"] == "configured" and configured["enabled"] is True
    assert configured["disabled_by"] == [] and configured["model_loaded"] is False
    missing = MultimodalService(path, model_name="").capability()
    assert missing["enabled"] is True and missing["status"] == "requires_backend"


def make_client(path, feature, encoder):
    app = FastAPI()

    def auth(authorization: str | None = Header(default=None)):
        if authorization != AUTH["Authorization"]:
            raise HTTPException(401, "Invalid bearer token")
        return "tenant"

    features = disabled(feature) if feature is not None else None
    service = install_multimodal_routes(app, auth, db_path=path, encoder=encoder, features=features)
    return TestClient(app), service


@pytest.mark.parametrize("entry_point", ["direct", "standalone_routes"])
@pytest.mark.parametrize("configuration", ["json", "file"])
def test_default_multimodal_permissions_honor_operator_environment_before_encoding(
    tmp_path, monkeypatch, entry_point, configuration,
):
    path = str(tmp_path / "assets.db")
    original = MultimodalService(path, encoder=CountingEncoder())
    original.ingest("tenant", red_image(), asset_id="asset", title="Original")
    before = original.store.get("tenant", "asset", original=True)
    if configuration == "json":
        monkeypatch.setenv("RAGS_FEATURES_JSON", '{"multimodal":false}')
    else:
        flags = tmp_path / "features.json"
        flags.write_text('{"multimodal":false}', encoding="utf-8")
        monkeypatch.setenv("RAGS_FEATURES_FILE", str(flags))
    encoder = CountingEncoder()
    if entry_point == "direct":
        service = MultimodalService(path, encoder=encoder)
        with pytest.raises(CapabilityError) as error:
            service.ingest("tenant", red_image(), asset_id="asset", title="Replacement")
        assert error.value.code == "feature_disabled" and error.value.status_code == 403
    else:
        client, service = make_client(path, None, encoder)
        with client:
            response = client.post("/assets/images", headers=AUTH,
                                   data={"asset_id": "asset", "title": "Replacement"},
                                   files={"file": ("valid.png", red_image(), "image/png")})
            assert response.status_code == 403 and response.json()["detail"]["code"] == "feature_disabled"
    assert service.capability()["disabled_by"] == ["multimodal"]
    assert encoder.calls == 0
    assert original.store.get("tenant", "asset", original=True) == before


@pytest.mark.parametrize("path,feature", [
    ("/assets/images", "uploads"), ("/assets/images", "embedding.fastembed"),
    ("/assets/search/image", "query"), ("/assets/search/image", "strategy.multimodal_native"),
])
def test_http_image_permissions_do_not_decode_invalid_uploaded_bytes(tmp_path, monkeypatch, path, feature):
    encoder = CountingEncoder()
    monkeypatch.setattr("rags.multimodal.validated_image", lambda *_: pytest.fail("Disabled upload decoded bytes"))
    client, service = make_client(str(tmp_path / "assets.db"), feature, encoder)
    with client:
        arguments = {"files": {"file": ("invalid.png", b"not an image", "image/png")}}
        assert client.post(path, **arguments).status_code == 401
        response = client.post(path, headers=AUTH, **arguments)
        assert response.status_code == 403 and response.json()["detail"]["code"] == "feature_disabled"
    assert encoder.calls == 0 and service.store.list("tenant") == []


def test_original_download_can_be_disabled_without_losing_search_reindex_or_metadata(tmp_path):
    path = str(tmp_path / "assets.db")
    unrestricted = MultimodalService(path, encoder=CountingEncoder())
    first = unrestricted.ingest("tenant", red_image(), asset_id="asset")
    client, service = make_client(path, "original_images", CountingEncoder())
    with client:
        assert client.get("/assets/images/asset").status_code == 401
        denied = client.get("/assets/images/asset", headers=AUTH)
        assert denied.status_code == 403 and denied.json()["detail"]["code"] == "feature_disabled"
        assert client.get("/assets/images", headers=AUTH).json()[0]["revision"] == first["revision"]
        assert service.get("tenant", "asset")["id"] == "asset"
        assert client.post("/assets/search", headers=AUTH, json={"query": "red"}).json()["citations"][0]["asset_id"] == "asset"
        refreshed = client.post("/assets/images/asset/reindex", headers=AUTH)
        assert refreshed.status_code == 200 and refreshed.json()["revision"] != first["revision"]
        assert client.get("/assets/capabilities", headers=AUTH).json()["original_images_enabled"] is False
    assert unrestricted.get("tenant", "asset", original=True)["original"] == red_image()
    assert unrestricted.capability()["original_images_enabled"] is True


def test_scope_projection_preserves_source_and_profiles_without_materializing_vectors(tmp_path):
    from rags.config import Settings
    from rags.providers import Providers
    from rags.retrieval import RAGService
    from rags.storage import Store

    store = Store(str(tmp_path / "source.db"))
    service = RAGService(store, Providers(Settings.from_env()))
    service.ingest("tenant", "Title", "Alpha retains source content.", "fixture", document_id="source")
    complete = store.scope("tenant")
    projected = store.scope("tenant", include_embeddings=False)
    assert projected == [{key: value for key, value in row.items() if key != "embedding"} for row in complete]
    assert projected and "embedding" in complete[0]
