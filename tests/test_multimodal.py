"""Native asset authorization/lifecycle tests; the fixture encoder is not CLIP."""
import io
import json
import math
import os

from fastapi import FastAPI, Header, HTTPException
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw
import pytest

from rags.multimodal import (
    MAX_IMAGE_BYTES, MultimodalService, clip_encoder, install_multimodal_routes,
    normalized_vector, validated_image,
)
from rags.registry import CapabilityError


def image_bytes(color="red", *, shape=None, format="PNG"):
    image = Image.new("RGB", (224, 224), "white" if shape else color)
    if shape:
        drawing = ImageDraw.Draw(image)
        if shape == "circle":
            drawing.ellipse((35, 35, 189, 189), fill=color)
        else:
            drawing.rectangle((35, 35, 189, 189), fill=color)
    stream = io.BytesIO()
    image.save(stream, format=format)
    image.close()
    return stream.getvalue()


class RGBFixtureEncoder:
    """Protocol fixture only: verifies storage/scoring, never semantic quality."""

    id = "fixture:rgb-v1"
    dimensions = 3
    profile = {"implementation": "protocol_fixture", "dimensions": 3}

    def __init__(self):
        self.image_calls = 0
        self.invalid = False

    def embed_image(self, image):
        self.image_calls += 1
        if self.invalid:
            return [0, 0, 0]
        return list(image.getpixel((0, 0)))

    def embed_text(self, query):
        return {"red": [1, 0, 0], "blue": [0, 0, 1], "green": [0, 1, 0]}.get(query, [1, 1, 1])


def app_for(path, encoder=None, model_name=""):
    app = FastAPI()

    def auth(authorization: str | None = Header(default=None)):
        keys = {"Bearer alpha-key": "alpha", "Bearer beta-key": "beta"}
        if authorization not in keys:
            raise HTTPException(401, "Invalid bearer token")
        return keys[authorization]

    service = install_multimodal_routes(app, auth, db_path=str(path), model_name=model_name, encoder=encoder)
    return TestClient(app), service


ALPHA = {"Authorization": "Bearer alpha-key"}
BETA = {"Authorization": "Bearer beta-key"}


def upload(client, raw, *, headers=ALPHA, asset_id="asset-a", **data):
    return client.post("/assets/images", headers=headers, data={"asset_id": asset_id, **data},
                       files={"file": ("source.png", raw, "image/png")})


def test_authenticated_asset_lifecycle_is_tenant_scoped(tmp_path):
    client, service = app_for(tmp_path / "assets.db", RGBFixtureEncoder())
    red, blue = image_bytes("red"), image_bytes("blue")
    assert upload(client, red, headers={}).status_code == 401
    created = upload(client, red, title="Original red image").json()
    assert created["dimensions"] == 3
    assert "original" not in created and "vector" not in created
    assert client.get("/assets/images/asset-a", headers=BETA).status_code == 404
    assert client.post("/assets/images/asset-a/reindex", headers=BETA).status_code == 404
    assert client.delete("/assets/images/asset-a", headers=BETA).status_code == 404
    assert client.get("/assets/images", headers=BETA).json() == []
    assert client.post("/assets/search", headers=BETA, json={"query": "red"}).json()["citations"] == []
    assert upload(client, blue, headers=BETA).status_code == 201
    original = client.get(created["asset_uri"], headers=ALPHA)
    assert original.content == red
    assert original.headers["content-type"] == "image/png"
    assert original.headers["cache-control"] == "no-store"
    assert original.headers["x-content-type-options"] == "nosniff"
    assert client.get(created["asset_uri"], headers=BETA).content == blue
    hit = client.post("/assets/search", headers=ALPHA, json={"query": "red"}).json()["citations"][0]
    assert hit["asset_id"] == "asset-a" and hit["title"] == "Original red image"
    assert hit["score"] == 1.0 and hit["evidence_kind"] == "original_image"
    assert client.delete(created["asset_uri"], headers=ALPHA).status_code == 200
    assert service.store.get("alpha", "asset-a", original=True) is None
    with service.store.connect() as connection:
        assert connection.execute("SELECT count(*) FROM image_assets WHERE tenant=?", ("alpha",)).fetchone()[0] == 0
    assert client.get(created["asset_uri"], headers=BETA).content == blue


def test_idempotence_failed_encoding_and_stale_model_reindex(tmp_path):
    encoder = RGBFixtureEncoder()
    client, service = app_for(tmp_path / "assets.db", encoder)
    red, blue = image_bytes("red"), image_bytes("blue")
    first = upload(client, red).json()
    repeat = upload(client, red).json()
    assert repeat["unchanged"] is True and repeat["revision"] == first["revision"]
    assert encoder.image_calls == 1
    encoder.invalid = True
    assert upload(client, blue).status_code == 502
    assert client.get("/assets/images/asset-a", headers=ALPHA).content == red
    assert service.store.get("alpha", "asset-a")["revision"] == first["revision"]
    encoder.invalid = False
    encoder.id = "fixture:rgb-v2"
    stale = client.post("/assets/search", headers=ALPHA, json={"query": "red"})
    assert stale.status_code == 409 and stale.json()["detail"]["code"] == "asset_reindex_required"
    refreshed = client.post("/assets/images/asset-a/reindex", headers=ALPHA).json()
    assert refreshed["encoder_id"] == encoder.id and refreshed["revision"] != first["revision"]
    assert client.get("/assets/images/asset-a", headers=ALPHA).content == red
    assert client.post("/assets/search", headers=ALPHA, json={"query": "red"}).status_code == 200


def test_concurrent_update_cannot_be_overwritten_after_encoding(tmp_path):
    path = str(tmp_path / "assets.db")
    first_encoder, concurrent_encoder = RGBFixtureEncoder(), RGBFixtureEncoder()
    first = MultimodalService(path, encoder=first_encoder)
    concurrent = MultimodalService(path, encoder=concurrent_encoder)
    first.ingest("alpha", image_bytes("red"), asset_id="same")
    original = first_encoder.embed_image

    def racing_encode(image):
        concurrent.ingest("alpha", image_bytes("blue"), asset_id="same")
        return original(image)

    first_encoder.embed_image = racing_encode
    with pytest.raises(CapabilityError) as error:
        first.ingest("alpha", image_bytes("green"), asset_id="same")
    assert error.value.code == "revision_conflict"
    assert first.store.get("alpha", "same", original=True)["original"] == image_bytes("blue")


def test_image_and_request_limits_are_explicit(tmp_path, monkeypatch):
    client, _ = app_for(tmp_path / "assets.db", RGBFixtureEncoder())
    assert upload(client, b"not an image").status_code == 422
    assert upload(client, b"x" * (MAX_IMAGE_BYTES + 1)).status_code == 413
    assert upload(client, image_bytes(format="GIF")).status_code == 415
    assert upload(client, image_bytes(), asset_id="../escape").status_code == 422
    assert upload(client, image_bytes(), title=" ").status_code == 422
    assert client.post("/assets/search", headers=ALPHA, json={"query": "red", "tenant": "beta"}).status_code == 422
    assert client.post("/assets/search", headers=ALPHA, json={"query": "red", "url": "https://example.com"}).status_code == 422
    assert client.post("/assets/search", headers=ALPHA, json={"query": "red", "top_k": 0}).status_code == 422
    monkeypatch.setattr("rags.multimodal.MAX_IMAGE_PIXELS", 100)
    assert upload(client, image_bytes()).status_code == 413


def test_model_is_disabled_until_operator_configures_it(tmp_path, monkeypatch):
    def must_not_load(*args):
        raise AssertionError("Disabled model must never initialize")

    monkeypatch.setattr("rags.multimodal.clip_encoder", must_not_load)
    client, _ = app_for(tmp_path / "assets.db")
    assert client.get("/assets/capabilities", headers=ALPHA).json()["status"] == "requires_backend"
    assert upload(client, image_bytes()).status_code == 503
    assert client.post("/assets/search", headers=ALPHA, json={"query": "red"}).status_code == 503
    assert client.get("/assets/images", headers=ALPHA).json() == []
    with pytest.raises(ValueError, match="declared aligned pair"):
        MultimodalService(str(tmp_path / "bad.db"), model_name="unrelated-encoder")


def test_image_query_and_stored_vectors_are_normalized(tmp_path):
    client, service = app_for(tmp_path / "assets.db", RGBFixtureEncoder())
    upload(client, image_bytes("red"), asset_id="red")
    upload(client, image_bytes("blue"), asset_id="blue")
    response = client.post("/assets/search/image", headers=ALPHA,
                           files={"file": ("query.png", image_bytes("blue"), "image/png")})
    assert response.status_code == 200
    assert response.json()["query_modality"] == "image"
    assert response.json()["citations"][0]["asset_id"] == "blue"
    for row in service.store.vectors("alpha"):
        assert math.isclose(math.hypot(*json.loads(row["vector"])), 1.0)


@pytest.mark.parametrize("vector", [[0, 0], [1], [float("nan"), 1], [float("inf"), 1], [True, 0]])
def test_invalid_vectors_are_never_indexed(vector):
    with pytest.raises(CapabilityError):
        normalized_vector(vector, 2)


@pytest.mark.skipif(os.getenv("RAGS_TEST_MULTIMODAL") != "1", reason="Set RAGS_TEST_MULTIMODAL=1 to load real paired CLIP checkpoints")
def test_real_clip_text_image_and_image_image_retrieval(tmp_path):
    """Actual pretrained CLIP execution over synthetic images, not a vector stub."""
    encoder = clip_encoder(os.getenv("RAGS_MODEL_CACHE") or None)
    service = MultimodalService(str(tmp_path / "real-assets.db"), model_name="clip-vit-b-32", encoder=encoder)
    red = image_bytes("red", shape="circle")
    blue = image_bytes("blue", shape="square")
    service.ingest("alpha", red, filename="a.png", asset_id="a")
    service.ingest("alpha", blue, filename="b.png", asset_id="b")
    red_query = service.search_text("alpha", "a red circle on a white background", 2)
    blue_query = service.search_text("alpha", "a blue square on a white background", 2)
    assert red_query["citations"][0]["asset_id"] == "a"
    assert blue_query["citations"][0]["asset_id"] == "b"
    image_query = service.search_image("alpha", red, 2)
    assert image_query["citations"][0]["asset_id"] == "a"
    assert image_query["citations"][0]["score"] > 0.999
    assert encoder.dimensions == 512
    assert len(encoder.profile["text_artifacts_sha256"]) == 64
    assert len(encoder.profile["image_artifacts_sha256"]) == 64
    assert service.search_text("beta", "a red circle")["citations"] == []
    with pytest.raises(CapabilityError) as error:
        service.search_text("alpha", "word " * 100)
    assert error.value.code == "query_too_long"
    image, media = validated_image(service.store.get("alpha", "a", original=True)["original"])
    assert media == "image/png" and image.size == (224, 224)
    image.close()
    assert service.store.delete("alpha", "a")
    assert all(hit["asset_id"] != "a" for hit in service.search_text("alpha", "a red circle")["citations"])
