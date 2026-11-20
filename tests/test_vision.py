import io

from fastapi.testclient import TestClient
from PIL import Image

from rags.api import create_app
from rags.config import Settings
from rags.vision import validate_image


def image_bytes():
    buffer = io.BytesIO()
    Image.new("RGB", (12, 10)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_image_type_validation():
    assert validate_image(image_bytes()) == "image/png"
    try:
        validate_image(b"not a real PNG")
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid image accepted")


def test_vision_caption_ingestion(monkeypatch, tmp_path):
    monkeypatch.setenv("RAGS_ENV", "test")
    monkeypatch.setenv("RAGS_DB_PATH", str(tmp_path / "images.db"))
    monkeypatch.setenv("RAGS_MODEL_PROVIDER", "openai")
    monkeypatch.setenv("RAGS_KEYS_JSON", '{"vision-key":"visual-tenant"}')
    monkeypatch.setattr(
        "rags.vision.describe_image",
        lambda raw, media_type, settings: "Diagram: Login Service connects to Session Store.",
    )
    with TestClient(create_app(Settings.from_env())) as client:
        headers = {"Authorization": "Bearer vision-key"}
        result = client.post(
            "/images", headers=headers,
            files={"file": ("architecture.png", image_bytes(), "image/png")},
        )
        assert result.status_code == 201, result.text
        query = client.post(
            "/query", headers=headers, json={
                "question": "Which service connects to Session Store?",
                "generate": False,
            },
        )
        assert query.status_code == 200
        assert query.json()["citations"][0]["source"] == "image:architecture.png"
        doc = client.get("/documents/" + result.json()["id"], headers=headers).json()
        assert doc["metadata"]["caption_generated_by"] == "openai"
