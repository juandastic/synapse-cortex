from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from app.api.routes import router
from app.core import security
from app.core.config import Settings, get_settings
from app.services.transcription import MAX_AUDIO_BYTES


@pytest.fixture
async def audio_api(monkeypatch):
    settings = Settings(
        _env_file=None,
        neo4j_password="test",
        synapse_api_secret="service-secret",
        OPEN_AI_API_KEY="private-openai-key",
    )
    monkeypatch.setattr(security, "get_settings", lambda: settings)
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_settings] = lambda: settings
    async_client = httpx.AsyncClient
    client = async_client(transport=httpx.ASGITransport(app=app), base_url="http://test")
    provider = SimpleNamespace(status=200, body={"text": "  Vivo en Bogotá.  "}, error=None)
    requests = []

    def respond(request):
        requests.append(request)
        if provider.error:
            raise provider.error
        return httpx.Response(provider.status, json=provider.body)

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: async_client(transport=httpx.MockTransport(respond), **kwargs),
    )
    async with client:
        yield client, requests, provider, settings


HEADERS = {"X-API-SECRET": "service-secret", "Content-Type": "audio/webm;codecs=opus"}


async def test_transcribes_raw_audio_without_persisting_or_returning_secrets(audio_api):
    client, requests, _, _ = audio_api
    response = await client.post("/v1/audio/transcriptions", headers=HEADERS, content=b"voice")
    assert response.status_code == 200
    assert response.json() == {"text": "Vivo en Bogotá."}
    assert len(requests) == 1
    assert str(requests[0].url) == "https://api.openai.com/v1/audio/transcriptions"
    assert requests[0].headers["authorization"] == "Bearer private-openai-key"
    assert b"gpt-transcribe" in requests[0].content
    assert b"recording.webm" in requests[0].content
    assert b"voice" in requests[0].content
    assert "private-openai-key" not in response.text


@pytest.mark.parametrize("secret,status", [(None, 422), ("incorrect", 401)])
async def test_rejects_unauthenticated_audio_before_calling_provider(audio_api, secret, status):
    client, requests, _, _ = audio_api
    headers = {"Content-Type": "audio/webm"}
    if secret:
        headers["X-API-SECRET"] = secret
    response = await client.post("/v1/audio/transcriptions", headers=headers, content=b"voice")
    assert response.status_code == status
    assert not requests


@pytest.mark.parametrize(
    "mime,body,status",
    [
        ("text/plain", b"voice", 415),
        ("audio/webm", b"", 400),
        ("audio/webm", b"x" * (MAX_AUDIO_BYTES + 1), 413),
    ],
)
async def test_invalid_uploads_never_reach_openai(audio_api, mime, body, status):
    client, requests, _, _ = audio_api
    response = await client.post(
        "/v1/audio/transcriptions", headers={**HEADERS, "Content-Type": mime}, content=body
    )
    assert response.status_code == status
    assert not requests


async def test_chunked_upload_cannot_bypass_size_limit(audio_api):
    client, requests, _, _ = audio_api

    async def chunks():
        yield b"x" * MAX_AUDIO_BYTES
        yield b"x"

    response = await client.post("/v1/audio/transcriptions", headers=HEADERS, content=chunks())
    assert response.status_code == 413
    assert not requests


@pytest.mark.parametrize(
    "upstream,status", [(401, 503), (404, 503), (429, 503), (500, 502), (400, 422)]
)
async def test_provider_errors_are_sanitized(audio_api, upstream, status):
    client, _, provider, _ = audio_api
    provider.status = upstream
    provider.body = {"error": {"message": "private-openai-key sensitive provider detail"}}
    response = await client.post("/v1/audio/transcriptions", headers=HEADERS, content=b"voice")
    assert response.status_code == status
    assert "private-openai-key" not in response.text


@pytest.mark.parametrize("text", ["", "  ", None, 7])
async def test_missing_speech_is_recoverable_instead_of_an_empty_message(audio_api, text):
    client, _, provider, _ = audio_api
    provider.body = {"text": text}
    response = await client.post("/v1/audio/transcriptions", headers=HEADERS, content=b"voice")
    assert response.status_code == 422


async def test_provider_timeout_is_recoverable(audio_api):
    client, _, provider, _ = audio_api
    provider.error = httpx.ReadTimeout("secret provider details")
    response = await client.post("/v1/audio/transcriptions", headers=HEADERS, content=b"voice")
    assert response.status_code == 504
    assert "secret provider details" not in response.text


async def test_missing_key_does_not_attempt_provider_request(audio_api):
    client, requests, _, settings = audio_api
    settings.open_ai_api_key = settings.open_ai_api_key.__class__("")
    response = await client.post("/v1/audio/transcriptions", headers=HEADERS, content=b"voice")
    assert response.status_code == 503
    assert not requests


@pytest.mark.parametrize("name", ["OPEN_AI_API_KEY", "OPENAI_API_KEY"])
def test_both_environment_key_names_are_supported_and_masked(monkeypatch, name):
    monkeypatch.delenv("OPEN_AI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv(name, "private-openai-key")
    settings = Settings(_env_file=None, neo4j_password="test", synapse_api_secret="test")
    assert settings.open_ai_api_key.get_secret_value() == "private-openai-key"
    assert "private-openai-key" not in repr(settings)
