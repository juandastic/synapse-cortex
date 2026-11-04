"""Service-authenticated dictation; raw audio is never persisted."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request

from app.api.dependencies import ApiKeyDep
from app.core.config import Settings, get_settings
from app.services.transcription import AUDIO_FORMATS, MAX_AUDIO_BYTES, transcribe_audio

router = APIRouter()


@router.post("/v1/audio/transcriptions")
async def transcribe(
    request: Request,
    _api_key: ApiKeyDep,
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict[str, str]:
    key = settings.open_ai_api_key.get_secret_value()
    if not key:
        raise HTTPException(503, "Transcription unavailable")
    mime_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if mime_type not in AUDIO_FORMATS:
        raise HTTPException(415, "Unsupported audio format")
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            size = int(content_length)
        except ValueError as exc:
            raise HTTPException(400, "Invalid content length") from exc
        if size < 0 or size > MAX_AUDIO_BYTES:
            raise HTTPException(413, "Audio exceeds 10 MB")
    audio = bytearray()
    async for chunk in request.stream():
        if len(audio) + len(chunk) > MAX_AUDIO_BYTES:
            raise HTTPException(413, "Audio exceeds 10 MB")
        audio.extend(chunk)
    if not audio:
        raise HTTPException(400, "Audio is empty")
    text = await transcribe_audio(bytes(audio), mime_type, key, settings.openai_transcription_model)
    return {"text": text}
