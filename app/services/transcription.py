"""Bounded, ephemeral audio-to-text requests to OpenAI."""

import httpx
from fastapi import HTTPException

MAX_AUDIO_BYTES = 10 * 1024 * 1024
AUDIO_FORMATS = {
    "audio/webm": "webm",
    "audio/mp4": "mp4",
    "audio/m4a": "m4a",
    "audio/x-m4a": "m4a",
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
}


async def transcribe_audio(audio: bytes, mime_type: str, api_key: str, model: str) -> str:
    # Do not log provider response bodies, audio, transcripts or authorization headers.
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(90, connect=10)) as client:
            response = await client.post(
                "https://api.openai.com/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {api_key}"},
                data={"model": model},
                files={"file": (f"recording.{AUDIO_FORMATS[mime_type]}", audio, mime_type)},
            )
    except httpx.TimeoutException as exc:
        raise HTTPException(504, "Transcription timed out") from exc
    except httpx.RequestError as exc:
        raise HTTPException(502, "Transcription provider unavailable") from exc

    if response.status_code in (400, 413, 415, 422):
        raise HTTPException(422, "Unable to transcribe this recording")
    if response.status_code in (401, 403, 404, 429):
        raise HTTPException(503, "Transcription unavailable")
    if not response.is_success:
        raise HTTPException(502, "Transcription provider unavailable")
    try:
        text = response.json()["text"]
        if not isinstance(text, str) or not text.strip() or len(text) > 100_000:
            raise ValueError("Invalid transcript")
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(422, "No readable speech was returned") from exc
    return text.strip()
