"""
Generation Service - Gemini streaming with OpenAI-compatible SSE format.

Wraps the Google GenAI SDK and outputs SSE-formatted chunks
compatible with OpenAI's streaming API format.
"""

import json
import logging
import time
import uuid
from collections.abc import AsyncGenerator

import httpx
from google import genai
from google.genai import types

from app.schemas.models import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionDelta,
    ChatCompletionRequest,
    ChatMessage,
)

logger = logging.getLogger(__name__)


class GenerationService:
    """Service for generating chat completions using Google Gemini."""

    def __init__(self, api_key: str):
        self._client = genai.Client(api_key=api_key)

    async def stream_chat_completion(
        self, request: ChatCompletionRequest
    ) -> AsyncGenerator[str, None]:
        """
        Stream chat completion in OpenAI-compatible SSE format.
        
        Args:
            request: The chat completion request.
        
        Yields:
            SSE-formatted strings: "data: {json}\n\n"
        """
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        try:
            # Convert messages to Gemini format
            gemini_contents = await self._convert_to_gemini_format(request.messages)

            # Stream the first chunk with role
            first_chunk = ChatCompletionChunk(
                id=completion_id,
                created=created,
                model=request.model,
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        delta=ChatCompletionDelta(role="assistant"),
                        finish_reason=None,
                    )
                ],
            )
            yield f"data: {first_chunk.model_dump_json()}\n\n"

            # Use async client for true async streaming
            async for chunk in await self._client.aio.models.generate_content_stream(
                model=request.model,
                contents=gemini_contents,
            ):
                if chunk.text:
                    content_chunk = ChatCompletionChunk(
                        id=completion_id,
                        created=created,
                        model=request.model,
                        choices=[
                            ChatCompletionChoice(
                                index=0,
                                delta=ChatCompletionDelta(content=chunk.text),
                                finish_reason=None,
                            )
                        ],
                    )
                    yield f"data: {content_chunk.model_dump_json()}\n\n"

            # Send final chunk with finish_reason
            final_chunk = ChatCompletionChunk(
                id=completion_id,
                created=created,
                model=request.model,
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        delta=ChatCompletionDelta(),
                        finish_reason="stop",
                    )
                ],
            )
            yield f"data: {final_chunk.model_dump_json()}\n\n"

            # Send the done marker
            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"Error during generation: {e}", exc_info=True)
            # Send error as a chunk
            error_data = json.dumps({"error": str(e)})
            yield f"data: {error_data}\n\n"
            yield "data: [DONE]\n\n"

    async def _download_image(self, url: str) -> tuple[bytes, str]:
        """
        Download image from a URL and return (bytes, mime_type).

        Uses the Content-Type response header to detect MIME type,
        falling back to image/jpeg if unavailable.
        """
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            response.raise_for_status()
            mime_type = response.headers.get("content-type", "image/jpeg").split(";")[0]
            return response.content, mime_type

    async def _build_parts(self, content: str | list) -> list[types.Part]:
        """
        Convert a message's content (string or multimodal array) into
        a list of Gemini Part objects.
        """
        if isinstance(content, str):
            return [types.Part.from_text(text=content)]

        parts: list[types.Part] = []
        for item in content:
            if item.type == "text":
                parts.append(types.Part.from_text(text=item.text))
            elif item.type == "image_url":
                image_bytes, mime_type = await self._download_image(
                    item.image_url.url
                )
                parts.append(
                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
                )
        return parts

    async def _convert_to_gemini_format(
        self, messages: list[ChatMessage]
    ) -> list[types.Content]:
        """
        Convert OpenAI-style messages to Gemini format.

        Gemini uses a different format:
        - System messages become part of the first user message
        - Messages are grouped into user/model turns
        - Multimodal content arrays are converted to multiple Part objects
        """
        gemini_contents: list[types.Content] = []
        system_prompt: str | None = None

        # Extract system prompt if present (always a string)
        for msg in messages:
            if msg.role == "system":
                system_prompt = msg.content if isinstance(msg.content, str) else None
                break

        # Build conversation history
        for msg in messages:
            if msg.role == "system":
                continue

            parts = await self._build_parts(msg.content)

            # Prepend system prompt to first user message
            if msg.role == "user" and system_prompt and not gemini_contents:
                parts.insert(0, types.Part.from_text(text=system_prompt))
                system_prompt = None  # Only prepend once

            role = "user" if msg.role == "user" else "model"
            gemini_contents.append(
                types.Content(role=role, parts=parts)
            )

        return gemini_contents
