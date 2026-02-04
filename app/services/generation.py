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
            gemini_contents = self._convert_to_gemini_format(request.messages)

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

    def _convert_to_gemini_format(self, messages: list[ChatMessage]) -> list[dict]:
        """
        Convert OpenAI-style messages to Gemini format.
        
        Gemini uses a different format:
        - System messages become part of the first user message
        - Messages are grouped into user/model turns
        """
        gemini_contents = []
        system_prompt = None

        # Extract system prompt if present
        for msg in messages:
            if msg.role == "system":
                system_prompt = msg.content
                break

        # Build conversation history
        for msg in messages:
            if msg.role == "system":
                continue

            content = msg.content

            # Prepend system prompt to first user message
            if msg.role == "user" and system_prompt and not gemini_contents:
                content = f"{system_prompt}\n\n{content}"
                system_prompt = None  # Only prepend once

            role = "user" if msg.role == "user" else "model"
            gemini_contents.append({"role": role, "parts": [content]})

        return gemini_contents
