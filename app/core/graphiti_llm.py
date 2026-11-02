"""Graphiti's OpenRouter adapter: explicit model roles and per-call observability."""

import json
import logging
import time

import httpx
import openai
from graphiti_core.llm_client.client import (
    LLMClient,
    get_extraction_language_instruction,
    is_server_or_retry_error,
)
from graphiti_core.llm_client.config import LLMConfig, ModelSize
from graphiti_core.llm_client.errors import EmptyResponseError, RateLimitError
from graphiti_core.prompts.models import Message
from pydantic import BaseModel
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_random_exponential

from app.core.openrouter_usage import openrouter_usage_properties
from app.core.posthog import capture_generation, get_llm_context, new_trace_id

logger = logging.getLogger(__name__)
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterGraphitiClient(LLMClient):
    """Use Graphiti's prompts/retries with OpenAI SDK strict JSON parsing.

    Capture events ourselves so OpenRouter's reported cost is retained and each
    actual attempt is attributed once, even when Graphiti retries a failed call.
    """

    def __init__(
        self,
        config: LLMConfig,
        *,
        reasoning_effort: str = "low",
        small_reasoning_effort: str = "none",
        client: openai.AsyncOpenAI | None = None,
    ):
        super().__init__(config)
        self.reasoning_effort = reasoning_effort
        self.small_reasoning_effort = small_reasoning_effort
        self.client = client or openai.AsyncOpenAI(
            api_key=config.api_key,
            base_url=OPENROUTER_BASE_URL,
            timeout=120,
            # Graphiti retries transient errors; avoid multiplying retry layers.
            max_retries=0,
        )

    async def close(self) -> None:
        await self.client.close()

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_random_exponential(multiplier=1, min=1, max=15),
        retry=retry_if_exception(
            lambda error: (
                is_server_or_retry_error(error) or isinstance(error, openai.APIConnectionError)
            )
        ),
        reraise=True,
    )
    async def _generate_response_with_retry(
        self,
        messages,
        response_model=None,
        max_tokens=16384,
        model_size=ModelSize.medium,
        prompt_name=None,
    ) -> dict:
        return await self._generate_response(
            messages, response_model, max_tokens, model_size, prompt_name
        )

    async def generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int | None = None,
        model_size: ModelSize = ModelSize.medium,
        group_id: str | None = None,
        prompt_name: str | None = None,
        *,
        attribute_extraction: bool = False,
    ) -> dict:
        # Graphiti's prompts can be reused. Do not append instructions in-place.
        messages = [message.model_copy(deep=True) for message in messages]
        instruction = get_extraction_language_instruction(group_id)
        if not messages or messages[0].role != "system":
            messages.insert(0, Message(role="system", content=instruction))
        else:
            messages[0].content += instruction
        self._apply_attribute_extraction_preamble(messages, attribute_extraction)
        small = model_size == ModelSize.small
        model = self.small_model if small else self.model
        with self.tracer.start_span("llm.generate") as span:
            span.add_attributes(
                {
                    "llm.provider": "openrouter",
                    "llm.model": model,
                    "model.size": model_size.value,
                }
            )
            if prompt_name:
                span.add_attributes({"prompt.name": prompt_name})
            try:
                return await self._generate_response_with_retry(
                    messages, response_model, max_tokens or self.max_tokens, model_size, prompt_name
                )
            except Exception as error:
                span.set_status("error", str(error))
                span.record_exception(error)
                raise

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = 16384,
        model_size: ModelSize = ModelSize.medium,
        prompt_name: str | None = None,
    ) -> dict:
        small = model_size == ModelSize.small
        model = self.small_model if small else self.model
        effort = self.small_reasoning_effort if small else self.reasoning_effort
        input_messages = [
            {"role": message.role, "content": self._clean_input(message.content)}
            for message in messages
        ]
        started_at = time.monotonic()
        response = None
        error_message = None
        try:
            kwargs = {
                "model": model,
                "messages": input_messages,
                "max_completion_tokens": max_tokens,
                "extra_body": {
                    "provider": {"require_parameters": True},
                    "reasoning": {"effort": effort, "exclude": True},
                },
            }
            if response_model is not None:
                response = await self.client.chat.completions.parse(
                    **kwargs, response_format=response_model
                )
            else:
                response = await self.client.chat.completions.create(
                    **kwargs, response_format={"type": "json_object"}
                )
            if not response.choices:
                raise EmptyResponseError("OpenRouter returned no choices")
            message = response.choices[0].message
            if message.refusal:
                raise ValueError(f"OpenRouter refused graph extraction: {message.refusal}")
            if not message.content:
                raise EmptyResponseError("OpenRouter returned no graph extraction content")
            if response_model is not None:
                return message.parsed.model_dump()
            return json.loads(message.content)
        except openai.RateLimitError as error:
            error_message = str(error)
            raise RateLimitError from error
        except openai.APIStatusError as error:
            error_message = str(error)
            # Graphiti's retry predicate understands HTTP 5xx errors.
            raise httpx.HTTPStatusError(
                str(error), request=error.request, response=error.response
            ) from error
        except Exception as error:
            error_message = str(error)
            # SDK parsing errors may carry a completion with billable usage.
            response = getattr(error, "completion", None) or response
            raise
        finally:
            self._capture_attempt(
                response,
                model,
                model_size,
                effort,
                input_messages,
                started_at,
                error_message,
                prompt_name,
            )

    def _capture_attempt(
        self, response, model, model_size, effort, messages, started_at, error, prompt_name
    ):
        context = get_llm_context()
        usage = response.usage.model_dump() if response and response.usage else {}
        choices = response.choices if response else []
        self.token_tracker.record(
            prompt_name, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
        )
        properties = {
            "pipeline": "graphiti",
            "prompt_name": prompt_name,
            "model_size": model_size.value,
            "reasoning_effort": effort,
            "requested_model": model,
            "$ai_latency": time.monotonic() - started_at,
            **openrouter_usage_properties(usage),
            "finish_reason": choices[0].finish_reason if choices else None,
        }
        try:
            capture_generation(
                context.distinct_id if context else "anonymous",
                context.trace_id if context else new_trace_id(),
                session_id=context.session_id if context else None,
                provider="openrouter",
                model=response.model if response else model,
                input_messages=messages,
                output=choices[0].message.content if choices else None,
                input_tokens=usage.get("prompt_tokens"),
                output_tokens=usage.get("completion_tokens"),
                error=error,
                properties={key: value for key, value in properties.items() if value is not None},
            )
        except Exception:
            logger.warning("Could not capture Graphiti generation analytics", exc_info=True)
