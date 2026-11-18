"""Internal results for advisory second opinions, never publication gates."""

from typing import Literal

from pydantic import BaseModel, Field


class RouteDecision(BaseModel):
    category: Literal["general", "medical", "mixed"]
    probability: float = Field(ge=0, le=1)
    usage: dict = Field(default_factory=dict)
    duration_ms: float


class ConsultationSource(BaseModel):
    title: str
    url: str
    provenance: Literal["provider_annotation", "model_link"] = "provider_annotation"


class ConsultationAdvice(BaseModel):
    content: str
    sources: list[ConsultationSource] = Field(default_factory=list)
    usage: dict = Field(default_factory=dict)
    model: str
    reasoning_effort: str
    duration_ms: float
    search_used: bool = False
    search_requests: int = Field(default=0, ge=0)
