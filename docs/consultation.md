# Consultative second opinion (v1)

The selected author remains responsible for the answer. An OpenAI colleague offers
optional corrections, relevant risks/benefits and supporting/contradicting evidence.
This is enrichment, not a quality gate or a claim of clinically verified accuracy.

## Request flow

`Convex /chat -> Cortex retrieval -> Decisions route -> author -> optional colleague -> author final`

With consultation disabled, the existing provider stream is passed through unchanged.
With consultation enabled, Decisions sees the latest six messages (bounded text and
current images), not the entire memory. General conversation, emotional support and
pure trauma/psychology use the normal single author pass. Medical/physiological and
mixed turns use a private author draft, a colleague call and a final author pass.
The colleague receives the same full persona, compilation, conversation and images
as the author, with memory/persona quoted as reference data. It can search the web;
search activity is established from provider usage/annotations, never from a link alone.
The author decides what advice to incorporate. Only its final text enters the chat
history and subsequent memory ingestion. Drafts/advice remain internal telemetry.

Routing and advice have hard deadlines. Failure replays the original author answer.
A final failure before any text also replays the draft. After final text is published,
interruption terminates that response with an error, without appending the draft.
Cancellation closes upstream generators; SSE comment heartbeats keep quiet stages alive.
No new product UI is required. Frontend local HTTP actions can be configured through
`VITE_CONVEX_SITE_URL`; cloud deployments keep deriving the site URL from cloud URL.

## Configuration

```dotenv
CONSULTATION_ENABLED=true
CONSULTATION_DECISION_MODEL=openai/gpt-6-luna-decisions
CONSULTATION_MODEL=openai/gpt-6.1-sol
CONSULTATION_REASONING_EFFORT=medium
CONSULTATION_TIMEOUT_SECONDS=60
CONSULTATION_ROUTING_TIMEOUT_SECONDS=4
CONSULTATION_SEARCH_ENABLED=true
OPEN_ROUTER_API_KEY=<existing server credential>
```

Default is disabled; restart Cortex after changing settings. Docker Compose forwards
all consultation settings. No direct OpenAI key or new client dependency is needed.
OpenRouter supplies both Decisions and OpenAI chat/search. Direct OpenAI transport
is not implemented in v1. Set `CONSULTATION_ENABLED=false` to return to the ordinary flow.
The flag applies deployment-wide; per-user rollout is not implemented in this version.

## Observability and accounting

All stages share the existing PostHog trace/session/user identity. `$ai_generation`
events use `consultation_stage=routing|draft|advice|final` (ordinary author: response).
`consultation.pipeline` records category/probability, selected model/effort, durations,
status, errors, actual search count, source provenance and stage token/cost totals.
Latency is in seconds in PostHog; summary duration fields explicitly use milliseconds.
Structured `consultation.completed` console logs omit conversation text and source URLs.

Advisor telemetry includes a bounded context excerpt and SHA-256/length of the full
context; images are redacted. The actual advisor request receives the full context.
Existing author telemetry retains its previous content policy. These internal traces
can contain sensitive conversation material and use the existing PostHog access policy.

Cortex emits aggregate usage once per published response. Tokens include routing,
draft, advice and final work; BYOK costs include upstream charges when reported.
Unusable HTTP200 colleague responses retain billed usage. Partial final failures
retain all known usage, without inventing missing author billing. `known_cost_usd`
is a lower bound; `cost_complete=false` on failure or unknown-stage costs. Vertex
currently has no returned dollar estimate, so mixed Vertex/OpenRouter totals remain
incomplete. Published grounding/cache metadata belongs only to the published author;
colleague sources are separate and label provider annotations vs model-written links.

Convex validates and persists internal `metadata.consultation`, associates its trace
with analytics, and strips this metadata from public message queries. Invalid metadata
is dropped without blocking answer persistence. Telemetry failures never block chat.
Search for the trace ID in PostHog LLM Analytics and `consultation.completed` logs.

## Tests and evaluation

`make verify` covers client routing/search/context, hard deadlines, both author
providers, request immutability, realistic SSE null usage, buffering, fail-open,
partial accounting, grounding identity, cancellation and telemetry failures. Run
`npm run verify:build` in the chat repo for Convex/public boundary and frontend tests.
These deterministic tests validate orchestration and recovery behavior, not clinical
accuracy or the factual quality of generated advice. Validate model changes separately
against representative cases and primary sources before changing the defaults.
