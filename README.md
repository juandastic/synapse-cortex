# Synapse Cortex

**Cognitive backend for the Synapse AI Chat application**. A stateless REST API that processes conversational data into a dynamic knowledge graph, enabling personalized long-term memory and intelligent context retrieval for AI assistants.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Core Features](#core-features)
- [Technical Architecture](#technical-architecture)
- [Backend Components](#backend-components)
- [API Endpoints](#api-endpoints)
- [Data Flow](#data-flow)
- [Technology Stack](#technology-stack)
- [Observability in Axiom](#observability-in-axiom)
- [Notion Export](#notion-export)
- [Notion Correction Import](#notion-correction-import)
- [Setup & Deployment](#setup--deployment)

---

## Overview

Synapse Cortex is a **knowledge graph-powered backend** designed to give AI chat applications long-term memory capabilities. Instead of treating each conversation in isolation, Synapse Cortex:

1. **Ingests** conversational data from chat sessions
2. **Extracts** entities, relationships, and facts using LLMs
3. **Stores** them in a temporal knowledge graph (Neo4j)
4. **Retrieves** relevant context for future conversations
5. **Visualizes** the knowledge graph for user exploration and debugging

The system is built on [Graphiti](https://github.com/getzep/graphiti), a temporal knowledge graph framework that handles entity resolution, relationship extraction, and temporal invalidation of outdated information.

---

## Core Features

### 1. 🧠 Knowledge Graph Ingestion
- **Session Processing**: Converts chat sessions into structured knowledge (entities, relationships, facts)
- **Entity Resolution**: Automatically merges duplicate entities (e.g., "Juan", "Juan Gómez", "JG" → single entity)
- **Temporal Awareness**: Tracks when information becomes valid/invalid (e.g., "used to work at X" vs "currently works at Y")
- **Intelligent Filtering**: Uses degree-based filtering to exclude low-confidence entities

### 2. 💬 OpenAI-Compatible Chat Completions
- **Streaming SSE**: Server-Sent Events format matching OpenAI's API
- **Google Gemini Backend**: Uses Gemini models (flash/pro) for generation
- **System Prompt Injection**: Seamlessly injects user knowledge into prompts
- **Async Architecture**: Non-blocking streaming with FastAPI

### 3. 🔍 Smart Context Retrieval (Hydration)
- **Two-Phase Compilation**:
  1. **Entity Definitions**: "What these concepts mean for this user"
  2. **Relational Dynamics**: "How these concepts interact over time"
- **Cypher-Optimized**: Direct Neo4j queries bypass Graphiti's abstraction for read performance
- **Connectivity-Based Ranking**: Prioritizes well-connected entities over noise

### 4. 🧩 GraphRAG - Per-Turn Retrieval-Augmented Generation
- **Hybrid Search**: Combines semantic embeddings + BM25 full-text search (RRF fusion) via Graphiti
- **Deduplication**: Only injects edges/nodes not already present in the hydrated base prompt
- **Automatic Gating**: Skips retrieval when the graph fits entirely in the prompt (`is_partial: false`)
- **Zero-LLM Overhead**: No agent or tool-calling loop — deterministic pipeline with ~1s latency

### 5. 🗺️ Knowledge Graph Visualization
- **React-Force-Graph Format**: Nodes and links ready for frontend rendering
- **Real-Time Corrections**: Natural language memory edits via Graphiti's episode pipeline
- **Temporal Filtering**: Only shows valid (non-expired) relationships

### 6. 📤 Notion Export
- **Graph-to-Notion Pipeline**: Exports a user's knowledge graph into structured Notion databases with a summary page
- **Dynamic Schema Design**: Gemini analyzes the graph and designs 3-10 category databases with optimal column schemas
- **MCP Agent Integration**: Uses the Notion MCP server via `create_react_agent` for flexible row creation and page building
- **Async with Polling**: Fire-and-forget pattern (202 + status polling) with step-level progress tracking
- **Clean Page on Export**: Automatically removes all existing content under the parent page before each export
- **Feedback Loop Columns**: Every database includes "Needs Review" (checkbox) and "Correction Notes" (rich_text) for corrections
- **Per-Request Auth**: Notion token is passed per-request (not server-side) for multi-tenant use

### 7. 🔄 Notion Correction Import
- **Feedback Loop**: Reads user corrections from exported Notion databases and applies them back to the knowledge graph
- **Smart Row Updates**: MCP agent intelligently updates affected columns or deletes rows that are no longer relevant
- **Graphiti Integration**: Uses `add_episode()` with `custom_extraction_instructions` to ensure corrections are language-consistent and properly contradict outdated edges
- **Partial Failure Handling**: Individual correction failures don't block the pipeline; detailed failure reports included in the result

### 8. 🔐 Security & Rate Limiting
- **API Key Authentication**: All endpoints (except `/health`) require `X-API-SECRET` header
- **Concurrency Control**: Configurable semaphore limits to avoid LLM rate limits (429 errors)
- **CORS Middleware**: Supports cross-origin requests for web frontends

---

## Technical Architecture

### System Overview

```
┌────────────────────────────────────────────────────────────────────┐
│                         CLIENT APPLICATION                          │
│                     (Synapse AI Chat Frontend)                      │
└───────────────────────────┬────────────────────────────────────────┘
                            │ REST API (JSON/SSE)
                            ▼
┌────────────────────────────────────────────────────────────────────┐
│                         SYNAPSE CORTEX API                          │
│                          (FastAPI + Uvicorn)                        │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐          │
│  │  Ingestion    │  │   Hydration   │  │  Generation   │          │
│  │   Service     │  │    Service    │  │    Service    │          │
│  └───────┬───────┘  └───────┬───────┘  └───────┬───────┘          │
│          │                  │                  │                   │
│          └──────────────────┼──────────────────┘                   │
│                             │                                      │
│  ┌──────────────────────────┴───────────────────────────┐          │
│  │              GRAPHITI CORE LAYER                     │          │
│  │  (Entity Resolution, Relationship Extraction,        │          │
│  │   Temporal Management, Embedding Search)             │          │
│  └──────────────────────────┬───────────────────────────┘          │
│                             │                                      │
└─────────────────────────────┼──────────────────────────────────────┘
                              │
                              ▼
         ┌────────────────────────────────────────┐
         │         NEO4J GRAPH DATABASE           │
         │  (Knowledge Graph Storage + Vectors)   │
         └────────────────────────────────────────┘
                              ▲
                              │
         ┌────────────────────────────────────────┐
         │      GOOGLE GEMINI API (External)      │
         │  - LLM: gemini-3-flash-preview         │
         │  - Embeddings: gemini-embedding-001    │
         │  - Reranker: gemini-3-flash-preview    │
         └────────────────────────────────────────┘
```

### Deployment Architectures

#### Production (Docker Compose + Caddy)
```
┌─────────────────────────────────────────────────────────────────┐
│                        DIGITAL OCEAN DROPLET                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Internet (HTTPS)                                               │
│        │                                                         │
│        ▼                                                         │
│   ┌──────────────┐                                               │
│   │    Caddy     │  Automatic SSL (Let's Encrypt)               │
│   │  (Port 443)  │  Reverse Proxy + HTTPS Termination           │
│   └──────┬───────┘                                               │
│          │                                                       │
│          ▼                                                       │
│   ┌──────────────┐                                               │
│   │   FastAPI    │  Python 3.12 + Uvicorn                       │
│   │  (Port 8000) │  API Logic + Graphiti Integration            │
│   └──────┬───────┘                                               │
│          │                                                       │
│          ▼                                                       │
│   ┌──────────────┐                                               │
│   │    Neo4j     │  Graph DB + Vector Index                     │
│   │  (Port 7687) │  Persistent Volume (neo4j_data)              │
│   └──────────────┘                                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### Local Development
```
┌─────────────────────────────────────────────────────────────────┐
│                       DEVELOPER MACHINE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   localhost:8000 ──▶ uvicorn (--reload)                          │
│                                  │                               │
│                                  ▼                               │
│                          FastAPI App (Native)                    │
│                                  │                               │
│                                  ▼                               │
│                          Docker: Neo4j (7687)                    │
│                          Browser UI: http://localhost:7474       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Backend Components

### 1. **Ingestion Service** (`app/services/ingestion.py`)
**Purpose**: Process chat sessions into the knowledge graph (async fire-and-forget)

**Key Functions**:
- `accept_session()`: Validates, creates job entry, launches background task, returns 202 immediately
- `_process_background()`: Runs `graphiti.add_episode()` asynchronously; updates job store on completion
- Validates sessions (minimum message count, character threshold)
- Formats messages into Graphiti episode format

**Data Flow**:
```
POST /ingest → Validation → Create Job → 202 Accepted
                    │
                    └─▶ Background: Graphiti.add_episode() → Update job store

GET /ingest/status/{jobId} → Poll until completed → Hydrate on-demand from Neo4j → Return compilation
```

**Validation Rules**:
- Minimum 1 message
- Minimum 5 total characters

**Job Store** (`app/services/job_store.py`): In-memory dict tracks job status. Compilation is fetched from Neo4j on-demand when status is `"completed"`, then the job is cleaned from memory. Requires `WEB_CONCURRENCY=1` (single worker).

### 2. **Hydration Service** (`app/services/hydration.py`)
**Purpose**: Build user knowledge compilations from the graph

Supports two strategies selected via `version` parameter: **V1** (full dump) and **V2** (budget-aware).

**Architecture (shared)**:
- **Direct Cypher Queries**: Bypasses Graphiti for read performance
- **Degree-Based Filtering**: Only includes entities with >= 2 connections (default)
- **Two-Phase Output**:
  1. **Conceptual Definitions**: Entity summaries ordered by connectivity
  2. **Relational Dynamics**: Relationships with facts, timestamps, and temporal context

**Output Format**:
```markdown
#### 1. CONCEPTUAL DEFINITIONS & IDENTITY ####
- **Entity Name**: Summary of what this entity represents
...

#### 2. RELATIONAL DYNAMICS & CAUSALITY ####
- Entity1 relates to Entity2: "fact describing relationship" [since: 2026-02-17]
...

### STATS ###
Definitions: X | Relations: Y | Est. Tokens: ~Z
```

#### V2: Budget-Aware Compilation (`app/services/hydration_v2.py`)

V1 dumps everything into the system prompt with no size control. V2 introduces a **cascading waterfill allocation** that maximizes context within a character budget (~120k chars / ~30k tokens), ensuring the compilation never blows up the context window.

**How it works:**

1. **Quality Gates** -- Only entities with degree >= 2 and temporally valid edges are fetched (same as V1).

2. **Fast Path** -- If the total formatted text fits within the budget, return everything with `is_partial: false`.

3. **Waterfill Allocation** -- When the budget is exceeded:

```
Budget: 120,000 chars
├── Block A: Nodes (40% = 48,000 chars)
│   Iterate by degree DESC, include atomically until budget exceeded.
│   Unused chars roll over to Block B.
│
└── Block B: Edges (60% + rollover)
    ├── P1: Hub-to-Hub (both nodes in top 20% by degree) — almost always included
    ├── P2: Hub + Recency (one hub node, sorted by date DESC)
    └── P3: Long Tail (low-degree nodes, sorted by date DESC, fills remaining budget)
```

**Hub classification** is done entirely in Python using a dictionary built from the node query results -- no extra DB calls. Edges are classified with O(1) lookups against `node_degree_map`.

**Feature flagging**: Pass `"version": "v2"` in the `/hydrate` request body. Defaults to `"v1"`.

**V2 returns metadata** for future GraphRAG deduplication:
```json
{
  "compilationMetadata": {
    "is_partial": true,
    "total_estimated_tokens": 29500,
    "included_node_ids": ["uuid-1", "uuid-2"],
    "included_edge_ids": ["uuid-x", "uuid-y"]
  }
}
```

This lets the frontend persist which nodes/edges are already in the system prompt, so dynamic retrieval only injects what the LLM doesn't already know:
```python
new_edges = [e for e in retrieved_edges if e.id not in metadata.included_edge_ids]
```

### 3. **GraphRAG Service** (`app/services/graph_rag.py`)
**Purpose**: Per-turn retrieval-augmented generation from the knowledge graph

When the hydrated compilation is partial (budget-constrained by V2), GraphRAG runs a hybrid search on every chat turn to retrieve long-tail edges and nodes that didn't make the cut, deduplicates them against what's already in the prompt, and injects the new context before generation.

**Pipeline**:
```
User message → Build query (last 3 messages) → Graphiti hybrid search (semantic + BM25)
  → Deduplicate against compilationMetadata.included_*_ids
  → Format context block → Append to system message → Stream Gemini
```

**Gating logic** (`get_rag_skip_reason`):
| Condition | Action |
|-----------|--------|
| No `user_id` | Skip — no graph to search |
| No `compilationMetadata` | Skip — no dedup info available |
| `is_partial == false` | Skip — full graph already in prompt |
| Otherwise | Run GraphRAG search |

**Key design decisions**:
- **No agent/tool-calling**: Direct deterministic pipeline — avoids the extra LLM round-trip latency that agent frameworks add (~2-5s vs ~1s)
- **Deduplication by UUID**: Uses `included_edge_ids` / `included_node_ids` from V2 metadata to ensure zero redundancy with the base prompt
- **Text sanitization**: Decodes HTML entities and collapses stray newlines in facts/summaries before injection
- **Graceful degradation**: On search failure, proceeds without RAG context (never blocks generation)

**Telemetry** (OTel `rag.*` namespace):
- `rag.enabled`, `rag.skipped_reason`
- `rag.search_duration_ms`, `rag.total_duration_ms`
- `rag.raw_edges_count`, `rag.deduped_edges_count`, `rag.injected_edges_count`
- `rag.raw_nodes_count`, `rag.deduped_nodes_count`, `rag.injected_nodes_count`
- `rag.query_chars`, `rag.context_block_chars`

### 4. **Generation Service** (`app/services/generation.py`)
**Purpose**: OpenAI-compatible streaming chat completions

**Features**:
- **SSE Format**: `data: {json}\n\n` chunks
- **Gemini Integration**: Google GenAI SDK with async streaming
- **System Prompt Handling**: Prepends system messages to first user message (Gemini requirement)
- **Error Handling**: Graceful error streaming with error chunks

**Stream Format**:
```
data: {"id": "chatcmpl-...", "choices": [{"delta": {"role": "assistant"}, ...}]}

data: {"id": "chatcmpl-...", "choices": [{"delta": {"content": "Hello"}, ...}]}

data: {"id": "chatcmpl-...", "choices": [{"delta": {}, "finish_reason": "stop"}]}

data: [DONE]
```

### 5. **Graph Service** (`app/services/graph.py`)
**Purpose**: Knowledge graph visualization and memory correction

**Key Functions**:
- **Get Graph**: Retrieves nodes and links in react-force-graph format
- **Correct Memory**: Applies natural language corrections via Graphiti's episode pipeline

**Correction Strategy**:
Instead of direct CRUD operations (which break embeddings), corrections are processed as new episodes. Graphiti automatically:
- Invalidates outdated relationships
- Creates new relationships
- Maintains temporal integrity

**Example**: "I no longer want to apply for the O-1 visa, I decided to stay in Colombia" → Graphiti invalidates old edges and creates new ones

### 6. **Notion Export Service** (`app/services/notion_export.py`)
**Purpose**: Export a user's knowledge graph into structured Notion databases

**Pipeline** (6 sequential steps, each with its own OTel span):

| Step | Name | Method | Description |
|------|------|--------|-------------|
| 0 | Clean Page | Notion SDK (`blocks.children.list` + `DELETE`) | Remove all existing content under the parent page |
| 1 | Hydrate | `HydrationService.build_user_knowledge(v1)` | Build the full graph compilation from Neo4j |
| 2a | Analyze Schemas | Gemini structured output (`SchemaResult`) | Design 3-10 category databases with column schemas |
| 2b | Extract Entries | Gemini structured output (`ExtractionResult`) | Extract rows for each category (one LLM call per category) |
| 3 | Create Databases | Notion SDK (`notion.request()`) | Create one Notion database per category under the parent page |
| 4 | Populate | MCP agent (`create_react_agent`) | Fill databases with extracted rows (batched, 12 rows per agent call) |
| 5 | Summarize | MCP agent | Create a "Knowledge Graph Overview" page with links to all databases |

**MCP Lifecycle**: Steps 4-5 spawn a Node.js subprocess (`npx @notionhq/notion-mcp-server`) that communicates with Python via stdin/stdout JSON-RPC. The `async with stdio_client(...)` context manager handles startup, communication, and cleanup. A semaphore (`max_concurrent_exports=3`) limits concurrent subprocesses.

**Job Store** (`app/services/notion_export_job_store.py`): Tracks `current_step`, `categories_count`, `entries_count`, `database_ids`, and `summary_page_url`. Same in-memory pattern as the ingest job store.

### 7. **Notion Correction Service** (`app/services/notion_correction.py`)
**Purpose**: Read user corrections from Notion and apply them back to the knowledge graph

**Pipeline** (3 steps, each with its own OTel span):

| Step | Name | Method | Description |
|------|------|--------|-------------|
| 0 | Discover Databases | Notion SDK (`blocks.children.list`) | Find all child databases under the parent page |
| 1 | Scan Flagged Rows | Notion SDK (`databases/{id}/query`) | Query each database for rows with "Needs Review" checked |
| 2a | Correct Graph | `graphiti.add_episode()` | Apply correction with `custom_extraction_instructions` for language + contradiction handling |
| 2b | Update Notion Row | MCP agent (`create_react_agent`) | Intelligently update affected columns or delete the row if no longer relevant |

**Correction Strategy**: Each correction is processed as a new Graphiti episode. The `custom_extraction_instructions` steer the LLM to extract only corrected facts (not the old state), and to write everything in the user's preferred language. Graphiti's built-in contradiction detection automatically invalidates outdated edges.

**MCP Agent Decision**: The agent receives the full context (current properties, column schema, updated node summaries, new facts, invalidated facts) and chooses to either update the row or archive it.

**Job Store** (`app/services/notion_correction_job_store.py`): Tracks `current_step`, `databases_scanned`, `corrections_found`, `corrections_applied`, `corrections_failed`, and `failed_corrections` (per-row error details).

---

## API Endpoints

### Authentication
All endpoints (except `/health`) require an `X-API-SECRET` header matching `SYNAPSE_API_SECRET` environment variable.

### Endpoints Reference

#### 🟢 `GET /health`
**Purpose**: Health check for load balancers and monitoring  
**Auth**: ❌ None required  
**Response**:
```json
{
  "status": "ok",
  "service": "synapse-cortex"
}
```

---

#### 📥 `POST /ingest`
**Purpose**: Accept a chat session for async processing (fire-and-forget)  
**Auth**: ✅ Required (`X-API-SECRET`)  
**Status**: `202 Accepted`  
**Request Body**:
```json
{
  "jobId": "convex-queue-id-abc123",
  "userId": "user-123",
  "sessionId": "session-abc",
  "messages": [
    {
      "role": "user",
      "content": "I'm planning to move to Spain next year",
      "timestamp": 1704067200000
    },
    {
      "role": "assistant",
      "content": "That's exciting! What's driving this decision?",
      "timestamp": 1704067205000
    }
  ],
  "metadata": {
    "sessionStartedAt": 1704067200000,
    "sessionEndedAt": 1704067300000,
    "messageCount": 2
  }
}
```

**Response** (202):
```json
{
  "jobId": "convex-queue-id-abc123",
  "status": "processing"
}
```

**Skipped** (insufficient messages): returns immediately with `status: "skipped"` and `userKnowledgeCompilation`.

**Duplicate submit** (same `jobId`): returns current status without re-processing.

**Client Flow**: Poll `GET /ingest/status/{jobId}` until `status` is `"completed"` or `"failed"`.

---

#### 📊 `GET /ingest/status/{job_id}`
**Purpose**: Poll for ingest job status and retrieve result when completed  
**Auth**: ✅ Required (`X-API-SECRET`)  

**Response** (processing):
```json
{
  "jobId": "convex-queue-id-abc123",
  "status": "processing"
}
```

**Response** (completed):
```json
{
  "jobId": "convex-queue-id-abc123",
  "status": "completed",
  "userKnowledgeCompilation": "#### 1. CONCEPTUAL DEFINITIONS & IDENTITY ####\n- **Spain**: Country user plans to move to...",
  "metadata": {
    "model": "gemini-3-flash-preview",
    "processing_time_ms": 15000.5,
    "nodes_extracted": 8,
    "edges_extracted": 12,
    "episode_id": "uuid-..."
  }
}
```

**Response** (failed):
```json
{
  "jobId": "convex-queue-id-abc123",
  "status": "failed",
  "error": "Error message",
  "code": "GRAPH_PROCESSING_ERROR"
}
```

**404**: Job not found (never submitted, already cleaned up, or backend restarted).

**Note**: When returning a terminal state (`"completed"` or `"failed"`), the job is removed from memory. Compilation is hydrated from Neo4j on-demand for completed jobs.

---

#### 💧 `POST /hydrate`
**Purpose**: Fetch current user knowledge compilation without processing new data  
**Auth**: ✅ Required  
**Request Body**:
```json
{
  "userId": "user-123",
  "version": "v2"
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `userId` | string | - | Required. User/group ID in the graph |
| `version` | `"v1"` \| `"v2"` | `"v1"` | Hydration strategy. V2 uses budget-aware waterfill allocation |

**Response (V1)**:
```json
{
  "success": true,
  "userKnowledgeCompilation": "#### 1. CONCEPTUAL DEFINITIONS & IDENTITY ####\n..."
}
```

**Response (V2)** -- includes `compilationMetadata` for GraphRAG deduplication:
```json
{
  "success": true,
  "userKnowledgeCompilation": "#### 1. CONCEPTUAL DEFINITIONS & IDENTITY ####\n...",
  "compilationMetadata": {
    "is_partial": true,
    "total_estimated_tokens": 29500,
    "included_node_ids": ["uuid-1", "uuid-2"],
    "included_edge_ids": ["uuid-x", "uuid-y"]
  }
}
```

**Use Cases**:
- Debugging current graph state
- Fetching context without re-indexing
- Feature-flagging V1 vs V2 compilation from the frontend

---

#### 💬 `POST /v1/chat/completions`
**Purpose**: OpenAI-compatible streaming chat completions  
**Auth**: ✅ Required  
**Request Body**:
```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "What's the weather like today?"
    }
  ],
  "model": "gemini-3-flash-preview",
  "stream": true
}
```

**Response**: Server-Sent Events (SSE)
```
data: {"id":"chatcmpl-abc123","choices":[{"delta":{"role":"assistant"},...}]}

data: {"id":"chatcmpl-abc123","choices":[{"delta":{"content":"Today"},...}]}

data: [DONE]
```

---

#### 🗺️ `GET /v1/graph/{group_id}`
**Purpose**: Retrieve knowledge graph in react-force-graph format  
**Auth**: ✅ Required  
**Response**:
```json
{
  "nodes": [
    {
      "id": "uuid-1234",
      "name": "Spain",
      "val": 5,
      "summary": "Country user plans to relocate to in 2025"
    }
  ],
  "links": [
    {
      "source": "uuid-1234",
      "target": "uuid-5678",
      "label": "RELATES_TO",
      "fact": "User plans to move to Spain for work opportunities"
    }
  ]
}
```

**Features**:
- Node `val` = connection count (controls visual sizing)
- Only returns valid (non-expired) relationships
- Excludes episodic nodes

---

#### ✏️ `POST /v1/graph/correction`
**Purpose**: Apply natural language memory corrections  
**Auth**: ✅ Required  
**Request Body**:
```json
{
  "group_id": "user-123",
  "correction_text": "I've changed my mind. I no longer want to move to Spain. I'm staying in Colombia."
}
```

**Response**:
```json
{
  "success": true
}
```

**How It Works**:
- Correction text is processed as a new Graphiti episode
- Graphiti automatically invalidates outdated edges
- Creates new edges reflecting the correction
- Preserves embeddings and temporal integrity

---

#### 📤 `POST /v1/notion/export`
**Purpose**: Export a user's knowledge graph into Notion databases (async)
**Auth**: ✅ Required (`X-API-SECRET`)
**Status**: `202 Accepted`
**Request Body**:
```json
{
  "userId": "user-123",
  "notionToken": "ntn_your_notion_integration_secret",
  "pageName": "Synapse",
  "language": "English"
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `userId` | string | - | Required. User/group ID whose graph to export |
| `notionToken` | string | - | Required. Notion internal integration secret |
| `pageName` | string | - | Required. Name of the parent Notion page (must be shared with the integration) |
| `language` | string | `"English"` | Output language for all generated Notion content |

**Response** (202):
```json
{
  "jobId": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "processing",
  "pageId": "12345678-abcd-1234-abcd-123456789abc"
}
```

The Notion token and page name are validated synchronously before returning 202. If the token is invalid or the page is not found, you get a 400 error immediately.

**Client Flow**: Poll `GET /v1/notion/export/status/{jobId}` for progress and results.

---

#### 📊 `GET /v1/notion/export/status/{job_id}`
**Purpose**: Poll for Notion export job status and retrieve result when completed
**Auth**: ✅ Required (`X-API-SECRET`)

**Response** (processing):
```json
{
  "jobId": "a1b2c3d4-...",
  "status": "processing",
  "progress": {
    "currentStep": "populating",
    "categoriesDesigned": 6,
    "entriesExtracted": 42
  }
}
```

Pipeline steps reported via `currentStep`: `"hydrating"` → `"analyzing"` → `"extracting_entries"` → `"creating_databases"` → `"populating"` → `"summarizing"` → `"done"`.

**Response** (completed):
```json
{
  "jobId": "a1b2c3d4-...",
  "status": "completed",
  "result": {
    "databaseIds": {
      "People": "db-id-1",
      "Health & Medications": "db-id-2",
      "Projects & Goals": "db-id-3"
    },
    "summaryPageUrl": "https://notion.so/abc123...",
    "categoriesCount": 6,
    "entriesCount": 42,
    "durationMs": 185000.5
  }
}
```

**Response** (failed):
```json
{
  "jobId": "a1b2c3d4-...",
  "status": "failed",
  "error": "Error message",
  "code": "UPSTREAM_TIMEOUT"
}
```

**404**: Job not found (never submitted, already consumed, or backend restarted).

**Note**: Terminal states (`"completed"` or `"failed"`) remove the job from memory after the response is returned.

---

#### 🔄 `POST /v1/notion/corrections`
**Purpose**: Import user corrections from Notion databases back into the knowledge graph
**Auth**: ✅ Required (`X-API-SECRET`)
**Status**: `202 Accepted`
**Request Body**:
```json
{
  "userId": "user-123",
  "notionToken": "ntn_your_token_here",
  "pageName": "Synapse",
  "language": "Spanish"
}
```

**Response**:
```json
{
  "jobId": "d1e2f3a4-...",
  "status": "processing",
  "pageId": "resolved-page-id"
}
```

**Client Flow**: Poll `GET /v1/notion/corrections/status/{jobId}` for progress and results.

---

#### 📊 `GET /v1/notion/corrections/status/{job_id}`
**Purpose**: Poll for Notion correction import job status
**Auth**: ✅ Required (`X-API-SECRET`)

**Response** (processing):
```json
{
  "jobId": "d1e2f3a4-...",
  "status": "processing",
  "progress": {
    "currentStep": "applying",
    "databasesScanned": 7,
    "correctionsFound": 3,
    "correctionsApplied": 1,
    "correctionsFailed": 0
  }
}
```

Pipeline steps reported via `currentStep`: `"scanning"` → `"applying"` → `"done"`.

**Response** (completed):
```json
{
  "jobId": "d1e2f3a4-...",
  "status": "completed",
  "result": {
    "correctionsFound": 3,
    "correctionsApplied": 2,
    "correctionsFailed": 1,
    "failedCorrections": [
      {"category": "Medications", "title": "Aspirin", "error": "LLM rate limit exceeded"}
    ],
    "durationMs": 95000.5
  }
}
```

**Response** (failed):
```json
{
  "jobId": "d1e2f3a4-...",
  "status": "failed",
  "error": "No databases found under the specified Notion page.",
  "code": "NO_DATABASES"
}
```

**Note**: Terminal states (`"completed"` or `"failed"`) remove the job from memory after the response is returned.

---

## Notion Export

### Overview

The Notion Export feature lets you export a user's entire knowledge graph into structured, browsable Notion databases. It reads the graph compilation from Neo4j, uses Gemini to dynamically design database schemas and extract entries, then creates everything in Notion under a parent page you specify.

### How It Works

```
1. CLIENT
   └─▶ POST /v1/notion/export
       {userId, notionToken, pageName, language}

2. ROUTE HANDLER (synchronous validation)
   ├─▶ Validate notionToken by resolving pageName → pageId
   ├─▶ Create job in memory store (status: "processing")
   ├─▶ Launch background task (asyncio.create_task)
   └─▶ Return 202 {jobId, pageId}

3. BACKGROUND PIPELINE (NotionExportService)
   │
   ├─▶ Step 0: CLEAN PAGE
   │   └─▶ List all child blocks under the parent page
   │       └─▶ Delete each block (databases, summaries, etc.)
   │           → Ensures a fresh page for every export
   │
   ├─▶ Step 1: HYDRATE
   │   └─▶ HydrationService.build_user_knowledge(userId, v1)
   │       → Full graph compilation text from Neo4j
   │
   ├─▶ Step 2a: DESIGN SCHEMAS
   │   └─▶ Gemini structured output (SchemaResult)
   │       → 3-10 categories with column definitions
   │
   ├─▶ Step 2b: EXTRACT ENTRIES
   │   └─▶ Gemini structured output (ExtractionResult) × N categories
   │       → All rows for each category
   │
   ├─▶ Step 3: CREATE DATABASES
   │   └─▶ Notion SDK (notion.request) × N categories
   │       → One database per category under the parent page
   │       → Each includes "Needs Review" + "Correction Notes" columns
   │
   ├─▶ Step 4: POPULATE (MCP agent)
   │   └─▶ npx @notionhq/notion-mcp-server (stdio subprocess)
   │       └─▶ ReAct agent creates rows via API-post-page
   │           (batched, 12 rows per agent call)
   │
   └─▶ Step 5: SUMMARIZE (MCP agent)
       └─▶ ReAct agent creates "Knowledge Graph Overview" page
           with overview text, database links, and feedback instructions

4. CLIENT (polling loop)
   └─▶ GET /v1/notion/export/status/{jobId}
       ├─▶ {status: "processing", progress: {currentStep, ...}} → poll again
       ├─▶ {status: "completed", result: {databaseIds, summaryPageUrl, ...}}
       └─▶ {status: "failed", error, code}
```

### Notion Setup

1. **Create an internal integration** at https://www.notion.so/profile/integrations
2. Copy the integration secret (starts with `ntn_`)
3. Create a parent page in Notion (e.g. "Synapse")
4. Share the page with your integration (page menu → "Connect to" → your integration)
5. Use the page name as the `pageName` parameter in the API request

### Example: Full Export Flow

```bash
# 1. Start the export
curl -X POST http://localhost:8000/v1/notion/export \
  -H "Content-Type: application/json" \
  -H "X-API-SECRET: your_secret" \
  -d '{
    "userId": "user-123",
    "notionToken": "ntn_your_token_here",
    "pageName": "Synapse",
    "language": "English"
  }'
# → 202 {"jobId": "abc-123", "status": "processing", "pageId": "..."}

# 2. Poll for status (repeat until completed/failed)
curl http://localhost:8000/v1/notion/export/status/abc-123 \
  -H "X-API-SECRET: your_secret"
# → {"jobId": "abc-123", "status": "processing", "progress": {"currentStep": "populating", ...}}

# 3. Final result
# → {"jobId": "abc-123", "status": "completed", "result": {"databaseIds": {...}, "summaryPageUrl": "https://notion.so/...", ...}}
```

### Prerequisites

The Notion export pipeline spawns a Node.js MCP subprocess. The server environment needs:
- **Node.js** (v18+) and **npx** available on `PATH`
- Network access to `registry.npmjs.org` (first run downloads `@notionhq/notion-mcp-server`)

### Axiom Observability

The pipeline emits spans under the `export.*` attribute namespace:

```apl
['synapse-cortex-traces']
| where startswith(name, 'notion_export.')
| summarize
    total = count(),
    avg_duration = avg(['attributes.export.duration_ms']),
    avg_categories = avg(['attributes.export.categories_count']),
    avg_entries = avg(['attributes.export.entries_count']),
    failed = countif(['attributes.operation.status'] == 'failed')
  by bin_auto(_time)
```

Per-step latency breakdown:

```apl
['synapse-cortex-traces']
| where startswith(name, 'notion_export.')
| summarize
    avg_ms = avg(['attributes.duration_ms']),
    p95_ms = percentile(['attributes.duration_ms'], 95),
    calls = count()
  by name
| order by avg_ms desc
```

---

## Notion Correction Import

### Overview

The Notion Correction Import feature closes the feedback loop: users flag incorrect data in the exported Notion databases, and the system reads those corrections, applies them to the knowledge graph, and intelligently updates or deletes the Notion rows.

Each exported database includes two feedback columns:
- **Needs Review** (checkbox): Flag a row that needs correction
- **Correction Notes** (rich_text): Describe what needs to be fixed

### How It Works

```
1. CLIENT
   └─▶ POST /v1/notion/corrections
       {userId, notionToken, pageName, language}

2. ROUTE HANDLER (synchronous validation)
   ├─▶ Validate notionToken by resolving pageName → pageId
   ├─▶ Create job in memory store (status: "processing")
   ├─▶ Launch background task (asyncio.create_task)
   └─▶ Return 202 {jobId, pageId}

3. BACKGROUND PIPELINE (NotionCorrectionService)
   │
   ├─▶ Step 0: DISCOVER DATABASES
   │   └─▶ List child_database blocks under the parent page
   │       → Map of category name → database ID
   │
   ├─▶ Step 1: SCAN FOR FLAGGED ROWS
   │   └─▶ Query each database filtering "Needs Review" == true
   │       └─▶ Extract property values, types, and correction notes
   │           → List of CorrectionItems
   │
   └─▶ Step 2: APPLY CORRECTIONS (per row, sequentially)
       │
       ├─▶ 2a. CORRECT GRAPH (Graphiti)
       │   └─▶ graphiti.add_episode() with:
       │       - Episode body containing old properties + user correction
       │       - custom_extraction_instructions for language + correction handling
       │       → Returns AddEpisodeResults (updated nodes, edges, invalidated facts)
       │
       └─▶ 2b. UPDATE NOTION ROW (MCP agent)
           └─▶ LangGraph ReAct agent with Notion MCP tools decides:
               - OPTION A: Update row (patch affected properties, uncheck flag)
               - OPTION B: Delete row (archive if entity is no longer relevant)
               Agent receives: current properties, column schema, node summaries,
               new facts, invalidated facts, and user correction notes.

4. CLIENT (polling loop)
   └─▶ GET /v1/notion/corrections/status/{jobId}
       ├─▶ {status: "processing", progress: {currentStep, correctionsApplied, ...}}
       ├─▶ {status: "completed", result: {correctionsFound, correctionsApplied, correctionsFailed, ...}}
       └─▶ {status: "failed", error, code}
```

### Example: Full Correction Flow

```bash
# 1. Start the correction import
curl -X POST http://localhost:8000/v1/notion/corrections \
  -H "Content-Type: application/json" \
  -H "X-API-SECRET: your_secret" \
  -d '{
    "userId": "user-123",
    "notionToken": "ntn_your_token_here",
    "pageName": "Synapse",
    "language": "Spanish"
  }'
# → 202 {"jobId": "def-456", "status": "processing", "pageId": "..."}

# 2. Poll for status
curl http://localhost:8000/v1/notion/corrections/status/def-456 \
  -H "X-API-SECRET: your_secret"
# → {"status": "processing", "progress": {"currentStep": "applying", "correctionsApplied": 1, ...}}

# 3. Final result
# → {"status": "completed", "result": {"correctionsFound": 3, "correctionsApplied": 2, "correctionsFailed": 1, ...}}
```

### How the MCP Agent Decides

The agent receives the full context of each correction and makes an intelligent decision:

- **Update**: If the correction modifies specific facts (e.g., "the dosage changed to 20mg"), the agent patches the relevant columns while keeping unaffected properties intact
- **Delete**: If the correction invalidates the entity entirely (e.g., "this concept is no longer relevant"), the agent archives the row

The `language` parameter ensures all updated property values are written in the user's preferred language.

### Axiom Observability

The correction pipeline emits spans under the `correction.*` attribute namespace:

```apl
['synapse-cortex-traces']
| where startswith(name, 'notion_correction.')
| summarize
    total = count(),
    avg_duration = avg(['attributes.duration_ms']),
    avg_found = avg(['attributes.correction.found']),
    avg_applied = avg(['attributes.correction.applied']),
    avg_failed = avg(['attributes.correction.failed'])
  by bin_auto(_time)
```

Per-step breakdown (graph correction vs Notion row update):

```apl
['synapse-cortex-traces']
| where startswith(name, 'notion_correction.')
| summarize
    avg_ms = avg(['attributes.duration_ms']),
    p95_ms = percentile(['attributes.duration_ms'], 95),
    calls = count()
  by name
| order by avg_ms desc
```

---

## Data Flow

### Ingestion Pipeline (Async with Polling)

```
1. CLIENT (Convex)
   └─▶ POST /ingest
       {jobId, userId, sessionId, messages, metadata}

2. INGESTION SERVICE (accept_session)
   ├─▶ Validate session (min messages, min chars)
   ├─▶ If insufficient: return 202 {status: "skipped", userKnowledgeCompilation} (hydrate immediately)
   ├─▶ If duplicate jobId: return 202 {status: "processing"} (no re-process)
   ├─▶ Create job in memory store (status: "processing")
   ├─▶ asyncio.create_task(_process_background)
   └─▶ Return 202 {jobId, status: "processing"}

3. BACKGROUND TASK (_process_background)
   └─▶ GRAPHITI CORE
       ├─▶ Extract entities (e.g., "hiking", "User")
       ├─▶ Extract relationships (e.g., User ENJOYS hiking)
       ├─▶ Generate embeddings (gemini-embedding-001)
       ├─▶ Search for existing similar entities
       ├─▶ Merge duplicates (entity resolution)
       ├─▶ Rerank candidates (Gemini reranker)
       └─▶ Write to Neo4j (nodes + edges + timestamps)
   └─▶ Update job store: status "completed" + metadata (no compilation stored)

4. CLIENT (Polling loop, e.g. exponential backoff: 2m, 5m, 10m, 30m)
   └─▶ GET /ingest/status/{jobId}
       ├─▶ 404: job not found → re-submit POST /ingest
       ├─▶ {status: "processing"} → poll again
       └─▶ {status: "completed"} → HYDRATION SERVICE
           ├─▶ Hydrate on-demand from Neo4j (~1-2s)
           ├─▶ Remove job from memory
           └─▶ Return full result to client
```

### Chat Completion Pipeline

```
1. CLIENT
   └─▶ POST /v1/chat/completions
       {messages, model, stream: true, user_id, compilationMetadata}

2. GRAPH RAG (conditional)
   ├─▶ Gate check: user_id present? compilationMetadata.is_partial == true?
   ├─▶ Build search query from last 3 messages
   ├─▶ Graphiti hybrid search (semantic + BM25, limit=10)
   ├─▶ Deduplicate against included_edge_ids / included_node_ids
   └─▶ Append context block to system message

3. GENERATION SERVICE
   ├─▶ Convert messages to Gemini format
   │   - Extract system prompt (now includes RAG context)
   │   - Prepend to first user message
   │   - Map roles: user/assistant → user/model
   └─▶ Call gemini.generate_content_stream()

4. GEMINI API
   └─▶ Stream token chunks

5. GENERATION SERVICE
   ├─▶ Wrap chunks in OpenAI SSE format
   │   data: {"choices": [{"delta": {"content": "..."}}]}
   ├─▶ Include RAG stats in final usage chunk (rag_enabled, rag_edges, etc.)
   └─▶ Send [DONE] marker

6. CLIENT
   └─▶ Receives streamed response
```

### Memory Correction Pipeline

```
1. CLIENT
   └─▶ POST /v1/graph/correction
       {group_id, correction_text}

2. GRAPH SERVICE
   └─▶ Call graphiti.add_episode(correction_text)

3. GRAPHITI CORE
   ├─▶ Process correction as new episode
   ├─▶ Extract new facts/relationships
   ├─▶ Search for conflicting edges
   ├─▶ Set invalid_at timestamp on outdated edges
   └─▶ Create new edges with valid_at = now

4. NEO4J
   └─▶ Graph updated with temporal integrity

5. CLIENT
   └─▶ Next hydration/graph fetch reflects changes
```

---

## Technology Stack

### Core Framework
- **FastAPI**: Modern async web framework with automatic OpenAPI docs
- **Uvicorn**: ASGI server with WebSocket and SSE support
- **Pydantic**: Data validation and settings management

### Knowledge Graph
- **Neo4j**: Graph database with vector search capabilities
- **Graphiti Core**: Temporal knowledge graph framework with entity resolution
- **Google Gemini**: LLM for entity extraction, embeddings, and reranking
  - **LLM Model**: `gemini-3-flash-preview` (configurable)
  - **Embedding Model**: `gemini-embedding-001` (3072 dimensions)
  - **Reranker Model**: `gemini-3-flash-preview`

### Deployment
- **Docker & Docker Compose**: Containerization and orchestration
- **Caddy**: Automatic HTTPS/SSL with Let's Encrypt
- **Digital Ocean**: Cloud hosting (production environment)

### Dependencies
```txt
fastapi>=0.109.0
uvicorn[standard]>=0.27.0
neo4j>=5.17.0
graphiti-core[google-genai]>=0.27.1
google-genai>=1.0.0
pydantic-settings>=2.1.0
python-dotenv>=1.0.0
sse-starlette>=1.8.0
```

---

## Observability in Axiom

The API now emits structured OpenTelemetry attributes for request-level and service-level debugging in Axiom.

### Attribute namespaces

- `chat.*`: completions, token usage, stream metrics, upstream Gemini details
- `rag.*`: GraphRAG gating, search latency, edge/node counts, dedup stats
- `ingest.*`: job lifecycle, counts, processing metadata
- `hydrate.*`: hydration request context and output size
- `export.*`: Notion export pipeline steps, categories, entries, database counts
- `correction.*`: Notion correction import pipeline, corrections found/applied/failed
- `graph.*`: graph retrieval/correction context and result counts
- `db.*`: Neo4j query type, records returned, query latency
- `upstream.*`: upstream status/error hints for Gemini and HTTP calls
- `error.*`: normalized category/code/type/message for failures

### Query ideas (Axiom)

Error rate by endpoint:

```apl
['synapse-cortex-traces']
| where ['attributes.http.route'] != ''
| summarize
    total = count(),
    failed = countif(['attributes.operation.status'] == 'failed'),
    error_rate = (countif(['attributes.operation.status'] == 'failed') * 100.0 / count())
  by route = ['attributes.http.route']
| order by error_rate desc
```

Chat completion performance + token usage:

```apl
['synapse-cortex-traces']
| where name == 'chat.completion.stream'
| summarize
    requests = count(),
    p95_ms = percentile(['attributes.chat.total_duration_ms'], 95),
    avg_total_tokens = avg(['attributes.chat.tokens.total']),
    avg_prompt_tokens = avg(['attributes.chat.tokens.prompt']),
    avg_completion_tokens = avg(['attributes.chat.tokens.completion'])
  by model = ['attributes.chat.model']
| order by requests desc
```

Gemini/upstream failures by category/status:

```apl
['synapse-cortex-traces']
| where ['attributes.upstream.error_type'] != ''
| summarize
    failures = count()
  by category = ['attributes.error.category'],
     code = ['attributes.error.code'],
     status = ['attributes.upstream.status_code']
| order by failures desc
```

GraphRAG retrieval performance:

```apl
['synapse-cortex-traces']
| where ['attributes.rag.enabled'] == true
| summarize
    requests = count(),
    p95_search_ms = percentile(['attributes.rag.search_duration_ms'], 95),
    avg_injected_edges = avg(['attributes.rag.injected_edges_count']),
    avg_injected_nodes = avg(['attributes.rag.injected_nodes_count']),
    avg_deduped = avg(['attributes.rag.deduped_edges_count'] + ['attributes.rag.deduped_nodes_count']),
    zero_injection_rate = (countif(['attributes.rag.injected_edges_count'] + ['attributes.rag.injected_nodes_count'] == 0) * 100.0 / count())
| order by requests desc
```

Slow Neo4j queries:

```apl
['synapse-cortex-traces']
| where startswith(name, 'db.cypher.')
| where ['attributes.db.query_duration_ms'] > 500
| project
    timestamp,
    name,
    query_type = ['attributes.db.query_type'],
    duration_ms = ['attributes.db.query_duration_ms'],
    records = ['attributes.db.records_returned']
| order by duration_ms desc
```

---

## Setup & Deployment

### Prerequisites

- **Python**: 3.12+
- **Node.js**: 18+ with `npx` on PATH (required for the Notion export MCP subprocess)
- **Docker**: Latest stable version
- **Docker Compose**: V2+
- **Google Gemini API Key**: [Get one here](https://ai.google.dev/)

### Environment Variables

Create a `.env` file based on `.env.example`:

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `NEO4J_URI` | Neo4j Bolt connection URI | `bolt://localhost:7687` | Yes |
| `NEO4J_USER` | Neo4j username | `neo4j` | Yes |
| `NEO4J_PASSWORD` | Neo4j password | - | Yes |
| `GOOGLE_API_KEY` | Google Gemini API key | - | Yes |
| `GRAPHITI_MODEL` | Gemini model for Graphiti | `gemini-3-flash-preview` | No |
| `SYNAPSE_API_SECRET` | API authentication secret | - | Yes |
| `SEMAPHORE_LIMIT` | Max concurrent LLM operations | `3` | No |

### Local Development

#### 1. Start Neo4j
```bash
docker-compose -f docker-compose.local.yml up -d
```

**Neo4j Access**:
- Browser UI: http://localhost:7474
- Bolt Protocol: `bolt://localhost:7687`
- Credentials: `neo4j` / `localpassword`

#### 2. Install Dependencies
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

#### 3. Configure Environment
```bash
cp .env.example .env
nano .env  # Edit with your credentials
```

#### 4. Run FastAPI
```bash
uvicorn app.main:app --reload
```

API available at: http://localhost:8000

#### 5. Test Endpoints
```bash
# Health check
curl http://localhost:8000/health

# Hydrate (requires X-API-SECRET header)
curl -X POST http://localhost:8000/hydrate \
  -H "Content-Type: application/json" \
  -H "X-API-SECRET: your_secret" \
  -d '{"userId": "test-user"}'
```

#### 6. Rails-style console (optional)
Interactive Python shell with Graphiti and all project services pre-loaded (like `rails c`):

```bash
# From project root, with venv active
python scripts/console.py
```

You get a REPL with `graphiti`, `neo4j_driver`, `graph_service`, `hydration_service`, `ingestion_service`, `generation_service`, and `settings` in scope. Try searches, call services, or run any Python. Install IPython for top-level `await` and a nicer prompt: `pip install ipython`.

**Examples in the console:**
```python
edges = await graphiti.search("what are my preferences?", group_id="user-123")
g = await graph_service.get_graph("user-123")
```

For a minimal search-only loop (no free-form code), use `python scripts/graphiti_repl.py` instead.

#### 7. Stop Services
```bash
# Stop Neo4j
docker-compose -f docker-compose.local.yml down

# Remove data volume (optional)
docker-compose -f docker-compose.local.yml down -v
```

---

### Production Deployment

#### 1. Server Provisioning
- Create a Digital Ocean droplet (Ubuntu 22.04 LTS recommended)
- Configure firewall to allow ports 80, 443, 22
- SSH into the server

#### 2. Clone Repository
```bash
git clone <your-repo-url> synapse-cortex
cd synapse-cortex
```

#### 3. Configure Environment
```bash
cp .env.example .env
nano .env
```

**Production `.env` example**:
```env
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=<strong-random-password>
GOOGLE_API_KEY=<your-gemini-api-key>
SYNAPSE_API_SECRET=<your-api-secret>
GRAPHITI_MODEL=gemini-3-flash-preview
SEMAPHORE_LIMIT=3
```

#### 4. Configure DNS
Create an A record pointing to your droplet's IP:
- **Host**: `synapse-cortex`
- **Value**: Droplet IP address
- **Domain**: `juandago.dev`

Result: `synapse-cortex.juandago.dev` → Droplet IP

#### 5. Deploy Stack
```bash
docker-compose up -d --build
```

This starts:
- **Caddy**: Automatic SSL on port 443
- **FastAPI**: API server on port 8000 (internal)
- **Neo4j**: Graph database on port 7687 (internal)

#### 6. Verify Deployment
```bash
# Check containers
docker-compose ps

# View logs
docker-compose logs -f

# Test API
curl https://synapse-cortex.juandago.dev/health
```

Expected response:
```json
{"status":"ok","service":"synapse-cortex"}
```

#### 7. Update Deployment
```bash
git pull
docker-compose up -d --build
```

#### 8. Monitor Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f api
docker-compose logs -f neo4j
docker-compose logs -f caddy
```

#### 9. Backup Neo4j Data
```bash
# Stop services
docker-compose down

# Backup neo4j_data volume
docker run --rm -v synapse-cortex_neo4j_data:/data -v $(pwd):/backup ubuntu tar czf /backup/neo4j-backup-$(date +%Y%m%d).tar.gz /data

# Restart services
docker-compose up -d
```

#### 10. Restore Neo4j Data
```bash
# Stop services
docker-compose down

# Restore from backup
docker run --rm -v synapse-cortex_neo4j_data:/data -v $(pwd):/backup ubuntu tar xzf /backup/neo4j-backup-YYYYMMDD.tar.gz -C /

docker run --rm \
  -v $(pwd):/backups \
  -v synapse-cortex_neo4j_data:/data \
  neo4j:5-community \
  neo4j-admin database load neo4j --from-path=/backups --overwrite-destination=true


# Restart services
docker-compose up -d
```

