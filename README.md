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

### 4. 🗺️ Knowledge Graph Visualization
- **React-Force-Graph Format**: Nodes and links ready for frontend rendering
- **Real-Time Corrections**: Natural language memory edits via Graphiti's episode pipeline
- **Temporal Filtering**: Only shows valid (non-expired) relationships

### 5. 🔐 Security & Rate Limiting
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

**Architecture**:
- **Direct Cypher Queries**: Bypasses Graphiti for read performance
- **Degree-Based Filtering**: Only includes entities with ≥2 connections (default)
- **Two-Phase Output**:
  1. **Conceptual Definitions**: Entity summaries ordered by connectivity
  2. **Relational Dynamics**: Relationships with facts, timestamps, and temporal context

**Output Format**:
```markdown
#### 1. CONCEPTUAL DEFINITIONS & IDENTITY ####
- **Entity Name**: Summary of what this entity represents
...

#### 2. RELATIONAL DYNAMICS & CAUSALITY ####
- Entity1 relates to Entity2: "fact describing relationship" [valid_at: timestamp]
...

### STATS ###
Definitions: X | Relations: Y | Est. Tokens: ~Z
```

### 3. **Generation Service** (`app/services/generation.py`)
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

### 4. **Graph Service** (`app/services/graph.py`)
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
  "userId": "user-123"
}
```

**Response**:
```json
{
  "success": true,
  "userKnowledgeCompilation": "#### 1. CONCEPTUAL DEFINITIONS & IDENTITY ####\n..."
}
```

**Use Cases**:
- Debugging current graph state
- Fetching context without re-indexing
- Testing hydration logic

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
       {messages, model, stream: true}

2. GENERATION SERVICE
   ├─▶ Convert messages to Gemini format
   │   - Extract system prompt
   │   - Prepend to first user message
   │   - Map roles: user/assistant → user/model
   └─▶ Call gemini.generate_content_stream()

3. GEMINI API
   └─▶ Stream token chunks

4. GENERATION SERVICE
   ├─▶ Wrap chunks in OpenAI SSE format
   │   data: {"choices": [{"delta": {"content": "..."}}]}
   └─▶ Send [DONE] marker

5. CLIENT
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

## Setup & Deployment

### Prerequisites

- **Python**: 3.12+
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

#### 6. Stop Services
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

# Restart services
docker-compose up -d
```

