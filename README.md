# Synapse Cortex

Cognitive backend for the Synapse AI Chat application. A stateless REST API that handles knowledge graph operations and LLM interactions.

## Features

- **Unified Ingest**: Process chat sessions via Graphiti and return updated user knowledge compilation
- **Chat Completions**: OpenAI-compatible streaming responses using Google Gemini
- **Knowledge Graph**: Neo4j-backed long-term memory with intelligent hydration

## Quick Start

### Prerequisites

- Python 3.12+
- Docker & Docker Compose
- Google Gemini API key

### Environment Setup

```bash
cp .env.example .env
```

Edit `.env` with your credentials:
```env
NEO4J_PASSWORD=localpassword
GOOGLE_API_KEY=your_gemini_api_key
SYNAPSE_API_SECRET=your_api_secret

# Optional: Change Graphiti model (default: gemini-3-flash-preview)
# GRAPHITI_MODEL=gemini-2.5-flash
```

---

## Local Development

Run Neo4j in Docker, FastAPI natively with hot-reload.

### 1. Start Neo4j

```bash
docker-compose -f docker-compose.local.yml up -d
```

This starts Neo4j with:
- **Browser UI**: http://localhost:7474
- **Bolt Protocol**: `bolt://localhost:7687`
- **Credentials**: `neo4j` / `localpassword`

### 2. Install Python Dependencies

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run FastAPI

```bash
uvicorn app.main:app --reload
```

API available at: http://localhost:8000

### 4. Stop Neo4j

```bash
docker-compose -f docker-compose.local.yml down
```

To also remove the data volume:
```bash
docker-compose -f docker-compose.local.yml down -v
```

---

## Production Deployment (Digital Ocean)

Full Docker stack with Neo4j, FastAPI, and Caddy (automatic SSL).

### 1. Server Setup

SSH into your Digital Ocean droplet and clone the repo:

```bash
git clone <your-repo-url> synapse-cortex
cd synapse-cortex
```

### 2. Configure Environment

```bash
cp .env.example .env
nano .env
```

Set production values:
```env
NEO4J_PASSWORD=<strong-random-password>
GOOGLE_API_KEY=<your-gemini-api-key>
SYNAPSE_API_SECRET=<your-api-secret>
SEMAPHORE_LIMIT=3

# Optional: Change Graphiti model (default: gemini-3-flash-preview)
# GRAPHITI_MODEL=gemini-2.5-flash
```

### 3. Configure DNS

Create an A record in your DNS provider:
- **Name**: `synapse-cortex`
- **Value**: Your droplet's IP address
- **Domain**: `juandago.dev`

Result: `synapse-cortex.juandago.dev` → Droplet IP

### 4. Deploy

```bash
docker-compose up -d --build
```

Caddy automatically obtains SSL certificates from Let's Encrypt.

### 5. Verify

```bash
# Check running containers
docker-compose ps

# View logs
docker-compose logs -f

# Test the API
curl https://synapse-cortex.juandago.dev/health
```

### 6. Update Deployment

```bash
git pull
docker-compose up -d --build
```

### 7. Stop Services

```bash
docker-compose down
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ingest` | POST | Process session messages and return user knowledge compilation |
| `/v1/chat/completions` | POST | OpenAI-compatible streaming chat completions |
| `/health` | GET | Health check endpoint |

## Authentication

All endpoints (except `/health`) require an `X-API-SECRET` header matching the configured `SYNAPSE_API_SECRET`.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Production Stack                      │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   Internet ──▶ Caddy (SSL/443) ──▶ FastAPI ──▶ Neo4j   │
│                                                         │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                  Local Development                       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   localhost:8000 ──▶ uvicorn ──▶ Neo4j (Docker:7687)   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```
