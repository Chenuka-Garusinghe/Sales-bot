# Sales-bot

OpenClaw + MiroFish integration for increasing sales performance. Demo version for SimplAI AU.

## Overview

This is a FastAPI demo application showcasing an AI-powered sales pipeline built on two core technologies:

- **OpenClaw**: Multi-agent orchestration for parallel lead gathering and evaluation
- **MiroFish**: Real-world sales interaction simulation for team training

All data is mocked for demo purposes -- no real API integrations required.

## Architecture

```
Sales-bot/
├── main.py                          # FastAPI entry point
├── requirements.txt
├── app/
│   ├── models/                      # Pydantic data models
│   │   ├── leads.py                 # Lead & agent result models
│   │   ├── discovery.py             # Transcript & insight models
│   │   └── mirofish.py              # Persona & scenario models
│   ├── mock_data/                   # Realistic Australian-market B2B data
│   │   ├── leads.py                 # ~24 leads across 4 sources
│   │   ├── transcripts.py           # 3 discovery call transcripts
│   │   ├── past_meetings.py         # 8 historical meetings (RAG corpus)
│   │   ├── personas.py              # 6 customer personas
│   │   └── scenarios.py             # 8-10 training scenario templates
│   ├── services/                    # Business logic
│   │   ├── openclaw.py              # Multi-agent parallel lead search
│   │   ├── lead_evaluator.py        # Deterministic filtering/scoring
│   │   ├── discovery_call.py        # Call analysis orchestrator
│   │   ├── rag.py                   # Mock RAG via keyword overlap
│   │   └── mirofish.py              # Scenario generation engine
│   └── routers/                     # API route handlers
│       ├── openclaw.py              # /api/openclaw/*
│       ├── discovery.py             # /api/discovery/*
│       └── mirofish.py              # /api/mirofish/*
```

## Sales Pipeline

### Stage 1: OpenClaw -- Parallel Lead Gathering

Four agents search different sources (Apollo, LinkedIn, web scrape, referrals) simultaneously using `asyncio.gather`. Results pass through a deterministic evaluation point that filters by authenticity, email validity, deduplication, industry relevance, and company size.

### Stage 2: Discovery Call Assistant

Processes sales call transcripts through a mock RAG pipeline that retrieves similar past meetings via keyword overlap. Generates insights including deal risks, objection patterns, coaching tips, and sentiment analysis. Analyzed calls are added to a growing datastore.

### Stage 3: MiroFish -- Training Scenarios

Generates simulated sales scenarios paired with customer personas of varying difficulty. Covers objection handling, negotiation, discovery, and closing categories. Each scenario includes a full conversation with coaching notes.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check + API overview |
| `/api/pipeline/overview` | GET | Full pipeline flow description |
| `/api/openclaw/run` | POST | Execute parallel lead gathering pipeline |
| `/api/openclaw/agents` | GET | List available agents |
| `/api/openclaw/leads/{run_id}` | GET | Retrieve cached run results |
| `/api/discovery/transcripts` | GET | List available transcripts |
| `/api/discovery/transcripts/{call_id}` | GET | Get full transcript |
| `/api/discovery/analyze/{call_id}` | POST | Run analysis + RAG pipeline |
| `/api/discovery/datastore` | GET | View growing past meetings store |
| `/api/mirofish/personas` | GET | List customer personas |
| `/api/mirofish/generate` | POST | Generate training scenarios |
| `/api/mirofish/scenarios/{id}` | GET | Get specific scenario |

## Setup

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Then open http://localhost:8000/docs for the interactive Swagger UI.

## Tech Stack

- **Python 3.11+**
- **FastAPI** -- async web framework
- **Pydantic v2** -- data validation and serialization
- **Uvicorn** -- ASGI server

## To run it
source .venv/bin/activate
uvicorn main:app --reload
Then open http://localhost:8000/docs for the interactive Swagger UI.