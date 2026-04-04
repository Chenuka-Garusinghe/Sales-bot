# Sales-bot

LangGraph ReAct agent + MiroFish integration for increasing sales performance. Demo version for SimplAI AU.

## Overview

This is a FastAPI demo application showcasing an AI-powered sales pipeline built on:

- **LangGraph + Claude**: A ReAct agent that reasons about which lead sources to query based on natural language prompts from a sales manager
- **MiroFish**: Real-world sales interaction simulation for team training

All lead data is mocked for demo purposes. The LangGraph agent uses a real Claude LLM call to decide which tools to invoke.

## Architecture

```
Sales-bot/
├── main.py                          # FastAPI entry point
├── requirements.txt
├── app/
│   ├── models/                      # Pydantic data models
│   │   ├── leads.py                 # Lead, evaluation, and agent response models
│   │   ├── discovery.py             # Transcript & insight models
│   │   └── mirofish.py              # Persona & scenario models
│   ├── mock_data/                   # Realistic Australian-market B2B data
│   │   ├── leads.py                 # ~24 leads across 4 sources
│   │   ├── transcripts.py           # 3 discovery call transcripts
│   │   ├── past_meetings.py         # 8 historical meetings (RAG corpus)
│   │   ├── personas.py              # 6 customer personas
│   │   └── scenarios.py             # 8-10 training scenario templates
│   ├── services/                    # Business logic
│   │   ├── lead_agent.py            # LangGraph ReAct agent with lead source tools
│   │   ├── lead_evaluator.py        # Deterministic filtering/scoring
│   │   ├── discovery_call.py        # Call analysis orchestrator
│   │   ├── rag.py                   # Mock RAG via keyword overlap
│   │   └── mirofish.py              # Scenario generation engine
│   └── routers/                     # API route handlers
│       ├── leads.py                 # /api/leads/*
│       ├── discovery.py             # /api/discovery/*
│       └── mirofish.py              # /api/mirofish/*
```

## Sales Pipeline

### Stage 1: LangGraph Lead Agent

A ReAct agent backed by Claude (`claude-sonnet-4-20250514`) receives a natural language prompt from a sales manager (e.g. "find mid-market fintech companies") and reasons about which lead source tools to call:

- **search_apollo** -- Apollo lead database
- **search_linkedin** -- LinkedIn professional network
- **search_web** -- Public web scraping
- **search_referrals** -- Partner referral network

The agent decides which tools to invoke and what filters to apply. Gathered leads then pass through a deterministic evaluator (authenticity, email, dedup, industry, company size, relevance scoring).

**Graph flow:** `prompt → agent (ReAct reasoning + tool calls) → evaluate (deterministic filter) → END`

### Stage 2: Discovery Call Assistant

Processes sales call transcripts through a mock RAG pipeline that retrieves similar past meetings via keyword overlap. Generates insights including deal risks, objection patterns, coaching tips, and sentiment analysis. Analyzed calls are added to a growing datastore.

### Stage 3: MiroFish -- Training Scenarios

Generates simulated sales scenarios paired with customer personas of varying difficulty. Covers objection handling, negotiation, discovery, and closing categories. Each scenario includes a full conversation with coaching notes.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check + API overview |
| `/api/pipeline/overview` | GET | Full pipeline flow description |
| `/api/leads/run` | POST | Execute LangGraph agent lead gathering (accepts `prompt` field) |
| `/api/leads/tools` | GET | List available lead source tools |
| `/api/leads/leads/{run_id}` | GET | Retrieve cached run results |
| `/api/discovery/transcripts` | GET | List available transcripts |
| `/api/discovery/transcripts/{call_id}` | GET | Get full transcript |
| `/api/discovery/analyze/{call_id}` | POST | Run analysis + RAG pipeline |
| `/api/discovery/datastore` | GET | View growing past meetings store |
| `/api/mirofish/personas` | GET | List customer personas |
| `/api/mirofish/generate` | POST | Generate training scenarios |
| `/api/mirofish/scenarios/{id}` | GET | Get specific scenario |

## Setup

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set your Anthropic API key
export ANTHROPIC_API_KEY="your-key-here"

# Run the server
uvicorn main:app --reload
```

Then open http://localhost:8000/docs for the interactive Swagger UI.

## Example Usage

```bash
# Let the agent decide which sources to search
curl -X POST http://localhost:8000/api/leads/run \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Find mid-market fintech companies with at least 100 employees"}'

# Search across all sources
curl -X POST http://localhost:8000/api/leads/run \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Search all available sources for healthcare and education leads"}'
```

## Tech Stack

- **Python 3.11+**
- **FastAPI** -- async web framework
- **LangGraph** -- ReAct agent orchestration
- **Claude Sonnet** (`claude-sonnet-4-20250514`) -- LLM backing the agent
- **Pydantic v2** -- data validation and serialization
- **Uvicorn** -- ASGI server
