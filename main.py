"""
SimplAI AU Sales Pipeline Demo -- FastAPI Application

This is the entry point. It registers three routers (one per pipeline stage)
and exposes two top-level endpoints for health check and pipeline overview.

Pipeline stages:
    Stage 1: /api/leads/*     -- LangGraph ReAct agent for lead gathering
    Stage 2: /api/discovery/* -- Discovery call analysis with mock RAG
    Stage 3: /api/mirofish/*  -- MiroFish training scenario generation

To run:
    source .venv/bin/activate
    uvicorn main:app --reload

Then open http://localhost:8000/docs for Swagger UI.
"""

from fastapi import FastAPI

from app.routers import discovery, leads, mirofish

# Create the FastAPI app -- title and description show up in Swagger UI at /docs
app = FastAPI(
    title="SimplAI AU - Sales Pipeline Demo",
    description="LangGraph ReAct agent + MiroFish sales pipeline demonstration.",
    version="0.2.0",
)

# Register each stage's router with its URL prefix and Swagger tag
app.include_router(leads.router, prefix="/api/leads", tags=["Lead Agent (LangGraph)"])
app.include_router(discovery.router, prefix="/api/discovery", tags=["Discovery Call Assistant"])
app.include_router(mirofish.router, prefix="/api/mirofish", tags=["MiroFish - Training Scenarios"])


@app.get("/")
async def root():
    """Health check and API overview. Hit this to verify the server is running."""
    return {
        "service": "SimplAI AU Sales Pipeline Demo",
        "version": "0.2.0",
        "pipelines": {
            "leads": "/api/leads/run - LangGraph ReAct agent lead gathering",
            "discovery": "/api/discovery/analyze/{call_id} - Call analysis + RAG",
            "mirofish": "/api/mirofish/generate - Training scenario generation",
        },
        "docs": "/docs",
    }


@app.get("/api/pipeline/overview")
async def pipeline_overview():
    """Returns a JSON description of the full pipeline flow.
    Useful for demo narration or building a frontend visualization.
    """
    return {
        "stage_1_lead_generation": {
            "name": "Lead Agent (LangGraph)",
            "description": "ReAct agent backed by Ollama/llama3.1 that reasons about which lead sources to query based on natural language prompts",
            "flow": [
                "natural_language_prompt",
                "react_reasoning",
                "tool_calls (apollo, linkedin, web, referrals)",
                "deterministic_evaluation",
                "qualified_leads_output",
            ],
        },
        "stage_2_discovery": {
            "name": "Discovery Call Assistant",
            "description": "Voice-to-text transcription with RAG-powered insights",
            "flow": [
                "transcript_ingestion",
                "rag_retrieval",
                "pattern_matching",
                "insight_generation",
            ],
        },
        "stage_3_training": {
            "name": "MiroFish",
            "description": "Simulated sales scenarios for rep training",
            "flow": [
                "persona_selection",
                "scenario_generation",
                "practice_interaction",
            ],
        },
    }
