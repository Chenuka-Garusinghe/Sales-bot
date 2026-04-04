from fastapi import FastAPI

from app.routers import discovery, leads, mirofish

app = FastAPI(
    title="SimplAI AU - Sales Pipeline Demo",
    description="LangGraph ReAct agent + MiroFish sales pipeline demonstration.",
    version="0.2.0",
)

app.include_router(leads.router, prefix="/api/leads", tags=["Lead Agent (LangGraph)"])
app.include_router(discovery.router, prefix="/api/discovery", tags=["Discovery Call Assistant"])
app.include_router(mirofish.router, prefix="/api/mirofish", tags=["MiroFish - Training Scenarios"])


@app.get("/")
async def root():
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
    return {
        "stage_1_lead_generation": {
            "name": "Lead Agent (LangGraph)",
            "description": "ReAct agent backed by Claude that reasons about which lead sources to query based on natural language prompts",
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
