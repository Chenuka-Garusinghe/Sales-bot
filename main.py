from fastapi import FastAPI

from app.routers import discovery, mirofish, openclaw

app = FastAPI(
    title="SimplAI AU - Sales Pipeline Demo",
    description="OpenClaw + MiroFish sales pipeline demonstration. All data is mock.",
    version="0.1.0",
)

app.include_router(openclaw.router, prefix="/api/openclaw", tags=["OpenClaw - Lead Gathering"])
app.include_router(discovery.router, prefix="/api/discovery", tags=["Discovery Call Assistant"])
app.include_router(mirofish.router, prefix="/api/mirofish", tags=["MiroFish - Training Scenarios"])


@app.get("/")
async def root():
    return {
        "service": "SimplAI AU Sales Pipeline Demo",
        "version": "0.1.0",
        "pipelines": {
            "openclaw": "/api/openclaw/run - Multi-agent lead gathering",
            "discovery": "/api/discovery/analyze/{call_id} - Call analysis + RAG",
            "mirofish": "/api/mirofish/generate - Training scenario generation",
        },
        "docs": "/docs",
    }


@app.get("/api/pipeline/overview")
async def pipeline_overview():
    return {
        "stage_1_lead_generation": {
            "name": "OpenClaw",
            "description": "Parallel multi-agent lead search across Apollo, LinkedIn, Web, Referrals",
            "flow": [
                "agents_search_parallel",
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
