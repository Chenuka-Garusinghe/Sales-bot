"""
Router for Stage 3: MiroFish Training Scenario endpoints.

All endpoints are prefixed with /api/mirofish (set in main.py).

Endpoints:
    GET  /api/mirofish/personas               -- list all customer personas
    GET  /api/mirofish/personas/{persona_id}   -- get a specific persona
    POST /api/mirofish/generate                -- generate training scenarios
    GET  /api/mirofish/scenarios/{scenario_id}  -- get a previously generated scenario
"""

from fastapi import APIRouter, HTTPException

from app.models.mirofish import CustomerPersona, MiroFishRequest, MiroFishResponse, TrainingScenario
from app.services.mirofish import generate_scenarios, get_persona, get_scenario, list_personas

router = APIRouter()


@router.get("/personas", response_model=list[CustomerPersona])
async def get_personas():
    """List all 6 customer personas available for scenario generation.
    Personas range from easy (friendly buyers) to hard (skeptical executives).
    """
    return list_personas()


@router.get("/personas/{persona_id}", response_model=CustomerPersona)
async def get_persona_detail(persona_id: str):
    """Get details for a specific persona (e.g. persona-001 through persona-006)."""
    persona = get_persona(persona_id)
    if not persona:
        raise HTTPException(status_code=404, detail=f"Persona '{persona_id}' not found")
    return persona


@router.post("/generate", response_model=MiroFishResponse)
async def generate_training_scenarios(request: MiroFishRequest | None = None):
    """Generate training scenarios for sales rep practice.

    Optional filters in request body:
        category: "objection_handling", "negotiation", "discovery", or "closing"
        difficulty: "easy", "medium", or "hard"
        count: how many scenarios to generate (default 3)

    Each scenario includes a customer persona, full conversation with coaching
    notes, and the ideal outcome the rep should aim for.
    """
    req = request or MiroFishRequest()
    return generate_scenarios(
        category=req.category,
        difficulty=req.difficulty,
        count=req.count,
    )


@router.get("/scenarios/{scenario_id}", response_model=TrainingScenario)
async def get_scenario_detail(scenario_id: str):
    """Retrieve a previously generated scenario by its ID.
    Scenarios are cached in memory after generation.
    You must call POST /generate first before fetching by ID.
    """
    scenario = get_scenario(scenario_id)
    if not scenario:
        raise HTTPException(status_code=404, detail=f"Scenario '{scenario_id}' not found. Generate scenarios first via POST /generate")
    return scenario
