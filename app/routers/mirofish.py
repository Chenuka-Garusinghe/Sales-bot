from fastapi import APIRouter, HTTPException

from app.models.mirofish import CustomerPersona, MiroFishRequest, MiroFishResponse, TrainingScenario
from app.services.mirofish import generate_scenarios, get_persona, get_scenario, list_personas

router = APIRouter()


@router.get("/personas", response_model=list[CustomerPersona])
async def get_personas():
    return list_personas()


@router.get("/personas/{persona_id}", response_model=CustomerPersona)
async def get_persona_detail(persona_id: str):
    persona = get_persona(persona_id)
    if not persona:
        raise HTTPException(status_code=404, detail=f"Persona '{persona_id}' not found")
    return persona


@router.post("/generate", response_model=MiroFishResponse)
async def generate_training_scenarios(request: MiroFishRequest | None = None):
    req = request or MiroFishRequest()
    return generate_scenarios(
        category=req.category,
        difficulty=req.difficulty,
        count=req.count,
    )


@router.get("/scenarios/{scenario_id}", response_model=TrainingScenario)
async def get_scenario_detail(scenario_id: str):
    scenario = get_scenario(scenario_id)
    if not scenario:
        raise HTTPException(status_code=404, detail=f"Scenario '{scenario_id}' not found. Generate scenarios first via POST /generate")
    return scenario
