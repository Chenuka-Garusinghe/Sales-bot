import random

from app.mock_data.personas import PERSONAS
from app.mock_data.scenarios import SCENARIO_TEMPLATES
from app.models.mirofish import (
    CustomerPersona,
    MiroFishResponse,
    ScenarioTurn,
    TrainingScenario,
)

# Cache for generated scenarios
_scenario_cache: dict[str, TrainingScenario] = {}


def list_personas() -> list[CustomerPersona]:
    return [CustomerPersona(**p) for p in PERSONAS]


def get_persona(persona_id: str) -> CustomerPersona | None:
    for p in PERSONAS:
        if p["persona_id"] == persona_id:
            return CustomerPersona(**p)
    return None


def generate_scenarios(
    category: str | None = None,
    difficulty: str | None = None,
    count: int = 3,
) -> MiroFishResponse:
    pool = list(SCENARIO_TEMPLATES)

    if category:
        pool = [s for s in pool if s["category"] == category]
    if difficulty:
        pool = [s for s in pool if s["difficulty"] == difficulty]

    selected = random.sample(pool, min(count, len(pool)))

    scenarios: list[TrainingScenario] = []
    for template in selected:
        matching_personas = [
            p for p in PERSONAS if p["difficulty"] == template["difficulty"]
        ]
        persona_data = random.choice(matching_personas) if matching_personas else PERSONAS[0]
        persona = CustomerPersona(**persona_data)

        scenario = TrainingScenario(
            scenario_id=template["scenario_id"],
            title=template["title"],
            description=template["description"],
            persona=persona,
            category=template["category"],
            conversation=[ScenarioTurn(**turn) for turn in template["conversation"]],
            ideal_outcome=template["ideal_outcome"],
            difficulty=template["difficulty"],
        )
        scenarios.append(scenario)
        _scenario_cache[scenario.scenario_id] = scenario

    return MiroFishResponse(
        scenarios=scenarios,
        total_available=len(SCENARIO_TEMPLATES),
    )


def get_scenario(scenario_id: str) -> TrainingScenario | None:
    return _scenario_cache.get(scenario_id)
