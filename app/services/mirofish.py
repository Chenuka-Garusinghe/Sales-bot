"""
Stage 3: MiroFish Training Scenario Service

Generates simulated sales training scenarios by pairing scenario templates
with customer personas. This lets sales reps practice against different
types of customers and situations.

The data comes from:
    - app/mock_data/personas.py   -- 6 customer archetypes (easy to hard)
    - app/mock_data/scenarios.py  -- 10 scenario templates across 4 categories

Categories: objection_handling, negotiation, discovery, closing
Difficulties: easy, medium, hard

The scenario generator:
    1. Filters the template pool by category and difficulty (if specified)
    2. Randomly samples the requested number of scenarios
    3. Pairs each scenario with a matching-difficulty persona
    4. Caches generated scenarios for retrieval by ID
"""

import random

from app.mock_data.personas import PERSONAS
from app.mock_data.scenarios import SCENARIO_TEMPLATES
from app.models.mirofish import (
    CustomerPersona,
    MiroFishResponse,
    ScenarioTurn,
    TrainingScenario,
)

# In-memory cache for generated scenarios so they can be fetched by ID later
# via GET /api/mirofish/scenarios/{scenario_id}
_scenario_cache: dict[str, TrainingScenario] = {}


def list_personas() -> list[CustomerPersona]:
    """Return all available customer personas. Used by GET /api/mirofish/personas."""
    return [CustomerPersona(**p) for p in PERSONAS]


def get_persona(persona_id: str) -> CustomerPersona | None:
    """Look up a single persona by ID. Returns None if not found."""
    for p in PERSONAS:
        if p["persona_id"] == persona_id:
            return CustomerPersona(**p)
    return None


def generate_scenarios(
    category: str | None = None,
    difficulty: str | None = None,
    count: int = 3,
) -> MiroFishResponse:
    """Generate training scenarios, optionally filtered by category and difficulty.

    This is the main function called by POST /api/mirofish/generate.

    Args:
        category: Filter to a specific category (e.g. "negotiation"). None = all.
        difficulty: Filter to a difficulty level (e.g. "hard"). None = all.
        count: How many scenarios to return (randomly sampled from the filtered pool).

    Returns:
        MiroFishResponse with the generated scenarios and total pool size.
    """
    # Start with the full template pool
    pool = list(SCENARIO_TEMPLATES)

    # Apply filters if specified
    if category:
        pool = [s for s in pool if s["category"] == category]
    if difficulty:
        pool = [s for s in pool if s["difficulty"] == difficulty]

    # Randomly pick 'count' scenarios (or fewer if pool is smaller)
    selected = random.sample(pool, min(count, len(pool)))

    scenarios: list[TrainingScenario] = []
    for template in selected:
        # Pair with a persona that matches the scenario's difficulty level
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
        # Cache so it can be retrieved later by ID
        _scenario_cache[scenario.scenario_id] = scenario

    return MiroFishResponse(
        scenarios=scenarios,
        total_available=len(SCENARIO_TEMPLATES),
    )


def get_scenario(scenario_id: str) -> TrainingScenario | None:
    """Retrieve a previously generated scenario by ID from the cache.
    Returns None if the scenario hasn't been generated yet.
    """
    return _scenario_cache.get(scenario_id)
