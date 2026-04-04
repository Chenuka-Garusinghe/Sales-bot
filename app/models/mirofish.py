from pydantic import BaseModel
from enum import Enum


class PersonaDifficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class CustomerPersona(BaseModel):
    persona_id: str
    name: str
    role: str
    company_type: str
    personality_traits: list[str]
    common_objections: list[str]
    buying_style: str
    difficulty: PersonaDifficulty


class ScenarioTurn(BaseModel):
    speaker: str
    text: str
    coaching_note: str | None = None


class TrainingScenario(BaseModel):
    scenario_id: str
    title: str
    description: str
    persona: CustomerPersona
    category: str
    conversation: list[ScenarioTurn]
    ideal_outcome: str
    difficulty: PersonaDifficulty


class MiroFishRequest(BaseModel):
    category: str | None = None
    difficulty: str | None = None
    count: int = 3


class MiroFishResponse(BaseModel):
    scenarios: list[TrainingScenario]
    total_available: int
