"""
Data models for Stage 3: MiroFish Training Scenario Simulation.

These models define customer personas and training scenarios
that sales reps can use to practice handling different situations.
"""

from pydantic import BaseModel
from enum import Enum


class PersonaDifficulty(str, Enum):
    """How challenging this persona/scenario is for a sales rep."""
    EASY = "easy"       # friendly, budget-ready prospects
    MEDIUM = "medium"   # some pushback on price/timeline
    HARD = "hard"       # skeptical, multi-stakeholder, competitive


class CustomerPersona(BaseModel):
    """A simulated customer archetype for training scenarios.
    Each persona has distinct traits that affect how they respond.

    buying_style: one of "analytical", "expressive", "driver", "amiable"
                  based on common sales psychology frameworks.
    """
    persona_id: str
    name: str                          # e.g. "Skeptical Steve"
    role: str                          # e.g. "Managing Director"
    company_type: str                  # e.g. "Traditional manufacturing firm"
    personality_traits: list[str]      # e.g. ["skeptical of AI", "experienced"]
    common_objections: list[str]       # what they typically push back on
    buying_style: str                  # analytical, expressive, driver, or amiable
    difficulty: PersonaDifficulty


class ScenarioTurn(BaseModel):
    """A single exchange in a training scenario conversation.

    coaching_note: hidden tip for the rep (only on rep turns),
                   explains WHY this response works well.
                   Set to None on prospect turns.
    """
    speaker: str                       # "rep" or "prospect"
    text: str                          # what they say
    coaching_note: str | None = None   # training tip (rep turns only)


class TrainingScenario(BaseModel):
    """A complete training scenario with a persona, dialogue, and ideal outcome.

    category: one of "objection_handling", "negotiation", "discovery", "closing"
    """
    scenario_id: str
    title: str                              # e.g. "The Price Pushback"
    description: str                        # what the scenario is about
    persona: CustomerPersona                # the customer the rep is talking to
    category: str                           # type of sales skill being trained
    conversation: list[ScenarioTurn]        # the example dialogue
    ideal_outcome: str                      # what a successful rep would achieve
    difficulty: PersonaDifficulty


class MiroFishRequest(BaseModel):
    """Request body for POST /api/mirofish/generate.
    All fields are optional -- omit them to get random scenarios.
    """
    category: str | None = None      # filter by category (e.g. "negotiation")
    difficulty: str | None = None    # filter by difficulty (e.g. "hard")
    count: int = 3                   # how many scenarios to generate


class MiroFishResponse(BaseModel):
    """Response from POST /api/mirofish/generate."""
    scenarios: list[TrainingScenario]
    total_available: int   # total scenarios in the pool (before filtering)
