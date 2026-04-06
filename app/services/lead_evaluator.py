"""
Deterministic Lead Evaluator

This runs AFTER the LangGraph agent gathers leads. It applies hard rules
to filter out bad leads and score the remaining ones. No LLM involved here --
pure logic so results are reproducible and auditable.

FILTER PIPELINE (order matters -- first failing rule rejects the lead):
    1. Authenticity check  -- is_authentic must be True
    2. Email validation    -- must contain "@"
    3. Deduplication       -- no two leads with the same email
    4. Industry relevance  -- must be in VALID_INDUSTRIES set
    5. Company size        -- must meet minimum threshold

SCORING (for leads that pass all filters):
    Base score: 0.5
    + 0.2 if company_size > 200
    + 0.2 if title contains seniority keywords (head, director, vp, cto, ceo, managing)
    + 0.1 if industry is saas or fintech
    Max: 1.0

TO CUSTOMIZE:
    - Add/remove industries in VALID_INDUSTRIES
    - Adjust scoring weights in _compute_relevance()
    - Add new filter rules in evaluate_leads()
"""

from collections import Counter

from app.models.leads import Lead, EvaluationResult

# Industries we consider relevant for SimplAI AU's services
# Leads outside these industries get rejected as "irrelevant_industry"
VALID_INDUSTRIES = {"healthcare", "fintech", "saas", "professional_services", "education"}


def _compute_relevance(lead: Lead) -> float:
    """Score a lead from 0.0 to 1.0 based on how promising they are.
    Higher score = more senior title, bigger company, high-value industry.
    """
    score = 0.5  # base score for any valid lead

    # Bigger companies = bigger potential deals
    if lead.company_size > 200:
        score += 0.2

    # Senior titles = decision-making power
    if lead.title and any(
        kw in lead.title.lower()
        for kw in ["head", "director", "vp", "cto", "ceo", "managing"]
    ):
        score += 0.2

    # SaaS and fintech are our strongest verticals
    if lead.industry in {"saas", "fintech"}:
        score += 0.1

    return min(score, 1.0)  # cap at 1.0


def evaluate_leads(
    all_leads: list[Lead], min_company_size: int = 50
) -> EvaluationResult:
    """Run all leads through the filter pipeline and score survivors.

    Args:
        all_leads: Raw leads from all tool calls (may contain dupes, fakes, etc.)
        min_company_size: Minimum employee count to qualify

    Returns:
        EvaluationResult with qualified leads sorted by relevance score (descending)
    """
    qualified: list[Lead] = []
    rejection_reasons: list[str] = []
    seen_emails: set[str] = set()  # for dedup tracking

    for lead in all_leads:
        # Rule 1: Authenticity -- some mock leads are intentionally flagged as fake
        if not lead.is_authentic:
            lead.rejection_reason = "failed_authenticity"
            rejection_reasons.append("failed_authenticity")
            continue

        # Rule 2: Valid email -- must have an @ sign at minimum
        if not lead.email or "@" not in lead.email:
            lead.rejection_reason = "invalid_email"
            rejection_reasons.append("invalid_email")
            continue

        # Rule 3: Dedup -- same person found in multiple sources
        if lead.email.lower() in seen_emails:
            lead.rejection_reason = "duplicate"
            rejection_reasons.append("duplicate")
            continue
        seen_emails.add(lead.email.lower())

        # Rule 4: Industry relevance -- we only sell to certain verticals
        if lead.industry.lower() not in VALID_INDUSTRIES:
            lead.rejection_reason = "irrelevant_industry"
            rejection_reasons.append("irrelevant_industry")
            continue

        # Rule 5: Company size -- too small = not enough budget
        if lead.company_size < min_company_size:
            lead.rejection_reason = "company_too_small"
            rejection_reasons.append("company_too_small")
            continue

        # Lead passed all filters -- score it
        lead.relevance_score = _compute_relevance(lead)
        qualified.append(lead)

    # Best leads first
    qualified.sort(key=lambda l: l.relevance_score, reverse=True)

    return EvaluationResult(
        total_gathered=len(all_leads),
        passed_filter=len(qualified),
        rejected=len(rejection_reasons),
        rejection_breakdown=dict(Counter(rejection_reasons)),
        qualified_leads=qualified,
    )
