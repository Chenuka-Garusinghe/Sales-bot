from collections import Counter

from app.models.leads import Lead, EvaluationResult

VALID_INDUSTRIES = {"healthcare", "fintech", "saas", "professional_services", "education"}


def _compute_relevance(lead: Lead) -> float:
    score = 0.5
    if lead.company_size > 200:
        score += 0.2
    if lead.title and any(
        kw in lead.title.lower()
        for kw in ["head", "director", "vp", "cto", "ceo", "managing"]
    ):
        score += 0.2
    if lead.industry in {"saas", "fintech"}:
        score += 0.1
    return min(score, 1.0)


def evaluate_leads(
    all_leads: list[Lead], min_company_size: int = 50
) -> EvaluationResult:
    qualified: list[Lead] = []
    rejection_reasons: list[str] = []
    seen_emails: set[str] = set()

    for lead in all_leads:
        if not lead.is_authentic:
            lead.rejection_reason = "failed_authenticity"
            rejection_reasons.append("failed_authenticity")
            continue

        if not lead.email or "@" not in lead.email:
            lead.rejection_reason = "invalid_email"
            rejection_reasons.append("invalid_email")
            continue

        if lead.email.lower() in seen_emails:
            lead.rejection_reason = "duplicate"
            rejection_reasons.append("duplicate")
            continue
        seen_emails.add(lead.email.lower())

        if lead.industry.lower() not in VALID_INDUSTRIES:
            lead.rejection_reason = "irrelevant_industry"
            rejection_reasons.append("irrelevant_industry")
            continue

        if lead.company_size < min_company_size:
            lead.rejection_reason = "company_too_small"
            rejection_reasons.append("company_too_small")
            continue

        lead.relevance_score = _compute_relevance(lead)
        qualified.append(lead)

    qualified.sort(key=lambda l: l.relevance_score, reverse=True)

    return EvaluationResult(
        total_gathered=len(all_leads),
        passed_filter=len(qualified),
        rejected=len(rejection_reasons),
        rejection_breakdown=dict(Counter(rejection_reasons)),
        qualified_leads=qualified,
    )
