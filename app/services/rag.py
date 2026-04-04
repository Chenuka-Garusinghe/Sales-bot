from app.mock_data.past_meetings import PAST_MEETINGS
from app.models.discovery import RAGMatch


def retrieve_similar_meetings(
    transcript_text: str, top_k: int = 3
) -> list[RAGMatch]:
    transcript_words = set(transcript_text.lower().split())

    scored: list[RAGMatch] = []
    for meeting in PAST_MEETINGS:
        meeting_keywords = set(kw.lower() for kw in meeting["keywords"])
        overlap = len(transcript_words & meeting_keywords)
        total = len(meeting_keywords)
        similarity = overlap / total if total > 0 else 0.0

        if similarity > 0.1:
            scored.append(
                RAGMatch(
                    matched_call_id=meeting["call_id"],
                    similarity_score=round(similarity, 3),
                    matched_excerpt=meeting["summary"][:200],
                    outcome=meeting["outcome"],
                    key_technique=meeting["key_technique"],
                )
            )

    scored.sort(key=lambda m: m.similarity_score, reverse=True)
    return scored[:top_k]
