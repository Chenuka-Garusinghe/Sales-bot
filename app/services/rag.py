"""
Mock RAG (Retrieval-Augmented Generation) Service

This simulates a RAG pipeline for the Discovery Call Assistant.
In production, you'd use vector embeddings (e.g. OpenAI embeddings)
stored in a vector DB (e.g. Pinecone, Weaviate) and do cosine similarity search.

In this demo, we use keyword overlap as a simple proxy:
    similarity = (number of matching keywords) / (total keywords in historical meeting)

The historical meetings are stored in app/mock_data/past_meetings.py.
Each has a list of keywords that represent what the meeting was about.
"""

from app.mock_data.past_meetings import PAST_MEETINGS
from app.models.discovery import RAGMatch


def retrieve_similar_meetings(
    transcript_text: str, top_k: int = 3
) -> list[RAGMatch]:
    """Find historical meetings most similar to the given transcript.

    Args:
        transcript_text: The full text of the current call transcript
        top_k: How many matches to return (default 3)

    Returns:
        List of RAGMatch objects sorted by similarity (highest first),
        only including matches above the 0.1 threshold.

    How it works:
        1. Split transcript into a set of words
        2. For each historical meeting, compute keyword overlap ratio
        3. Filter out matches below 0.1 similarity
        4. Return the top_k best matches
    """
    # Turn the transcript into a set of lowercase words for matching
    transcript_words = set(transcript_text.lower().split())

    scored: list[RAGMatch] = []
    for meeting in PAST_MEETINGS:
        # Each historical meeting has a curated keyword list
        meeting_keywords = set(kw.lower() for kw in meeting["keywords"])

        # Similarity = what fraction of the meeting's keywords appear in the transcript
        overlap = len(transcript_words & meeting_keywords)
        total = len(meeting_keywords)
        similarity = overlap / total if total > 0 else 0.0

        # Only include meetings with meaningful similarity
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

    # Best matches first
    scored.sort(key=lambda m: m.similarity_score, reverse=True)
    return scored[:top_k]
