from fastapi import APIRouter, HTTPException

from app.models.discovery import CallTranscript, CallTranscriptSummary, DiscoveryAnalysisResponse
from app.services.discovery_call import analyze_call, get_datastore, get_transcript, list_transcripts

router = APIRouter()


@router.get("/transcripts", response_model=list[CallTranscriptSummary])
async def get_transcripts():
    return list_transcripts()


@router.get("/transcripts/{call_id}", response_model=CallTranscript)
async def get_transcript_detail(call_id: str):
    transcript = get_transcript(call_id)
    if not transcript:
        raise HTTPException(status_code=404, detail=f"Transcript '{call_id}' not found")
    return transcript


@router.post("/analyze/{call_id}", response_model=DiscoveryAnalysisResponse)
async def analyze_discovery_call(call_id: str):
    result = analyze_call(call_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"Transcript '{call_id}' not found")
    return result


@router.get("/datastore")
async def view_datastore():
    store = get_datastore()
    return {"total_entries": len(store), "entries": store}
