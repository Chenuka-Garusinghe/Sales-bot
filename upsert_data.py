# Import the Pinecone library
import asyncio
import json
import os

from pinecone import Pinecone
from googletrans import Translator
from tqdm import tqdm

DATA_PATH = "salestalk-dataset/data/japanese-salestalk-dataset_v1.json"


async def translate_text(translator: Translator, transcript: str) -> str:
    result = await translator.translate(transcript, dest="en")
    return result.text


async def model_data(translator: Translator, data: dict) -> dict:
    """Convert a single dialogue JSON object into a Pinecone record."""
    # Build the full transcript as readable text
    transcript_lines = []
    for utt in data["utterances"]:
        if utt["speaker"] == "system":
            continue
        transcript_lines.append(f"{utt['speaker']}: {utt['message']}")
    transcript = await translate_text(translator, "\n".join(transcript_lines))

    # Extract purchase intention scores
    evals = {e["label"]: e["answer"] for e in data.get("user_dialogue_evals", [])}

    record = {
        "_id": str(data["dialogue_id"]),
        "text": transcript,
        "sales_id": data["sales_id"],
        "user_id": data["user_id"],
    }
    before = evals.get("before_purchase_intention")
    after = evals.get("after_purchase_intention")
    if before is not None:
        record["before_purchase_intention"] = before
    if after is not None:
        record["after_purchase_intention"] = after
    return record


async def load_data(path: str) -> list[dict]:
    """Load all dialogues from a JSONL file."""
    records = []
    async with Translator() as translator:
        lines = open(path).readlines()
        for line in tqdm(lines, desc="Translating dialogues"):
            dialogue = json.loads(line)
            records.append(await model_data(translator, dialogue))
    return records


def main():
    # Initialize a Pinecone client with your API key
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    # pc.Index("sales-bot-db-demo").delete(delete_all=True, namespace="salestalk")

    # Create a dense index with integrated embedding
    index_name = "sales-bot-db-demo"
    if not pc.has_index(index_name):
        pc.create_index_for_model(
            name=index_name,
            cloud="aws",
            region="us-east-1",
            embed={"model": "llama-text-embed-v2", "field_map": {"text": "text"}},
        )

    index = pc.Index(index_name)
    records = asyncio.run(load_data(DATA_PATH))
    # Upsert in batches of 20
    for i in tqdm(range(0, len(records), 20), desc="Upserting to Pinecone"):
        batch = records[i : i + 20]
        index.upsert_records(namespace="salestalk", records=batch)
    print(f"Upserted {len(records)} records.")


if __name__ == "__main__":
    main()
