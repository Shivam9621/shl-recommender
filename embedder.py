"""
embedder.py  —  Builds and persists the ChromaDB vector store from catalog.json.
Run once: python embedder.py
After this, main.py loads the store at startup (no re-building needed).
"""

import json
import chromadb
from chromadb.utils import embedding_functions

CATALOG_FILE  = "catalog.json"
CHROMA_DIR    = "./chroma_store"          # persisted to disk
COLLECTION    = "shl_assessments"
# EMBED_MODEL   = "all-MiniLM-L6-v2"       # fast, free, 384-dim


def build_document(item: dict) -> str:
    """
    Combine all fields into one rich text string for embedding.
    The richer the text, the better semantic search works.
    Format is designed to match natural hiring-manager language.
    """
    test_type_map = {
        "A": "Ability & Aptitude",
        "B": "Biodata & Situational Judgement",
        "C": "Competencies",
        "D": "Development & 360",
        "E": "Assessment Exercises",
        "K": "Knowledge & Skills",
        "P": "Personality & Behavior",
        "S": "Simulations",
    }
    type_labels = [test_type_map.get(t, t) for t in item.get("test_types", [])]

    job_levels = ", ".join(item.get("job_levels", [])) or "All levels"
    languages  = ", ".join(item.get("languages",  [])) or "English"
    types_str  = ", ".join(type_labels) or "General"
    remote     = "supports remote testing" if item.get("remote_testing") else "no remote testing"
    adaptive   = "adaptive/IRT scoring" if item.get("adaptive_irt") else ""

    parts = [
        f"Assessment name: {item['name']}",
        f"Description: {item.get('description', '')}",
        f"Test type: {types_str}",
        f"Suitable job levels: {job_levels}",
        f"Available languages: {languages}",
        f"Remote testing: {remote}",
    ]
    if adaptive:
        parts.append(f"Scoring: {adaptive}")

    return "\n".join(parts)


def main():
    # Load catalog
    with open(CATALOG_FILE, encoding="utf-8") as f:
        catalog = json.load(f)
    print(f"Loaded {len(catalog)} assessments from {CATALOG_FILE}")

    # Set up ChromaDB with persistent storage
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Use sentence-transformers embedding function (runs locally, free)
    ef = embedding_functions.ONNXMiniLM_L6_V2()

    # Delete existing collection if re-running
    try:
        client.delete_collection(COLLECTION)
        print("Deleted existing collection (fresh rebuild)")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},   # cosine similarity
    )

    # Build documents, IDs, and metadata
    documents = []
    ids       = []
    metadatas = []

    for i, item in enumerate(catalog):
        doc = build_document(item)
        documents.append(doc)
        ids.append(str(i))
        metadatas.append({
            "name":           item["name"],
            "url":            item["url"],
            "test_types":     ",".join(item.get("test_types", [])),
            "remote_testing": str(item.get("remote_testing", False)),
            "adaptive_irt":   str(item.get("adaptive_irt",   False)),
            "job_levels":     ",".join(item.get("job_levels", [])),
        })

    # Add in batches of 100 (avoids memory spikes)
    BATCH = 100
    for start in range(0, len(documents), BATCH):
        end = min(start + BATCH, len(documents))
        collection.add(
            documents=documents[start:end],
            ids=ids[start:end],
            metadatas=metadatas[start:end],
        )
        print(f"  Embedded [{end}/{len(documents)}]")

    print(f"\n✅ ChromaDB store built at '{CHROMA_DIR}' — {collection.count()} vectors")

    # ── Quick smoke test ──────────────────────────────────────────────────────
    print("\n--- Smoke test: 'Java developer mid-level' ---")
    results = collection.query(
        query_texts=["Java developer mid-level knowledge test"],
        n_results=5,
    )
    for name, url in zip(
        results["metadatas"][0],
        results["documents"][0],
    ):
        print(f"  • {name['name']}  ({name['test_types']})")

    print("\n--- Smoke test: 'personality test for sales manager' ---")
    results = collection.query(
        query_texts=["personality test for sales manager"],
        n_results=5,
    )
    for name in results["metadatas"][0]:
        print(f"  • {name['name']}  ({name['test_types']})")


if __name__ == "__main__":
    main()