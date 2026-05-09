import json
import chromadb
from chromadb.utils import embedding_functions

CATALOG_FILE = "catalog.json"
CHROMA_DIR   = "./chroma_store"
COLLECTION   = "shl_assessments"


def build_if_needed():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    ef     = embedding_functions.ONNXMiniLM_L6_V2()

    try:
        col = client.get_collection(name=COLLECTION, embedding_function=ef)
        if col.count() > 0:
            print(f"[startup] Chroma already has {col.count()} vectors — skipping rebuild.")
            return
    except Exception:
        pass

    print("[startup] Building ChromaDB index from catalog.json...")

    with open(CATALOG_FILE, encoding="utf-8") as f:
        catalog = json.load(f)

    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass

    col = client.create_collection(
        name=COLLECTION,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

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

    documents, ids, metadatas = [], [], []
    for i, item in enumerate(catalog):
        type_labels = [test_type_map.get(t, t) for t in item.get("test_types", [])]
        doc = "\n".join([
            f"Assessment name: {item['name']}",
            f"Description: {item.get('description', '')}",
            f"Test type: {', '.join(type_labels)}",
            f"Suitable job levels: {', '.join(item.get('job_levels', []))}",
            f"Available languages: {', '.join(item.get('languages', []))}",
            f"Remote testing: {'yes' if item.get('remote_testing') else 'no'}",
        ])
        documents.append(doc)
        ids.append(str(i))
        metadatas.append({
            "name":           item["name"],
            "url":            item["url"],
            "test_types":     ",".join(item.get("test_types", [])),
            "remote_testing": str(item.get("remote_testing", False)),
            "adaptive_irt":   str(item.get("adaptive_irt", False)),
            "job_levels":     ",".join(item.get("job_levels", [])),
        })

    BATCH = 100
    for start in range(0, len(documents), BATCH):
        end = min(start + BATCH, len(documents))
        col.add(
            documents=documents[start:end],
            ids=ids[start:end],
            metadatas=metadatas[start:end],
        )
        print(f"[startup] Embedded [{end}/{len(documents)}]")

    print(f"[startup] Done — {col.count()} vectors ready.")
