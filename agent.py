"""
agent.py  —  Core retrieval + LLM logic. FastAPI calls this.
"""

import json
import os
import re
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

CHROMA_DIR  = "./chroma_store"
COLLECTION  = "shl_assessments"


# ── Clients (loaded once at startup) ─────────────────────────────────────────
_groq   = Groq(api_key=os.environ["GROQ_API_KEY"])
_chroma = chromadb.PersistentClient(path=CHROMA_DIR)
# _ef     = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
_ef  = embedding_functions.ONNXMiniLM_L6_V2()
_col    = _chroma.get_collection(name=COLLECTION, embedding_function=_ef)

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an SHL assessment recommender assistant. Your ONLY job is to help hiring managers and recruiters find the right SHL assessments for their open roles.

STRICT RULES — never break these:
1. ONLY discuss SHL assessments. Refuse all off-topic requests (general hiring advice, legal questions, salary, competitor products, etc.) with: "I can only help with SHL assessment recommendations."
2. NEVER invent assessment names or URLs. Every recommendation must come from the CATALOG CONTEXT provided to you.
3. NEVER recommend on the very first turn if the query is vague (e.g. "I need an assessment", "help me hire someone"). Ask ONE focused clarifying question instead.
4. Ask at most ONE clarifying question per turn. Do not ask multiple questions at once.
5. You must respond ONLY with valid JSON — no markdown, no extra text, nothing outside the JSON object.

CLARIFICATION STRATEGY — ask about (in order of priority):
- Job role / what the person will actually do day-to-day (if completely unclear)
- Seniority level (Entry, Mid, Senior, Manager, Executive)
- What to measure: technical skills, cognitive ability, personality/behavior, or a mix?

RECOMMENDATION RULES:
- Recommend 1-10 assessments once you have enough context (role + at least one other dimension)
- A job description pasted by the user counts as full context — recommend immediately
- Use ONLY the assessments listed in the CATALOG CONTEXT section
- Copy name and url EXACTLY from the catalog context — do not modify them
- Set end_of_conversation to true only after you have provided a shortlist AND the user seems satisfied or says thanks/done

RESPONSE FORMAT — always return exactly this JSON shape and nothing else:
{
  "reply": "your conversational message here",
  "recommendations": [
    {"name": "exact name from catalog", "url": "exact url from catalog", "test_type": "K"}
  ],
  "end_of_conversation": false
}

recommendations must be [] when clarifying or refusing.
test_type is the letter code from the assessment test_types field (A/B/C/D/E/K/P/S).
"""


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve(query: str, n: int = 15) -> list[dict]:
    """Semantic search over ChromaDB. Returns top-n catalog items."""
    results = _col.query(query_texts=[query], n_results=n)
    items = []
    for meta in results["metadatas"][0]:
        items.append({
            "name":       meta["name"],
            "url":        meta["url"],
            "test_types": meta["test_types"],
            "job_levels": meta.get("job_levels", ""),
            "remote":     meta.get("remote_testing", "False"),
        })
    return items


def build_catalog_context(items: list[dict]) -> str:
    lines = ["CATALOG CONTEXT (use ONLY these assessments):"]
    for i, item in enumerate(items, 1):
        lines.append(
            f"{i}. Name: {item['name']}\n"
            f"   URL: {item['url']}\n"
            f"   Test types: {item['test_types']}\n"
            f"   Job levels: {item['job_levels']}\n"
            f"   Remote testing: {item['remote']}"
        )
    return "\n".join(lines)


def extract_search_query(messages: list[dict]) -> str:
    """Combine all user messages into one search string for retrieval."""
    user_msgs = [m["content"] for m in messages if m["role"] == "user"]
    return " ".join(user_msgs)[-500:]


# ── Guardrails ────────────────────────────────────────────────────────────────

def is_off_topic(text: str) -> bool:
    """Fast regex check before hitting the LLM."""
    patterns = [
        r"\bsalar(y|ies)\b", r"\bcompensation\b",
        r"\blegal\b", r"\blawsuit\b", r"\bdiscrimination\b",
        r"ignore (previous|above|all) instructions",
        r"forget (your|all) instructions",
        r"\byou are now\b", r"\bpretend (you|to be)\b",
        r"\bact as\b",
        r"\bpolitics\b", r"\bsports\b", r"\bweather\b",
        r"\bwrite (me )?(a |an )?(poem|story|essay|code)\b",
    ]
    t = text.lower()
    return any(re.search(p, t) for p in patterns)


# ── LLM call ──────────────────────────────────────────────────────────────────

def call_llm(messages: list[dict], catalog_context: str) -> dict:
    """Calls Groq Llama-3.3-70B and returns parsed JSON dict."""
    system = f"{SYSTEM_PROMPT}\n\n{catalog_context}"

    groq_messages = [{"role": "system", "content": system}]
    for m in messages:
        groq_messages.append({"role": m["role"], "content": m["content"]})

    response = _groq.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=groq_messages,
        temperature=0.2,
        max_tokens=1500,
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown fences the model sometimes adds
    raw = re.sub(r"^```json\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"^```\s*",     "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\s*```$",     "", raw, flags=re.MULTILINE)
    raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "reply": "I encountered an issue processing your request. Could you rephrase?",
            "recommendations": [],
            "end_of_conversation": False,
        }


# ── Main entry point ──────────────────────────────────────────────────────────

def chat(messages: list[dict]) -> dict:
    """
    Called by FastAPI for every POST /chat request.
    messages: full conversation history [{"role": "user"|"assistant", "content": "..."}]
    """
    if not messages:
        return {
            "reply": "Hello! I can help you find the right SHL assessments. What role are you hiring for?",
            "recommendations": [],
            "end_of_conversation": False,
        }

    # Guardrail: check latest user message
    last_user = next(
        (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
    )
    if is_off_topic(last_user):
        return {
            "reply": "I can only help with SHL assessment recommendations.",
            "recommendations": [],
            "end_of_conversation": False,
        }

    # Retrieve relevant catalog items
    query   = extract_search_query(messages)
    items   = retrieve(query, n=15)
    context = build_catalog_context(items)

    # Call LLM
    result = call_llm(messages, context)

    # CRITICAL URL sanitization — never let LLM hallucinate URLs
    valid_urls = {item["url"] for item in items}
    safe_recs  = [
        r for r in result.get("recommendations", [])
        if r.get("url") in valid_urls
    ]
    result["recommendations"] = safe_recs[:10]

    # Ensure schema keys always exist
    result.setdefault("reply",               "How can I help you find the right assessment?")
    result.setdefault("recommendations",     [])
    result.setdefault("end_of_conversation", False)

    return result
