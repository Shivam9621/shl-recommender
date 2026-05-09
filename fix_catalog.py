"""
fix_catalog.py  —  re-scrapes only the detail fields using correct selectors.
Reads existing catalog.json, fixes description + job_levels + languages, saves back.
Run once: python fix_catalog.py
"""

import json, time, requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

def scrape_detail(session: requests.Session, url: str) -> dict:
    """
    Returns dict with keys: description, job_levels, languages
    Uses the exact h4-based structure of SHL detail pages:
      <h4>Description</h4>  <p>actual text</p>
      <h4>Job levels</h4>   <p>Director, Graduate, ...</p>
    Languages come from the downloads section (e.g. 'English (USA)', 'English International')
    """
    result = {"description": "", "job_levels": [], "languages": []}
    try:
        resp = session.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # ── Description ──────────────────────────────────────────────────────
        # Find <h4> whose text is "Description", then grab the next <p>
        for h4 in soup.find_all("h4"):
            if h4.get_text(strip=True).lower() == "description":
                # next sibling that is a <p>
                nxt = h4.find_next_sibling("p")
                if nxt:
                    result["description"] = nxt.get_text(strip=True)
                break

        # ── Job levels ───────────────────────────────────────────────────────
        # Find <h4>Job levels</h4>, then the next <p> contains comma-separated levels
        for h4 in soup.find_all("h4"):
            if "job level" in h4.get_text(strip=True).lower():
                nxt = h4.find_next_sibling("p")
                if nxt:
                    raw = nxt.get_text(strip=True)
                    # raw looks like: "Director, Entry-Level, Executive, ..."
                    levels = [l.strip().rstrip(",") for l in raw.split(",") if l.strip()]
                    result["job_levels"] = [l for l in levels if l]
                break

        # ── Languages ────────────────────────────────────────────────────────
        # Languages appear as plain text labels next to download links
        # Each one is in its own <p> or text node after a download <li>
        # Strategy: grab all text in the page after "Downloads" heading,
        # then match against known language strings
        all_langs = [
            "English (USA)", "English International", "English (Australia)",
            "English (Canada)", "English (South Africa)", "English (Malaysia)",
            "English (Singapore)", "Spanish", "Latin American Spanish",
            "French", "French (Canada)", "French (Belgium)",
            "German", "Portuguese", "Portuguese (Brazil)",
            "Chinese Simplified", "Chinese Traditional",
            "Arabic", "Japanese", "Korean", "Dutch", "Italian",
            "Russian", "Turkish", "Polish", "Swedish", "Danish",
            "Norwegian", "Finnish", "Bulgarian", "Croatian", "Czech",
            "Estonian", "Flemish", "Greek", "Hungarian", "Icelandic",
            "Indonesian", "Latvian", "Lithuanian", "Romanian",
            "Serbian", "Slovak", "Thai", "Vietnamese", "Malay",
        ]
        page_text = soup.get_text(" ", strip=True)
        result["languages"] = [lg for lg in all_langs if lg in page_text]

    except Exception as e:
        print(f"    ⚠  {url}  →  {e}")

    return result


def main():
    with open("catalog.json", encoding="utf-8") as f:
        catalog = json.load(f)

    print(f"Loaded {len(catalog)} assessments. Fixing detail fields...\n")
    session = requests.Session()

    for i, item in enumerate(catalog):
        print(f"  [{i+1}/{len(catalog)}] {item['name']}")
        detail = scrape_detail(session, item["url"])
        catalog[i]["description"] = detail["description"]
        catalog[i]["job_levels"]   = detail["job_levels"]
        catalog[i]["languages"]    = detail["languages"]
        time.sleep(0.5)

    with open("catalog.json", "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)

    print(f"\n✅ catalog.json updated.")

    # ── Verification ─────────────────────────────────────────────────────────
    print("\n--- Verification: first 3 entries ---")
    import pprint
    for item in catalog[:3]:
        pprint.pprint({
            "name":        item["name"],
            "description": item["description"][:120] + "..." if len(item["description"]) > 120 else item["description"],
            "job_levels":  item["job_levels"],
            "languages":   item["languages"][:5],
        })
        print()


if __name__ == "__main__":
    main()