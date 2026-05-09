import json
import time
import requests
from bs4 import BeautifulSoup

BASE_CATALOG = "https://www.shl.com/products/product-catalog/"
DETAIL_BASE  = "https://www.shl.com"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

# ── helpers ──────────────────────────────────────────────────────────────────

def get_soup(url: str) -> BeautifulSoup:
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def parse_listing_table(soup: BeautifulSoup, table_header: str) -> list[dict]:
    """
    Find the table whose <th> contains `table_header` text and parse its rows.
    Returns list of dicts with name, url, remote_testing, adaptive_irt, test_types.
    """
    assessments = []

    # Find all tables on the page
    for table in soup.find_all("table"):
        header_row = table.find("tr")
        if not header_row:
            continue
        header_text = header_row.get_text(" ", strip=True)
        if table_header not in header_text:
            continue

        # Found our table — parse body rows
        rows = table.find_all("tr")[1:]   # skip header row
        for row in rows:
            cols = row.find_all("td")
            if len(cols) < 4:
                continue

            # Col 0: name + link
            a_tag = cols[0].find("a")
            if not a_tag:
                continue
            name = a_tag.get_text(strip=True)
            href = a_tag.get("href", "")
            url  = DETAIL_BASE + href if href.startswith("/") else href

            # Col 1: Remote Testing  — contains <span> or <img> when "yes"
            remote = len(cols[1].find_all(["span", "img", "svg"])) > 0

            # Col 2: Adaptive/IRT
            adaptive = len(cols[2].find_all(["span", "img", "svg"])) > 0

            # Col 3: Test Type letters (A B C D E K P S)
            type_text = cols[3].get_text(" ", strip=True)
            test_types = [ch for ch in type_text.split() if ch in "ABCDEKPS"]

            assessments.append({
                "name": name,
                "url": url,
                "remote_testing": remote,
                "adaptive_irt": adaptive,
                "test_types": test_types,
                "description": "",
                "job_levels": [],
                "languages": [],
            })
        break   # stop after first matching table

    return assessments


def scrape_detail(session: requests.Session, assessment: dict) -> dict:
    """Enrich one assessment with description, job_levels, languages from its detail page."""
    try:
        resp = session.get(assessment["url"], headers=HEADERS, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Description: first <p> inside main content area
        # SHL detail pages have a description paragraph near the top
        for p in soup.find_all("p"):
            text = p.get_text(strip=True)
            if len(text) > 40:          # skip tiny/empty paragraphs
                assessment["description"] = text
                break

        full_text = soup.get_text(" ", strip=True)

        job_level_keywords = [
            "Director", "Entry-Level", "Executive", "Front Line Manager",
            "General Population", "Graduate", "Manager", "Mid-Professional",
            "Professional Individual Contributor", "Supervisor",
        ]
        assessment["job_levels"] = [jl for jl in job_level_keywords if jl in full_text]

        language_keywords = [
            "English (USA)", "English International", "Spanish", "French",
            "German", "Portuguese", "Portuguese (Brazil)", "Chinese Simplified",
            "Chinese Traditional", "Arabic", "Japanese", "Korean", "Dutch",
            "Italian", "Russian", "Turkish", "Polish", "Swedish", "Danish",
            "Norwegian", "Finnish",
        ]
        assessment["languages"] = [lg for lg in language_keywords if lg in full_text]

    except Exception as e:
        print(f"    ⚠  Detail error for '{assessment['name']}': {e}")

    return assessment


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    all_assessments = []
    session = requests.Session()

    # ── Phase 1: scrape all listing pages ────────────────────────────────────
    print("=== Phase 1: Scraping catalog listing pages ===")
    start = 0
    page_num = 1
    while True:
        url = f"{BASE_CATALOG}?start={start}&type=1"
        print(f"  Page {page_num} (start={start}) → {url}")
        try:
            soup = get_soup(url)
        except Exception as e:
            print(f"  ERROR fetching page: {e}")
            break

        batch = parse_listing_table(soup, "Individual Test Solutions")
        if not batch:
            print("  → No assessments found, stopping.")
            break

        all_assessments.extend(batch)
        print(f"  → Got {len(batch)} items (total so far: {len(all_assessments)})")

        start    += 12
        page_num += 1
        time.sleep(0.8)

    print(f"\nTotal listing entries: {len(all_assessments)}")

    # ── Phase 2: enrich with detail pages ────────────────────────────────────
    print("\n=== Phase 2: Scraping detail pages ===")
    for i, item in enumerate(all_assessments):
        print(f"  [{i+1}/{len(all_assessments)}] {item['name']}")
        all_assessments[i] = scrape_detail(session, item)
        time.sleep(0.6)

    # ── Save ─────────────────────────────────────────────────────────────────
    with open("catalog.json", "w", encoding="utf-8") as f:
        json.dump(all_assessments, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Saved {len(all_assessments)} assessments → catalog.json")

    # Quick sanity check
    print("\n--- Sample entry ---")
    import pprint
    pprint.pprint(all_assessments[0] if all_assessments else "EMPTY")


if __name__ == "__main__":
    main()