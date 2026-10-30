"""
data_loader.py - load and validate places_v1.csv.

Returns clean Python dicts the rest of the backend can trust:
  * numbers are floats/ints, months are lists of ints, categories are lists of strings
  * every place gets a `search_text` field used for embeddings
  * places with unusable data are reported and skipped (or raise in strict mode)

CLI check:
    python data_loader.py ../data/places_v1.csv
"""

import csv
import sys
from collections import Counter
from pathlib import Path

VALID_CATEGORIES = {
    "heritage", "nature", "wildlife", "beach", "spiritual",
    "adventure", "food", "hill-station", "culture",
}
VALID_PACE = {"easy", "moderate", "hard"}

# Rough bounding box of Karnataka
LAT_MIN, LAT_MAX = 11.5, 18.5
LNG_MIN, LNG_MAX = 74.0, 78.6

REQUIRED = ["id", "name", "cluster", "lat", "lng", "suggested_categories",
            "avg_visit_hours", "best_months", "pace_intensity"]


def _split(value, sep=";"):
    return [p.strip() for p in (value or "").replace(",", sep).split(sep) if p.strip()]


def _num(value, cast=float):
    try:
        return cast(str(value).strip())
    except (ValueError, TypeError):
        return None


def build_search_text(p):
    """Text that gets embedded. Rich natural text retrieves better than keywords."""
    parts = [p["name"] + ".", "Type: " + ", ".join(p["categories"]) + "."]
    parts.append(f"Region: {p['cluster']} ({p['town']})." if p.get("town") else f"Region: {p['cluster']}.")
    if p.get("tags"):
        parts.append(p["tags"] + ".")
    body = p.get("wiki_extract") or p.get("voyage_extract") or ""
    if body:
        parts.append(body[:700])
    return " ".join(parts)


def load_places(path, strict=False):
    """
    Returns (places, issues).
    issues is a list of (severity, place_id, message) with severity 'ERROR' or 'WARN'.
    ERROR rows are skipped. In strict mode any ERROR raises ValueError.
    """
    places, issues, seen_ids = [], [], set()

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing_cols = [c for c in REQUIRED if c not in (reader.fieldnames or [])]
        if missing_cols:
            raise ValueError(f"CSV is missing required columns: {missing_cols}")
        rows = list(reader)

    for row in rows:
        pid = (row.get("id") or "").strip()
        errs, warns = [], []

        if not pid:
            issues.append(("ERROR", "?", "row without id"))
            continue
        if pid in seen_ids:
            errs.append("duplicate id")
        seen_ids.add(pid)

        lat, lng = _num(row.get("lat")), _num(row.get("lng"))
        if lat is None or lng is None:
            errs.append("missing lat/lng")
        elif not (LAT_MIN <= lat <= LAT_MAX and LNG_MIN <= lng <= LNG_MAX):
            errs.append(f"coordinates ({lat}, {lng}) outside Karnataka")

        cats = [c.lower() for c in _split(row.get("suggested_categories"))]
        bad = [c for c in cats if c not in VALID_CATEGORIES]
        if not cats:
            errs.append("no categories")
        if bad:
            errs.append(f"unknown categories {bad}")

        hours = _num(row.get("avg_visit_hours"))
        if hours is None or hours <= 0:
            errs.append("bad avg_visit_hours")

        try:
            months = sorted({int(m) for m in _split(row.get("best_months"))})
            if not months or months[0] < 1 or months[-1] > 12:
                raise ValueError
        except ValueError:
            errs.append("bad best_months")
            months = []

        pace = (row.get("pace_intensity") or "").strip().lower()
        if pace not in VALID_PACE:
            errs.append(f"bad pace '{pace}'")

        # Soft checks
        fee = _num(row.get("entry_fee_inr"))
        if fee is None:
            warns.append("entry fee unknown (treated as 0 in budget)")
        if not (row.get("wiki_extract") or row.get("voyage_extract") or row.get("voyage_description") or row.get("tags")):
            warns.append("no descriptive text: retrieval quality will be weaker")

        for e in errs:
            issues.append(("ERROR", pid, e))
        for w in warns:
            issues.append(("WARN", pid, w))
        if errs:
            continue

        place = {
            "id": pid,
            "name": row["name"].strip(),
            "district": (row.get("district") or "").strip(),
            "cluster": row["cluster"].strip(),
            "town": (row.get("town") or "").strip(),
            "lat": lat, "lng": lng,
            "categories": cats,
            "priority": (row.get("priority") or "O").strip().upper(),
            "avg_visit_hours": hours,
            "best_months": months,
            "pace_intensity": pace,
            "entry_fee_inr": fee if fee is not None else 0.0,
            "fee_known": fee is not None,
            "tags": (row.get("tags") or "").strip(),
            "wiki_extract": (row.get("wiki_extract") or "").strip(),
            "voyage_extract": (row.get("voyage_extract") or row.get("voyage_description") or "").strip(),
            "source_url": (row.get("source_url") or "").strip(),
        }
        place["search_text"] = build_search_text(place)
        places.append(place)

    if strict and any(sev == "ERROR" for sev, _, _ in issues):
        raise ValueError("Validation errors:\n" + "\n".join(
            f"  {pid}: {msg}" for sev, pid, msg in issues if sev == "ERROR"))
    return places, issues


def summarize(places, issues):
    print(f"Loaded {len(places)} valid places")
    print("By cluster :", dict(Counter(p["cluster"] for p in places)))
    cat_counts = Counter(c for p in places for c in p["categories"])
    print("By category:", dict(cat_counts))
    empty = VALID_CATEGORIES - set(cat_counts)
    if empty:
        print("Categories with NO places:", sorted(empty))

    errors = [i for i in issues if i[0] == "ERROR"]
    warns = [i for i in issues if i[0] == "WARN"]
    print(f"\n{len(errors)} errors (rows skipped), {len(warns)} warnings")
    for sev, pid, msg in errors:
        print(f"  ERROR {pid}: {msg}")
    warn_by_msg = Counter(msg for _, _, msg in warns)
    for msg, n in warn_by_msg.items():
        print(f"  WARN  {n} places: {msg}")


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).parent.parent / "data" / "places_v1.csv")
    loaded, found = load_places(csv_path)
    summarize(loaded, found)