"""
check_coords.py - find rows where the scraper probably attached the WRONG page or coordinates.

Three checks:
  1. DUPLICATE: two different places closer than 150 m (e.g. Somanathapura had Belur's coordinates)
  2. OUTLIER:   a place far from the median of its own cluster (default 70 km)
  3. TITLE:     the matched Wikipedia title looks nothing like the place name
Plus a reminder list of coordinates that came from OpenStreetMap (worth a quick map check).

Usage:
    python check_coords.py ../data/places_v1.csv
"""

import csv
import math
import re
import statistics
import sys
from collections import defaultdict
from difflib import SequenceMatcher

DUP_KM = 0.15
OUTLIER_KM = 70
TITLE_SIM = 0.55


def hav(a, b):
    r = 6371.0
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    h = (math.sin((p2 - p1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(math.radians(b[1] - a[1]) / 2) ** 2)
    return 2 * r * math.asin(math.sqrt(h))


def clean(name):
    name = re.sub(r"\(.*?\)", "", name).split(",")[0]
    return re.sub(r"\s+", " ", name).strip().lower()


def main(path):
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    pts = {}
    for r in rows:
        try:
            pts[r["id"]] = (float(r["lat"]), float(r["lng"]))
        except (ValueError, KeyError):
            pass
    byid = {r["id"]: r for r in rows}
    problems = 0

    print("== 1. DUPLICATE coordinates (different places within %.0f m) ==" % (DUP_KM * 1000))
    ids = list(pts)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            d = hav(pts[a], pts[b])
            if d < DUP_KM:
                print(f"  {a} and {b} are {d * 1000:.0f} m apart -> one of them is probably wrong")
                problems += 1

    print("\n== 2. OUTLIERS (far from their own cluster) ==")
    clusters = defaultdict(list)
    for r in rows:
        if r["id"] in pts:
            clusters[r["cluster"]].append(r["id"])
    for cl, members in clusters.items():
        med = (statistics.median(pts[m][0] for m in members), statistics.median(pts[m][1] for m in members))
        for m in members:
            d = hav(pts[m], med)
            if d > OUTLIER_KM:
                print(f"  {m} ({byid[m]['name']}) is {d:.0f} km from the centre of '{cl}'")
                problems += 1

    print("\n== 3. SUSPICIOUS wiki title (low similarity to the place name) ==")
    for r in rows:
        title = (r.get("wiki_title") or "").strip()
        if not title:
            continue
        sim = SequenceMatcher(None, clean(r["name"]), title.lower()).ratio()
        contained = clean(r["name"]) in title.lower() or title.lower() in clean(r["name"])
        if sim < TITLE_SIM and not contained:
            print(f"  {r['id']}: name '{clean(r['name'])}' vs wiki '{title}' (sim {sim:.2f})")
            problems += 1

    osm = [r["id"] for r in rows if r.get("coord_source") == "nominatim"]
    if osm:
        print("\n== Coordinates from OpenStreetMap search (spot-check on a map) ==")
        print("  " + ", ".join(osm))

    print(f"\n{problems} suspicious items. Fix them via COORDS / BAD_WIKI_MATCH in apply_manual_fixes.py")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "../data/places_v1.csv")