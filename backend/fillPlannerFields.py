#!/usr/bin/env python3
"""
fill_planner_fields.py - fill avg_visit_hours, best_months and pace_intensity.

These three fields are what the routing logic needs to pack a day and filter by
travel month. They are NOT reliably available online, so this script assigns
sensible values in three layers (first match wins):

    1. OVERRIDES    - hand-set values for marquee places (by id)
    2. NAME RULES   - keyword rules on the place name ("falls", "temple", "national park" ...)
    3. CATEGORY     - defaults from the primary category (first one listed)

Every filled row gets a `v1_fields_source` value ("override", "rule:<name>" or
"category:<cat>") so you know which rows deserve a manual sanity check.
The "category:*" rows are the weakest guesses: review those first.

Existing non-empty values are kept unless you pass --force.

Usage:
    python fill_planner_fields.py                                  # defaults below
    python fill_planner_fields.py --in places_scraped.csv --out places_v1.csv
    python fill_planner_fields.py --force                          # overwrite existing values

Formats written:
    avg_visit_hours : number of hours, e.g. 1.5   (excludes travel time)
    best_months     : semicolon list of month numbers, e.g. 10;11;12;1;2;3
    pace_intensity  : easy | moderate | hard
"""

import argparse
import csv
import re
import sys
from collections import Counter

# ----------------------------------------------------------------- month profiles
# Approximate, based on typical Karnataka weather and access (monsoon = Jun-Sep).
PROFILES = {
    "all_year": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    "mild":     [9, 10, 11, 12, 1, 2, 3],      # heritage / city sightseeing: avoid peak summer heat
    "winter":   [10, 11, 12, 1, 2, 3],         # cool, dry, clear
    "hills":    [9, 10, 11, 12, 1, 2, 3],      # hill stations: post-monsoon green to winter
    "monsoon":  [6, 7, 8, 9, 10, 11],          # waterfalls at their fullest
    "wildlife": [11, 12, 1, 2, 3, 4, 5],       # dry season: animals visit water, easier sightings
    "trek":     [10, 11, 12, 1, 2],            # post-monsoon, avoids leeches/landslides/heat
    "beach":    [10, 11, 12, 1, 2, 3],         # sea is calm; water activities stop in monsoon
}

# ----------------------------------------------------------------- category defaults
# primary category -> (visit_hours, pace, month_profile)
CATEGORY_DEFAULTS = {
    "heritage":     (1.5,  "easy",     "mild"),
    "culture":      (1.0,  "easy",     "mild"),
    "spiritual":    (1.25, "easy",     "all_year"),
    "nature":       (1.5,  "easy",     "mild"),
    "wildlife":     (3.0,  "moderate", "wildlife"),
    "beach":        (1.5,  "easy",     "beach"),
    "adventure":    (2.5,  "moderate", "trek"),
    "food":         (0.75, "easy",     "all_year"),
    "hill-station": (2.0,  "moderate", "hills"),
}
FALLBACK = (1.5, "easy", "mild")

# ----------------------------------------------------------------- keyword rules on name
# (regex, rule_name, visit_hours, pace, month_profile); first match wins
NAME_RULES = [
    (r"national park|wildlife sanctuary|tiger reserve|reservoir", "safari",      3.0,  "moderate", "wildlife"),
    (r"elephant camp",                                            "camp",        2.5,  "easy",     "mild"),
    (r"zoo|zoological",                                           "zoo",         3.0,  "easy",     "mild"),
    (r"bird sanctuary",                                           "birds",       1.5,  "easy",     "winter"),
    (r"falls",                                                    "waterfall",   1.0,  "easy",     "monsoon"),
    (r"peak|trek",                                                "peak_trek",   3.0,  "hard",     "trek"),
    (r"viewpoint|end point|raja's seat",                          "viewpoint",   1.0,  "easy",     "hills"),
    (r"island",                                                   "island",      2.5,  "easy",     "beach"),
    (r"beach",                                                    "beach",       1.5,  "easy",     "beach"),
    (r"palace",                                                   "palace",      1.5,  "easy",     "mild"),
    (r"fort|battery",                                             "fort",        1.0,  "easy",     "mild"),
    (r"cathedral|church|chapel",                                  "church",      0.75, "easy",     "all_year"),
    (r"market",                                                   "market",      1.0,  "easy",     "all_year"),
    (r"hotel|sweet",                                              "food",        0.75, "easy",     "all_year"),
    (r"monastery",                                                "monastery",   1.5,  "easy",     "all_year"),
    (r"plantation|estate",                                        "plantation",  2.5,  "easy",     "winter"),
    (r"museum|gallery",                                           "museum",      1.0,  "easy",     "all_year"),
    (r"dam|gardens",                                              "dam_garden",  1.5,  "easy",     "mild"),
    (r"temple|matha|peetha|mutt",                                 "temple",      1.25, "easy",     "all_year"),
    (r"hills|betta",                                              "hill",        2.0,  "moderate", "hills"),
]

# ----------------------------------------------------------------- manual overrides (by id)
# id -> (visit_hours, pace, month_profile OR explicit list of months)
# Approximate values from typical visitor experience. Edit freely: this is YOUR domain knowledge.
OVERRIDES = {
    # Mysuru city + outskirts
    "mys_palace":          (2.0,  "easy",     "mild"),
    "mys_chamundi":        (2.0,  "easy",     "all_year"),
    "mys_zoo":             (3.0,  "easy",     "mild"),
    "mys_brindavan":       (2.0,  "easy",     "mild"),
    "mys_somanathapura":   (1.5,  "easy",     "mild"),
    "mys_srirangapatna":   (2.5,  "easy",     "mild"),
    "mys_ranganathittu":   (1.5,  "easy",     "winter"),
    "mys_talakadu":        (2.0,  "easy",     "mild"),
    # Kabini-Bandipur
    "kab_backwaters":      (3.0,  "easy",     "wildlife"),
    "kab_nagarahole":      (3.0,  "moderate", "wildlife"),
    "kab_bandipur":        (3.0,  "moderate", "wildlife"),
    "kab_brhills":         (3.0,  "moderate", "wildlife"),
    "kab_gopalaswamy":     (2.0,  "easy",     "hills"),
    "kab_mmhills":         (2.5,  "easy",     "all_year"),
    "kab_shivanasamudra":  (1.5,  "easy",     "monsoon"),
    # Coorg
    "crg_abbey":           (1.0,  "easy",     "monsoon"),
    "crg_rajaseat":        (1.0,  "easy",     "hills"),
    "crg_mandalpatti":     (2.5,  "moderate", "hills"),
    "crg_dubare":          (2.5,  "easy",     "winter"),
    "crg_namdroling":      (1.5,  "easy",     "all_year"),
    "crg_talakaveri":      (2.0,  "moderate", "hills"),
    "crg_coffee":          (2.5,  "easy",     "winter"),
    "crg_tadiandamol":     (5.0,  "hard",     "trek"),
    "crg_nisargadhama":    (1.5,  "easy",     "mild"),
    # Chikkamagaluru
    "chk_mullayanagiri":   (2.5,  "moderate", "trek"),
    "chk_budangiri":       (2.5,  "moderate", "hills"),
    "chk_hebbe":           (3.5,  "moderate", "trek"),
    "chk_kemmangundi":     (2.5,  "easy",     "hills"),
    "chk_kudremukh":       (6.0,  "hard",     "trek"),
    "chk_sringeri":        (2.0,  "easy",     "all_year"),
    "chk_horanadu":        (1.5,  "easy",     "all_year"),
    "chk_bhadra":          (3.0,  "moderate", "wildlife"),
    # Hassan
    "has_belur":           (1.5,  "easy",     "mild"),
    "has_halebidu":        (1.5,  "easy",     "mild"),
    "has_shravanabelagola": (2.5, "moderate", "winter"),   # ~600 steps, barefoot
    "has_manjarabad":      (1.0,  "easy",     "hills"),
    # Mangaluru
    "mng_dharmasthala":    (2.5,  "easy",     "all_year"),
    "mng_kukke":           (1.5,  "easy",     "all_year"),
    "mng_pilikula":        (3.0,  "easy",     "mild"),
    "mng_sultanbattery":   (0.75, "easy",     "mild"),
    # Udupi
    "udp_krishna":         (1.5,  "easy",     "all_year"),
    "udp_malpe":           (2.0,  "easy",     "beach"),
    "udp_stmarys":         (2.5,  "easy",     "beach"),
    "udp_kollur":          (1.5,  "easy",     "all_year"),
    "udp_manipal":         (1.0,  "easy",     "mild"),
}

VALID_PACE = {"easy", "moderate", "hard"}


# ----------------------------------------------------------------- logic
def to_months(spec):
    """Profile name or explicit list -> sorted list of month numbers."""
    if isinstance(spec, str):
        return PROFILES[spec]
    return sorted(set(spec))


def fmt_hours(h):
    return f"{h:g}"


def primary_category(row):
    cats = (row.get("categories") or row.get("suggested_categories") or "").replace(",", ";")
    parts = [c.strip().lower() for c in cats.split(";") if c.strip()]
    return parts[0] if parts else ""


def decide(row):
    """Return (hours, pace, months, source) for a row."""
    pid = row.get("id", "")
    if pid in OVERRIDES:
        h, p, m = OVERRIDES[pid]
        return h, p, to_months(m), "override"

    name = row.get("name", "").lower()
    for pattern, rule_name, h, p, prof in NAME_RULES:
        if re.search(pattern, name):
            return h, p, to_months(prof), f"rule:{rule_name}"

    cat = primary_category(row)
    if cat in CATEGORY_DEFAULTS:
        h, p, prof = CATEGORY_DEFAULTS[cat]
        return h, p, to_months(prof), f"category:{cat}"
    h, p, prof = FALLBACK
    return h, p, to_months(prof), "category:unknown"


def is_empty(value):
    return value is None or str(value).strip() == ""


def main():
    ap = argparse.ArgumentParser(description="Fill visit hours, best months and pace for each place.")
    ap.add_argument("--in", dest="infile", default="places_scraped.csv")
    ap.add_argument("--out", dest="outfile", default="places_v1.csv")
    ap.add_argument("--force", action="store_true", help="overwrite values that are already filled")
    args = ap.parse_args()

    try:
        with open(args.infile, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            fieldnames = list(reader.fieldnames)
    except FileNotFoundError:
        sys.exit(f"Input file not found: {args.infile}")

    for col in ("avg_visit_hours", "best_months", "pace_intensity", "v1_fields_source"):
        if col not in fieldnames:
            fieldnames.append(col)

    sources = Counter()
    for row in rows:
        h, p, months, src = decide(row)
        touched = False
        if args.force or is_empty(row.get("avg_visit_hours")):
            row["avg_visit_hours"] = fmt_hours(h); touched = True
        if args.force or is_empty(row.get("best_months")):
            row["best_months"] = ";".join(map(str, months)); touched = True
        if args.force or is_empty(row.get("pace_intensity")):
            row["pace_intensity"] = p; touched = True
        if touched:
            row["v1_fields_source"] = src
        sources[(row.get("v1_fields_source") or "kept").split(":")[0]] += 1

    # Sanity checks on the final values
    problems = []
    for row in rows:
        try:
            if float(row["avg_visit_hours"]) <= 0:
                problems.append(f"{row['id']}: visit hours must be > 0")
        except ValueError:
            problems.append(f"{row['id']}: bad avg_visit_hours '{row['avg_visit_hours']}'")
        if row["pace_intensity"] not in VALID_PACE:
            problems.append(f"{row['id']}: bad pace '{row['pace_intensity']}'")
        try:
            ms = [int(x) for x in row["best_months"].split(";")]
            if not ms or any(m < 1 or m > 12 for m in ms):
                raise ValueError
        except ValueError:
            problems.append(f"{row['id']}: bad best_months '{row['best_months']}'")

    with open(args.outfile, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print(f"Filled {len(rows)} rows -> {args.outfile}")
    print("Source breakdown:", dict(sources))
    weak = [r["name"] for r in rows if str(r.get("v1_fields_source", "")).startswith("category")]
    if weak:
        print(f"\nWeakest guesses (category defaults only), review these {len(weak)}:")
        for n in weak:
            print("  -", n)
    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print("  !", p)
        sys.exit(1)
    print("\nAll values passed sanity checks.")


if __name__ == "__main__":
    main()