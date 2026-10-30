"""
scoring.py - turn raw semantic matches into a ranked, explainable shortlist.

Why this exists: semantic search alone confuses similar-sounding places (e.g. a fort
beating a beach for "relaxing beaches"). Here we combine it with hard facts from your
dataset: interest/category match, travel month, pace and place priority.

    score = (W_SEM * semantic + W_INT * interest + W_PRI * priority) * season_mult * pace_mult

Every place also gets a list of human-readable `reasons`, which becomes the
"why this place" line in the UI and your explainability pitch for the judges.

Usage:
    from scoring import Preferences, select_places
    prefs = Preferences(days=3, interests=["heritage", "food"], month=12, pace="balanced",
                        free_text="temples and filter coffee")
    shortlist = select_places(prefs, retriever)
    for item in shortlist:
        print(item["place"]["name"], round(item["score"], 2), item["reasons"])
"""

import math
from dataclasses import dataclass, field

# ------------------------------------------------------------------ tunable weights
W_SEM, W_INT, W_PRI = 0.45, 0.40, 0.15
OUT_OF_SEASON_MULT = 0.45          # soft penalty; set to 0 to hard-exclude
PACE_MULT = {                       # user pace -> multiplier by place intensity
    "relaxed":  {"easy": 1.0, "moderate": 0.90, "hard": 0.70},
    "balanced": {"easy": 1.0, "moderate": 1.00, "hard": 0.90},
    "packed":   {"easy": 1.0, "moderate": 1.00, "hard": 1.00},
}
STOPS_PER_DAY = {"relaxed": 3, "balanced": 4, "packed": 5}
POOL_FACTOR = 1.6                   # shortlist size = days * stops/day * this (routing trims later)

# Natural-language phrase per category, used for one retrieval query per chosen interest
INTEREST_PHRASES = {
    "heritage":     "historic monuments, ancient temples and palaces",
    "culture":      "local culture, museums, markets and traditions",
    "spiritual":    "pilgrimage temples and spiritual places",
    "nature":       "scenic nature, waterfalls and landscapes",
    "wildlife":     "wildlife safari, national park and birdwatching",
    "beach":        "beaches and seaside relaxation",
    "adventure":    "trekking, adventure and outdoor activities",
    "food":         "local food, coffee and regional cuisine",
    "hill-station": "misty hill station and viewpoints",
}

MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July",
               "August", "September", "October", "November", "December"]


@dataclass
class Preferences:
    days: int = 3
    interests: list = field(default_factory=list)   # category names from the fixed list
    month: int | None = None                         # 1-12, None = unknown
    pace: str = "balanced"                           # relaxed | balanced | packed
    free_text: str = ""                              # optional natural-language request
    exclude_ids: list = field(default_factory=list)


def _minmax(values):
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [0.5] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def gather_candidates(prefs, retriever, k_text=30, k_interest=12):
    """
    Run one query for the free text plus one per chosen interest, then merge,
    keeping each place's best semantic score. This guarantees every interest
    gets candidates, not just the one that dominates the embedding space.
    """
    best = {}  # id -> (place, semantic)

    def absorb(results):
        for place, score in results:
            if place["id"] not in best or score > best[place["id"]][1]:
                best[place["id"]] = (place, score)

    interest_text = ", ".join(INTEREST_PHRASES.get(i, i) for i in prefs.interests)
    main_query = (prefs.free_text + ". " + interest_text).strip(" .") or "popular tourist attractions"
    absorb(retriever.search(main_query, k=k_text, exclude_ids=prefs.exclude_ids))
    for interest in prefs.interests:
        absorb(retriever.search(INTEREST_PHRASES.get(interest, interest),
                                k=k_interest, exclude_ids=prefs.exclude_ids))

    # Safety net: every place tagged with a chosen interest is a candidate, even if
    # retrieval missed it. It enters at the lowest semantic score seen, so category
    # tags (hard facts) can still lift it via the interest term.
    if prefs.interests and best:
        floor = min(s for _, s in best.values())
        for place in retriever.places:
            if (place["id"] not in best and place["id"] not in prefs.exclude_ids
                    and set(prefs.interests) & set(place["categories"])):
                best[place["id"]] = (place, floor)
    return list(best.values())


def score_candidates(prefs, candidates):
    if not candidates:
        return []
    sems = _minmax([s for _, s in candidates])
    chosen = set(prefs.interests)
    results = []

    for (place, _raw), sem in zip(candidates, sems):
        reasons = []

        # Interest match
        overlap = chosen & set(place["categories"])
        if chosen:
            interest = min(1.0, 0.7 + 0.3 * (len(overlap) - 1)) if overlap else 0.0
            if overlap:
                reasons.append("matches your interest in " + " & ".join(sorted(overlap)))
        else:
            interest = 0.5

        priority = 1.0 if place["priority"] == "M" else 0.5
        if place["priority"] == "M":
            reasons.append("a must-see in the region")

        # Season
        season_mult = 1.0
        if prefs.month:
            if prefs.month in place["best_months"]:
                reasons.append(f"good to visit in {MONTH_NAMES[prefs.month - 1]}")
            else:
                season_mult = OUT_OF_SEASON_MULT
                reasons.append(f"best outside {MONTH_NAMES[prefs.month - 1]} (off-season)")

        # Pace
        pace_mult = PACE_MULT.get(prefs.pace, PACE_MULT["balanced"])[place["pace_intensity"]]
        if pace_mult < 1.0:
            reasons.append(f"{place['pace_intensity']} activity for a {prefs.pace} trip")

        score = (W_SEM * sem + W_INT * interest + W_PRI * priority) * season_mult * pace_mult
        if score <= 0:
            continue
        results.append({
            "place": place,
            "score": score,
            "breakdown": {"semantic": round(sem, 3), "interest": round(interest, 3),
                          "priority": priority, "season": season_mult, "pace": pace_mult},
            "reasons": reasons,
        })

    results.sort(key=lambda r: -r["score"])
    return results


def select_places(prefs, retriever, pool_factor=POOL_FACTOR):
    """
    Full pipeline: gather -> score -> shortlist, ensuring each chosen interest is represented.
    The shortlist is a bit larger than needed; routing.py trims it to fit each day's time.
    """
    scored = score_candidates(prefs, gather_candidates(prefs, retriever))
    pool_size = min(len(scored), math.ceil(
        prefs.days * STOPS_PER_DAY.get(prefs.pace, 4) * pool_factor))

    chosen_ids, pool = set(), []
    # 1) guarantee a minimum per interest
    if prefs.interests:
        per_interest = max(2, pool_size // (2 * len(prefs.interests)))
        for interest in prefs.interests:
            taken = 0
            for item in scored:
                pid = item["place"]["id"]
                if pid in chosen_ids or interest not in item["place"]["categories"]:
                    continue
                pool.append(item); chosen_ids.add(pid); taken += 1
                if taken >= per_interest or len(pool) >= pool_size:
                    break
    # 2) fill the rest by overall score
    for item in scored:
        if len(pool) >= pool_size:
            break
        if item["place"]["id"] not in chosen_ids:
            pool.append(item); chosen_ids.add(item["place"]["id"])

    pool.sort(key=lambda r: -r["score"])
    return pool