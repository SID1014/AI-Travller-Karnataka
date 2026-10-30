"""
stays.py - city price alerts and a rough accommodation estimate from data/city_stay_costs.csv.

Deliberately simple for V1: one base cluster per night, rooms = ceil(travelers / 2).
Your own stays idea can replace estimate_stays() without touching anything else.
"""

import csv
import math
from pathlib import Path

TIERS = ("budget", "mid", "luxury")
LEVEL_ORDER = {"low": 0, "moderate": 1, "high": 2, "very_high": 3}


def _int(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def load_stays(path):
    rows = {}
    p = Path(path)
    if not p.exists():
        return rows
    for r in csv.DictReader(open(p, newline="", encoding="utf-8")):
        r["peak"] = [int(m) for m in (r.get("peak_months") or "").split(";") if m.strip().isdigit()]
        for k in ("hostel_dorm_from_inr", "budget_min_inr", "budget_max_inr", "mid_min_inr",
                  "mid_max_inr", "luxury_min_inr"):
            r[k] = _int(r.get(k))
        rows[r["cluster"]] = r
    return rows


def price_alerts(stays, clusters, month):
    """Alerts for the clusters in this trip. 'high'/'very_high' places are listed first."""
    out = []
    for c in clusters:
        r = stays.get(c)
        if not r:
            continue
        out.append({
            "cluster": c,
            "price_level": r["price_level"],
            "alert": r["price_alert"],
            "in_peak_season": bool(month and month in r["peak"]),
            "hostel_dorm_from_inr": r["hostel_dorm_from_inr"],
            "budget_range_inr": [r["budget_min_inr"], r["budget_max_inr"]],
            "mid_range_inr": [r["mid_min_inr"], r["mid_max_inr"]],
            "confidence": r["confidence"],
            "activity_costs_note": r.get("activity_costs_note", ""),
        })
    out.sort(key=lambda a: -LEVEL_ORDER.get(a["price_level"], 0))
    return out


def tier_price(row, tier):
    """(price per room per night, label). Luxury uses the 'from' price."""
    if tier == "budget" and row["budget_min_inr"] is not None:
        return (row["budget_min_inr"] + (row["budget_max_inr"] or row["budget_min_inr"])) // 2, "typical"
    if tier == "mid" and row["mid_min_inr"] is not None:
        return (row["mid_min_inr"] + (row["mid_max_inr"] or row["mid_min_inr"])) // 2, "typical"
    if tier == "luxury" and row["luxury_min_inr"] is not None:
        return row["luxury_min_inr"], "from"
    return None, ""


def estimate_stays(stays, day_bases, tier, travelers):
    """
    day_bases: base cluster per day, in order. Nights = days - 1 (night i is spent in day i's base).
    Returns {"total_inr", "rooms", "nights", "lines": [...], "note"} or None.
    """
    if tier not in TIERS or len(day_bases) < 2:
        return None
    rooms = max(1, math.ceil(travelers / 2))
    lines, total = [], 0
    for night, base in enumerate(day_bases[:-1], 1):
        row = stays.get(base)
        price, label = tier_price(row, tier) if row else (None, "")
        if price is None:
            lines.append({"night": night, "cluster": base, "per_room_inr": None, "subtotal_inr": None})
            continue
        sub = price * rooms
        total += sub
        lines.append({"night": night, "cluster": base, "per_room_inr": price, "label": label, "subtotal_inr": sub})
    return {"tier": tier, "rooms": rooms, "nights": len(day_bases) - 1, "total_inr": total, "lines": lines,
            "note": "Rough estimate from typical nightly prices; peak weekends can cost much more."}