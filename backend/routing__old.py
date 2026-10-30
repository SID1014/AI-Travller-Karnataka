"""
routing.py - turn the scored shortlist into a feasible, day-by-day itinerary.

Pipeline (all deterministic, no LLM, so it works even if every API is down):
  1. Build each day greedily: seed with the best-scored place, then keep adding the place with
     the best (score - ALPHA * extra_hours) while the day still fits its time, stop-count and
     driving limits. Nearby places are cheap to add; far-away ones must be worth the drive.
  2. Order the days so consecutive days are geographically close (nearest-centroid chain).
  3. Order the stops inside each day by brute-force shortest open path (n <= 7, so it's optimal).
  4. Build a clock timeline: start time by pace, travel legs, visit durations, a lunch break.
  5. Return JSON-friendly output plus warnings (long transfers, late finishes, empty days).

Travel time model (no API needed): haversine distance x 1.4 road factor, at 40 km/h on plains
and 30 km/h on ghat/hill roads (Coorg, Chikkamagaluru, or any coast <-> inland leg).

Stays are NOT modelled yet. Each day exposes `base_cluster`, which is the hook for it.

CLI demo:
    python routing.py ../data/places_v1.csv --days 3 --interests heritage,food --month 12 --pace balanced
"""

import argparse
import itertools
import math
from collections import Counter

from scoring import STOPS_PER_DAY, Preferences, select_places

# ------------------------------------------------------------------ tunables
ROAD_FACTOR_PLAIN, SPEED_PLAIN = 1.25, 45.0     # km road per km straight-line, km/h
ROAD_FACTOR_GHAT,  SPEED_GHAT  = 1.30, 38.0
HILLY = {"Coorg", "Chikkamagaluru"}
COASTAL = {"Mangaluru", "Udupi-Malpe"}

DAY_HOURS = {"relaxed": 7.5, "balanced": 9.0, "packed": 10.5}   # start-to-finish window, lunch included
MAX_TRAVEL = {"relaxed": 3.0, "balanced": 4.0, "packed": 5.0}   # driving hours per day
START_TIME = {"relaxed": 9.5, "balanced": 9.0, "packed": 8.0}
LUNCH_H, LUNCH_FROM = 1.0, 12.5
MAX_LEG = {"relaxed": 1.5, "balanced": 2.0, "packed": 2.5}   # longest single drive inside a day (hours)
ALPHA = 0.15            # score points lost per extra DRIVING hour added to a day
POOL_FACTOR_ROUTING = 2.5   # routing sees a bigger shortlist so each day can be filled locally
LATE_FINISH = 20.0      # warn if a day ends after 8 pm
LONG_TRANSFER = 4.0     # warn if moving between days takes longer than this
BRUTE_FORCE_MAX = 7


# ------------------------------------------------------------------ travel model
def haversine_km(a, b):
    r = 6371.0
    p1, p2 = math.radians(a["lat"]), math.radians(b["lat"])
    dphi = p2 - p1
    dl = math.radians(b["lng"] - a["lng"])
    h = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def leg(a, b):
    """(hours, road_km) between two points that carry 'lat', 'lng', 'cluster'."""
    ghat = (a["cluster"] in HILLY or b["cluster"] in HILLY
            or ((a["cluster"] in COASTAL) != (b["cluster"] in COASTAL)))
    km = haversine_km(a, b) * (ROAD_FACTOR_GHAT if ghat else ROAD_FACTOR_PLAIN)
    return km / (SPEED_GHAT if ghat else SPEED_PLAIN), km


def fmt_time(h):
    h = round(h * 60) / 60
    return f"{int(h):02d}:{int(round((h - int(h)) * 60)) % 60:02d}"


# ------------------------------------------------------------------ ordering
def path_travel(order, origin=None):
    total = leg(origin, order[0])[0] if origin else 0.0
    for a, b in zip(order, order[1:]):
        total += leg(a, b)[0]
    return total


def best_order(places, origin=None):
    """Shortest open path through the places (optimal for n <= 7, nearest-neighbour above)."""
    if len(places) == 1:
        return list(places)
    if len(places) <= BRUTE_FORCE_MAX:
        return list(min(itertools.permutations(places), key=lambda o: path_travel(o, origin)))
    remaining, cur = list(places), origin
    order = []
    if cur is None:
        cur = remaining.pop(0)
        order.append(cur)
    while remaining:
        nxt = min(remaining, key=lambda p: leg(cur, p)[0])
        remaining.remove(nxt)
        order.append(nxt)
        cur = nxt
    return order


def stop_weight(place):
    """Quick stops (markets, food, chapels) count half toward the per-day stop limit."""
    return 0.5 if place["avg_visit_hours"] <= 1.0 else 1.0


def day_load(places):
    """(total_hours incl. lunch, travel_hours, longest_leg_hours) for the best ordering."""
    order = best_order(places)
    legs = [leg(a, b)[0] for a, b in zip(order, order[1:])]
    visit = sum(p["avg_visit_hours"] for p in places)
    travel = sum(legs)
    return visit + travel + LUNCH_H, travel, (max(legs) if legs else 0.0)


def centroid(places):
    return {"lat": sum(p["lat"] for p in places) / len(places),
            "lng": sum(p["lng"] for p in places) / len(places),
            "cluster": Counter(p["cluster"] for p in places).most_common(1)[0][0]}


# ------------------------------------------------------------------ day building
def build_days(items, days, pace, start_cluster=None):
    cap, max_travel, max_leg = DAY_HOURS[pace], MAX_TRAVEL[pace], MAX_LEG[pace]
    max_stops = STOPS_PER_DAY[pace]
    remaining = sorted(items, key=lambda it: -it["score"])
    built = []

    for d in range(days):
        if not remaining:
            break
        seed = remaining[0]
        if d == 0 and start_cluster:
            in_start = [it for it in remaining if it["place"]["cluster"] == start_cluster]
            if in_start:
                seed = in_start[0]
        day = [seed]
        remaining.remove(seed)

        while remaining:
            cur_weight = sum(stop_weight(it["place"]) for it in day)
            _, cur_travel, _ = day_load([it["place"] for it in day])
            best, best_gain = None, None
            for cand in remaining:
                if cur_weight + stop_weight(cand["place"]) > max_stops:
                    continue
                load, travel, longest = day_load([it["place"] for it in day] + [cand["place"]])
                if load > cap or travel > max_travel or longest > max_leg:
                    continue
                gain = cand["score"] - ALPHA * (travel - cur_travel)
                if best is None or gain > best_gain:
                    best, best_gain = cand, gain
            if best is None or best_gain <= 0:
                break
            day.append(best)
            remaining.remove(best)
        built.append(day)
    return built, remaining


def order_days(day_groups, origin=None):
    """Chain days by nearest centroid so consecutive days are geographically close."""
    if len(day_groups) <= 1:
        return day_groups
    cents = [centroid([it["place"] for it in g]) for g in day_groups]
    idx = list(range(len(day_groups)))
    if origin:
        first = min(idx, key=lambda i: leg(origin, cents[i])[0])
    else:
        first = max(idx, key=lambda i: sum(it["score"] for it in day_groups[i]))
    chain = [first]
    idx.remove(first)
    while idx:
        nxt = min(idx, key=lambda i: leg(cents[chain[-1]], cents[i])[0])
        chain.append(nxt)
        idx.remove(nxt)
    return [day_groups[i] for i in chain]


# ------------------------------------------------------------------ timeline
def build_timeline(order_items, pace, origin=None):
    places = [it["place"] for it in order_items]
    t = START_TIME[pace]
    stops, lunch_done, lunch_after, travel_total = [], False, None, 0.0
    origin_leg = None

    if origin:
        h, km = leg(origin, places[0])
        origin_leg = {"hours": round(h, 2), "km": round(km)}
        t += h
        travel_total += h

    for i, it in enumerate(order_items):
        p = it["place"]
        h = km = 0.0
        if i > 0:
            h, km = leg(places[i - 1], p)
            if not lunch_done and (t >= LUNCH_FROM or (t >= 11.5 and t + h > 13.5)):
                t += LUNCH_H
                lunch_done, lunch_after = True, i - 1
            t += h
            travel_total += h
        arrive = t
        t += p["avg_visit_hours"]
        stops.append({
            "id": p["id"], "name": p["name"], "cluster": p["cluster"], "town": p["town"],
            "lat": p["lat"], "lng": p["lng"], "categories": p["categories"],
            "arrive": fmt_time(arrive), "depart": fmt_time(t),
            "visit_hours": p["avg_visit_hours"],
            "travel_from_prev_hours": round(h, 2), "travel_from_prev_km": round(km),
            "pace_intensity": p["pace_intensity"],
            "entry_fee_inr": p["entry_fee_inr"], "fee_known": p["fee_known"],
            "score": round(it["score"], 3), "reasons": it["reasons"],
        })
    return {"stops": stops, "end_hour": t, "end_time": fmt_time(t),
            "start_time": fmt_time(START_TIME[pace]), "travel_hours": round(travel_total, 2),
            "lunch_after_stop": lunch_after, "from_origin": origin_leg}


# ------------------------------------------------------------------ public API
def plan_itinerary(prefs, retriever, start_cluster=None):
    """Full pipeline: score -> select -> route. Returns a JSON-friendly dict."""
    pace = prefs.pace if prefs.pace in DAY_HOURS else "balanced"
    shortlist = select_places(prefs, retriever, pool_factor=POOL_FACTOR_ROUTING)
    warnings = []
    if not shortlist:
        return {"days": [], "unassigned": [], "warnings": ["No matching places found."], "summary": {}}

    origin = None
    if start_cluster:
        members = [p for p in retriever.places if p["cluster"] == start_cluster]
        if members:
            origin = centroid(members)
        else:
            warnings.append(f"Unknown start cluster '{start_cluster}', ignored.")
            start_cluster = None

    groups, leftover = build_days(shortlist, prefs.days, pace, start_cluster)
    groups = order_days(groups, origin)

    days, prev_last = [], None
    for n, group in enumerate(groups, 1):
        ordered_places = best_order([it["place"] for it in group], origin if n == 1 else None)
        by_id = {it["place"]["id"]: it for it in group}
        ordered_items = [by_id[p["id"]] for p in ordered_places]
        tl = build_timeline(ordered_items, pace, origin if n == 1 else None)

        transfer = None
        if prev_last is not None:
            h, km = leg(prev_last, ordered_places[0])
            transfer = {"hours": round(h, 2), "km": round(km)}
            if h > LONG_TRANSFER:
                warnings.append(f"Day {n}: about {h:.1f} h of travel from the previous day's last stop.")
        if tl["end_hour"] > LATE_FINISH:
            warnings.append(f"Day {n} runs until {tl['end_time']}: consider fewer stops.")
        prev_last = ordered_places[-1]

        base = Counter(s["cluster"] for s in tl["stops"]).most_common(1)[0][0]
        days.append({"day": n, "base_cluster": base, **tl, "transfer_from_previous_day": transfer})

    if len(days) < prefs.days:
        warnings.append(f"Only {len(days)} of {prefs.days} days could be filled with matching places.")

    all_stops = [s for d in days for s in d["stops"]]
    summary = {
        "total_stops": len(all_stops),
        "total_travel_hours": round(sum(d["travel_hours"] for d in days), 1),
        "entry_fees_per_person_inr": round(sum(s["entry_fee_inr"] for s in all_stops)),
        "places_with_unknown_fee": sum(1 for s in all_stops if not s["fee_known"]),
        "clusters": sorted({s["cluster"] for s in all_stops}),
    }
    return {"days": days, "unassigned": [it["place"]["name"] for it in leftover],
            "warnings": warnings, "summary": summary}


# ------------------------------------------------------------------ CLI demo
def print_itinerary(plan):
    for d in plan["days"]:
        print(f"\n=== Day {d['day']}  ({d['base_cluster']})  {d['start_time']} - {d['end_time']}  "
              f"driving {d['travel_hours']} h ===")
        if d["transfer_from_previous_day"]:
            t = d["transfer_from_previous_day"]
            print(f"   (transfer from previous day: {t['km']} km, {t['hours']} h)")
        if d["from_origin"]:
            print(f"   (from start city: {d['from_origin']['km']} km, {d['from_origin']['hours']} h)")
        for i, s in enumerate(d["stops"]):
            leg_txt = f"  <- {s['travel_from_prev_km']} km / {s['travel_from_prev_hours']} h" if i else ""
            print(f"  {s['arrive']}-{s['depart']}  {s['name'][:46]:<46} [{','.join(s['categories'])}]{leg_txt}")
            print(f"              why: {'; '.join(s['reasons'])}")
            if d["lunch_after_stop"] == i:
                print("  ~~ lunch break ~~")
    print("\nSummary:", plan["summary"])
    if plan["unassigned"]:
        print("Not scheduled:", ", ".join(plan["unassigned"][:8]))
    for w in plan["warnings"]:
        print("WARNING:", w)


if __name__ == "__main__":
    from data_loader import load_places
    from retrieval import PlaceRetriever

    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--interests", default="")
    ap.add_argument("--month", type=int)
    ap.add_argument("--pace", default="balanced", choices=["relaxed", "balanced", "packed"])
    ap.add_argument("--text", default="")
    ap.add_argument("--start", default=None, help="start cluster, e.g. 'Mysuru City'")
    args = ap.parse_args()

    places, _ = load_places(args.csv)
    retriever = PlaceRetriever(places)
    prefs = Preferences(days=args.days, interests=[i for i in args.interests.split(",") if i],
                        month=args.month, pace=args.pace, free_text=args.text)
    print_itinerary(plan_itinerary(prefs, retriever, start_cluster=args.start))