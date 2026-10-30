"""
routing.py - region first, then days, then stops, then the clock. No LLM, always works.

  1. REGION: pick the best compact group of areas (clusters) for this trip length, e.g. a 2-day beach trip
     stays on the coast instead of hopping to Mysuru. Scored by place quality and interest coverage, minus
     the drive between areas. Longer trips may use more areas, but never areas too far apart.
  2. DAYS: each day is built around ONE area (plus very close neighbours). Extra days go to the areas with
     the most good places left. Moving between areas happens only between days.
  3. STOPS: inside a day, the order is the best feasible one: shortest driving that also respects typical
     opening hours (hours.py), a lunch break, the day's time limit and the longest allowed single drive.
     A day that includes an early-opening place (safari, trek) may start earlier.
  4. CLOCK: arrival and departure times, waits, and warnings (long transfers, late finishes, places that
     may close before the visit ends).

Travel times are estimates (straight-line distance x road factor, slower on ghat roads), not live routing.
Opening windows are typical, not live. Stays are not modelled; each day exposes `base_cluster` as the hook.

CLI demo:
    python routing.py ../data/places_v1.csv --days 3 --interests heritage,food --month 12 --pace balanced
"""

import argparse
import itertools
import math
from collections import Counter
from dataclasses import replace

from hours import format_windows, typical_windows
from scoring import STOPS_PER_DAY, Preferences, select_places

# ------------------------------------------------------------------ travel model
ROAD_FACTOR_PLAIN, SPEED_PLAIN = 1.25, 45.0     # km road per km straight-line, km/h
ROAD_FACTOR_GHAT, SPEED_GHAT = 1.30, 38.0
HILLY = {"Coorg", "Chikkamagaluru"}
COASTAL = {"Mangaluru", "Udupi-Malpe"}

# ------------------------------------------------------------------ day limits
DAY_HOURS = {"relaxed": 7.5, "balanced": 9.0, "packed": 10.5}   # from the day's start to its end, lunch included
MAX_TRAVEL = {"relaxed": 3.0, "balanced": 4.0, "packed": 5.0}   # driving hours between stops in a day
MAX_LEG = {"relaxed": 1.5, "balanced": 2.0, "packed": 2.5}      # longest single drive inside a day
START_TIME = {"relaxed": 9.5, "balanced": 9.0, "packed": 8.0}
LUNCH_H, LUNCH_FROM = 1.0, 12.5
MAX_STOPS_HARD = 6
BRUTE_FORCE_MAX = 6      # try every order up to this many stops (720 orders), else nearest-neighbour

# ------------------------------------------------------------------ day building
ALPHA = 0.30                  # score points lost per extra hour of driving/waiting added to a day
CROSS_CLUSTER_PENALTY = 0.08  # a place from a neighbouring area must be a bit better than a local one
ADJACENT_H = 1.0              # areas whose centres are this close can share a day
POOL_FACTOR_ROUTING = 3.0     # shortlist size for the chosen region (name kept for debug_scores.py)
BROAD_POOL_FACTOR = 4.0       # wide pool used only to choose the region

# ------------------------------------------------------------------ opening hours
MAX_WAIT_H = 0.75             # willing to wait this long for a place to open
MAX_CUT_H = 0.5               # a visit may lose at most this much to closing time; more = not feasible
CUT_PENALTY = 2.0             # cost per hour of such a cut, so full visits are preferred
EARLY_FLOOR = 5.5             # earliest day start we will ever plan

# ------------------------------------------------------------------ region choice
TRANSFER_LAMBDA = 0.25        # score points lost per hour of driving between areas
CLUSTER_COST = 0.10           # small cost per extra area (more areas = more packing and unpacking)
INTEREST_BONUS = 0.30         # per requested interest the region can serve
SHORTFALL_COST = 0.03         # per missing useful place (kept small so filler cannot justify a detour)
MIN_USEFUL = 0.20             # filler never drives region choice: below this score, or (when interests are chosen) not matching one
SEED_TRIES = 3                # each day is grown from the top few seeds; the best day wins
DIAMETER = {1: 0.0, 2: 2.5, 3: 3.5, 4: 4.0, 5: 4.5, 6: 5.0, 7: 5.5}   # max drive between any two areas, by trip days

LATE_FINISH = 20.0
LONG_TRANSFER = 4.0


# ------------------------------------------------------------------ travel helpers
def haversine_km(a, b):
    r = 6371.0
    p1, p2 = math.radians(a["lat"]), math.radians(b["lat"])
    h = (math.sin((p2 - p1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(math.radians(b["lng"] - a["lng"]) / 2) ** 2)
    return 2 * r * math.asin(math.sqrt(h))


_LEG_CACHE = {}


def leg(a, b):
    """(hours, road_km) between two points carrying 'lat', 'lng', 'cluster'. Cached."""
    key = (a["lat"], a["lng"], a["cluster"], b["lat"], b["lng"], b["cluster"])
    hit = _LEG_CACHE.get(key)
    if hit is None:
        ghat = (a["cluster"] in HILLY or b["cluster"] in HILLY
                or ((a["cluster"] in COASTAL) != (b["cluster"] in COASTAL)))
        km = haversine_km(a, b) * (ROAD_FACTOR_GHAT if ghat else ROAD_FACTOR_PLAIN)
        hit = (km / (SPEED_GHAT if ghat else SPEED_PLAIN), km)
        _LEG_CACHE[key] = hit
    return hit


def fmt_time(h):
    m = int(round(h * 60))
    return f"{(m // 60) % 24:02d}:{m % 60:02d}"


def centroid(places):
    return {"lat": sum(p["lat"] for p in places) / len(places),
            "lng": sum(p["lng"] for p in places) / len(places),
            "cluster": Counter(p["cluster"] for p in places).most_common(1)[0][0]}


def stop_weight(place):
    """Quick stops (markets, food, chapels) count half toward the per-day stop limit."""
    return 0.5 if place["avg_visit_hours"] <= 1.0 else 1.0


def is_exclusive(place):
    """Safaris and treks: a long visit that must start early, so at most one fits in a day."""
    return place["avg_visit_hours"] >= 3.0 and typical_windows(place)[0][0] <= 7.0


# ------------------------------------------------------------------ the clock for one ordered day
def simulate(order, start, origin=None):
    """
    Walk through `order` from `start`. Returns None if some stop cannot fit an opening window
    (wait <= MAX_WAIT_H, at least min(visit, 1 h) before closing, and at most MAX_CUT_H lost to closing),
    else a dict of timings.
    """
    t = start
    stops, lunch_done, lunch_after = [], False, None
    travel_in = travel_origin = wait_total = cut_total = longest = 0.0
    if origin:
        h, _ = leg(origin, order[0])
        t += h
        travel_origin = h
    for i, p in enumerate(order):
        h = km = 0.0
        if i > 0:
            h, km = leg(order[i - 1], p)
            if not lunch_done and (t >= LUNCH_FROM or (t >= 11.5 and t + h > 13.5)):
                t += LUNCH_H
                lunch_done, lunch_after = True, i - 1
            t += h
            travel_in += h
            longest = max(longest, h)
        arrive, visit, begin, close = t, p["avg_visit_hours"], None, None
        for o, c in typical_windows(p):
            b = max(arrive, o)
            if b - arrive <= MAX_WAIT_H and b + min(visit, 1.0) <= c and b + visit - c <= MAX_CUT_H:
                begin, close = b, c
                break
        if begin is None:
            return None
        end_visit = begin + visit
        cut = max(0.0, end_visit - close)
        depart = end_visit - cut
        wait_total += begin - arrive
        cut_total += cut
        stops.append({"place": p, "arrive": arrive, "begin": begin, "depart": depart, "wait": begin - arrive,
                      "cut": cut, "leg_h": h, "leg_km": km})
        t = depart
    return {"stops": stops, "end": t, "travel_in": travel_in, "travel_origin": travel_origin,
            "wait": wait_total, "cut": cut_total, "longest": longest, "lunch_after": lunch_after}


def _valid(sim, start, pace, enforce_day_cap=True):
    if enforce_day_cap and sim["end"] > start + DAY_HOURS[pace]:
        return False
    return sim["travel_in"] <= MAX_TRAVEL[pace] and sim["longest"] <= MAX_LEG[pace]


def _cost(sim):
    return sim["travel_in"] + sim["travel_origin"] + sim["wait"] + CUT_PENALTY * sim["cut"]


def _nearest_neighbour(places):
    rest = sorted(places, key=lambda p: typical_windows(p)[0][0])
    order = [rest.pop(0)]
    while rest:
        nxt = min(rest, key=lambda p: leg(order[-1], p)[0])
        rest.remove(nxt)
        order.append(nxt)
    return order


def _best_for_start(places, start, pace, origin, enforce_day_cap=True):
    orders = (itertools.permutations(places) if len(places) <= BRUTE_FORCE_MAX
              else [_nearest_neighbour(places)])
    best = None
    for order in orders:
        sim = simulate(order, start, origin)
        if sim is None or not _valid(sim, start, pace, enforce_day_cap):
            continue
        c = _cost(sim)
        if best is None or c < best[0]:
            best = (c, list(order), sim)
    return best


def evaluate(places, pace, origin=None):
    """
    Best feasible order for these places, or None.
    An earlier start is tried ONLY when the day contains a safari or trek (they must start early),
    never just to squeeze in more stops when the day is simply full.
    """
    start = START_TIME[pace]
    res = _best_for_start(places, start, pace, origin)
    if res is None and any(is_exclusive(p) for p in places):
        early = max(EARLY_FLOOR, min(typical_windows(p)[0][0] for p in places))
        if early < start:
            start = early
            res = _best_for_start(places, start, pace, origin)
    if res is None:
        return None
    cost, order, sim = res
    return {"order": order, "sim": sim, "start": start, "cost": cost}


# ------------------------------------------------------------------ step 1: choose the region
def _best_chain(subset, hubs, origin):
    """Visiting order of areas with the least driving (from the origin if there is one)."""
    best = None
    for perm in itertools.permutations(subset):
        hours = leg(origin, hubs[perm[0]])[0] if origin else 0.0
        hours += sum(leg(hubs[a], hubs[b])[0] for a, b in zip(perm, perm[1:]))
        if best is None or hours < best[1]:
            best = (list(perm), hours)
    return best


def choose_region(items, prefs, pace, hubs, origin=None, widen=0):
    """Best compact set of areas for this trip. Returns the areas in visiting order."""
    by_cluster = {}
    for it in items:
        by_cluster.setdefault(it["place"]["cluster"], []).append(it)
    names = sorted(c for c in by_cluster if c in hubs)
    days = prefs.days
    need = days * STOPS_PER_DAY[pace]
    max_k = min(len(names), max(1, math.ceil(days * 0.7)) + widen)
    limit = DIAMETER.get(days, 5.5) + 1.5 * widen
    wanted = set(prefs.interests)

    best = None
    for k in range(1, max_k + 1):
        for subset in itertools.combinations(names, k):
            if any(leg(hubs[a], hubs[b])[0] > limit for a, b in itertools.combinations(subset, 2)):
                continue
            ranked = sorted((it for c in subset for it in by_cluster[c]
                             if it["score"] >= MIN_USEFUL and (not wanted or wanted & set(it["place"]["categories"]))),
                            key=lambda it: -it["score"])
            pool, exclusive = [], 0
            for it in ranked:
                if is_exclusive(it["place"]):
                    if exclusive >= days:
                        continue
                    exclusive += 1
                pool.append(it)
                if len(pool) >= need:
                    break
            value = sum(it["score"] for it in pool) - SHORTFALL_COST * max(0, need - len(pool))
            if wanted:
                covered = {cat for it in pool for cat in it["place"]["categories"]} & wanted
                value += INTEREST_BONUS * len(covered)
            chain, hours = _best_chain(subset, hubs, origin)
            value -= TRANSFER_LAMBDA * hours + CLUSTER_COST * (k - 1)
            if best is None or value > best[0]:
                best = (value, chain)
    return best[1] if best else names[:1]


def allocate_days(chain, shortlist, days, pace, hubs, wanted=frozenset()):
    """
    Give each trip day to the area whose next day (really built) is worth most.
      Phase 1: count only places that match the traveller's interests, so a wildlife trip spends its days
               where the wildlife is, and areas that would only add filler get no day.
      Phase 2: if days are still unassigned, fill them with the best available content of any kind, so a
               5-day request still gets 5 days.
    Opening a new area costs its drive from the areas already chosen.
    """
    sims = {}

    def day_value(c, k):
        """(interest-matching value, total value) of the k-th day built around area c."""
        st = sims.setdefault(c, {"pool": list(shortlist), "vals": []})
        while len(st["vals"]) <= k:
            res = build_day(st["pool"], c, pace, hubs)
            if res is None:
                st["vals"].append(None)          # nothing left that fits a day here
                continue
            items, ev = res
            for it in items:
                st["pool"].remove(it)
            total = sum(it["score"] for it in items) - ALPHA * ev["cost"]
            good = sum(it["score"] for it in items
                       if not wanted or wanted & set(it["place"]["categories"])) - ALPHA * ev["cost"]
            st["vals"].append((good, total))
        return st["vals"][k]

    def pick(chosen, which):
        """Phase 0 needs a positive interest-matching value; phase 1 takes any day that can still be built."""
        best_c, best_v = None, (0.0 if which == 0 else float("-inf"))
        for c in chain:
            k = chosen.get(c, 0)
            dv = day_value(c, k)
            if dv is None:
                continue
            v = dv[which]
            if k == 0 and chosen:
                v -= TRANSFER_LAMBDA * min(leg(hubs[c], hubs[o])[0] for o in chosen)
            if v > best_v:
                best_c, best_v = c, v
        return best_c

    chosen = {}
    for phase in (0, 1):
        while sum(chosen.values()) < days:
            c = pick(chosen, phase)
            if c is None:
                break
            chosen[c] = chosen.get(c, 0) + 1
    if not chosen:
        chosen = {chain[0]: 1}
    return {c: chosen[c] for c in chain if c in chosen}


# ------------------------------------------------------------------ step 2 and 3: build a day
def _grow(seed, seed_ev, candidates, cluster, pace):
    """Greedily add places to a day that starts from `seed`."""
    day, cur, max_weight = [seed], seed_ev, STOPS_PER_DAY[pace]
    while len(day) < MAX_STOPS_HARD:
        weight = sum(stop_weight(it["place"]) for it in day)
        best = None
        for cand in candidates:
            if cand in day or weight + stop_weight(cand["place"]) > max_weight:
                continue
            ev = evaluate([it["place"] for it in day] + [cand["place"]], pace)
            if ev is None:
                continue
            gain = (cand["score"] - ALPHA * (ev["cost"] - cur["cost"])
                    - (CROSS_CLUSTER_PENALTY if cand["place"]["cluster"] != cluster else 0.0))
            if best is None or gain > best[0]:
                best = (gain, cand, ev)
        if best is None or best[0] <= 0:
            break
        day.append(best[1])
        cur = best[2]
    return day, cur


def build_day(remaining, cluster, pace, hubs):
    """Day around one area, grown from the top few seeds; the most valuable result wins. -> (items, ev) or None."""
    def near(it):
        c = it["place"]["cluster"]
        return c == cluster or (c in hubs and leg(hubs[c], hubs[cluster])[0] <= ADJACENT_H)

    candidates = [it for it in remaining if near(it)]
    own = sorted((it for it in candidates if it["place"]["cluster"] == cluster), key=lambda it: -it["score"])
    best = None
    tried = 0
    for seed in own:
        ev = evaluate([seed["place"]], pace)
        if ev is None:
            continue
        day, cur = _grow(seed, ev, candidates, cluster, pace)
        value = sum(it["score"] for it in day) - ALPHA * cur["cost"]
        if best is None or value > best[0]:
            best = (value, day, cur)
        tried += 1
        if tried >= SEED_TRIES:
            break
    return (best[1], best[2]) if best else None


# ------------------------------------------------------------------ step 4: the clock
def build_timeline(ordered_items, ev, origin_leg=None):
    sim = ev["sim"]
    stops = []
    for it, s in zip(ordered_items, sim["stops"]):
        p = it["place"]
        stops.append({
            "id": p["id"], "name": p["name"], "cluster": p["cluster"], "town": p["town"],
            "lat": p["lat"], "lng": p["lng"], "categories": p["categories"],
            "arrive": fmt_time(s["begin"]), "depart": fmt_time(s["depart"]),
            "visit_hours": p["avg_visit_hours"],
            "travel_from_prev_hours": round(s["leg_h"], 2), "travel_from_prev_km": round(s["leg_km"]),
            "pace_intensity": p["pace_intensity"],
            "entry_fee_inr": p["entry_fee_inr"], "fee_known": p["fee_known"],
            "typical_hours": format_windows(typical_windows(p)),
            "wait_minutes": round(s["wait"] * 60), "cut_minutes": round(s["cut"] * 60),
            "score": round(it["score"], 3), "reasons": it["reasons"],
        })
    return {"stops": stops, "end_hour": sim["end"], "end_time": fmt_time(sim["end"]),
            "start_time": fmt_time(ev["start"]),
            "travel_hours": round(sim["travel_in"] + sim["travel_origin"], 2),
            "lunch_after_stop": sim["lunch_after"], "from_origin": origin_leg}


# ------------------------------------------------------------------ public API
def plan_itinerary(prefs, retriever, start_cluster=None):
    """Full pipeline: score -> choose region -> allocate days -> build days -> clock. JSON-friendly dict."""
    pace = prefs.pace if prefs.pace in DAY_HOURS else "balanced"
    warnings = []

    broad = select_places(prefs, retriever, pool_factor=BROAD_POOL_FACTOR)
    if not broad:
        return {"days": [], "unassigned": [], "warnings": ["No matching places found."], "summary": {}}

    by_cluster = {}
    for p in retriever.places:
        by_cluster.setdefault(p["cluster"], []).append(p)
    hubs = {c: centroid(ps) for c, ps in by_cluster.items()}

    origin = None
    if start_cluster:
        if start_cluster in hubs:
            origin = hubs[start_cluster]
        else:
            warnings.append(f"Unknown start cluster '{start_cluster}', ignored.")
            start_cluster = None

    def attempt(widen):
        chain = choose_region(broad, prefs, pace, hubs, origin, widen)
        outside = [p["id"] for p in retriever.places if p["cluster"] not in chain]
        shortlist = select_places(replace(prefs, exclude_ids=list(prefs.exclude_ids) + outside), retriever,
                                  pool_factor=POOL_FACTOR_ROUTING) or broad
        alloc = allocate_days(chain, shortlist, prefs.days, pace, hubs, frozenset(prefs.interests))
        remaining, built = list(shortlist), []
        for c in chain:
            for _ in range(alloc.get(c, 0)):
                res = build_day(remaining, c, pace, hubs)
                if res is None:
                    break
                items, ev = res
                for it in items:
                    remaining.remove(it)
                built.append((c, items, ev))
        return chain, built, remaining

    # Start with the most compact region. If it cannot fill every day, widen it one step and keep
    # the attempt that fills the most days (the first one wins ties, so compact is preferred).
    best = None
    for widen in (0, 1):
        result = attempt(widen)
        if best is None or len(result[1]) > len(best[1][1]):
            best = (widen, result)
        if len(result[1]) >= prefs.days:
            break
    widened, (chain, built, remaining) = best
    if widened and built:
        warnings.append("The closest areas did not have enough matching places for every day, "
                        "so the route stretches a little further. Expect longer drives between some days.")

    if not built:
        return {"days": [], "unassigned": [it["place"]["name"] for it in remaining[:9]],
                "warnings": warnings + ["Could not fit any place into a day. Try a more relaxed pace or more days."],
                "summary": {}, "region": {"clusters": chain, "note": ""}}

    days, prev_last = [], None
    for n, (c, items, ev) in enumerate(built, 1):
        places = [it["place"] for it in items]
        origin_leg = None
        if n == 1 and origin:
            h, km = leg(origin, ev["order"][0])
            ev_o = evaluate(places, pace, origin)
            if ev_o:
                ev = ev_o
                h, km = leg(origin, ev["order"][0])
            else:
                warnings.append(f"Day 1 may start later than shown because of the drive from {start_cluster}.")
            origin_leg = {"hours": round(h, 2), "km": round(km)}
            if h > 3.0:
                warnings.append(f"Day 1 begins with about {h:.1f} h of driving from {start_cluster}.")
        by_id = {it["place"]["id"]: it for it in items}
        ordered_items = [by_id[p["id"]] for p in ev["order"]]
        tl = build_timeline(ordered_items, ev, origin_leg)

        transfer = None
        if prev_last is not None:
            h, km = leg(prev_last, ev["order"][0])
            transfer = {"hours": round(h, 2), "km": round(km)}
            if h > LONG_TRANSFER:
                warnings.append(f"Day {n}: about {h:.1f} h of travel from the previous day's last stop.")
        if tl["end_hour"] > LATE_FINISH:
            warnings.append(f"Day {n} runs until {tl['end_time']}: consider fewer stops.")
        for s in tl["stops"]:
            if s["cut_minutes"] >= 15:
                warnings.append(f"Day {n}: {s['name'].split(' (')[0]} may close before you finish "
                                f"(typical hours {s['typical_hours']}). Check opening hours.")
        prev_last = ev["order"][-1]
        days.append({"day": n, "base_cluster": c, **tl, "transfer_from_previous_day": transfer})

    if len(days) < prefs.days:
        warnings.append(f"Only {len(days)} of {prefs.days} days could be filled with matching places "
                        f"in {' and '.join(chain)}.")

    all_stops = [s for d in days for s in d["stops"]]
    used = [d["base_cluster"] for d in days]
    areas = list(dict.fromkeys(used))
    note = (f"Based around {areas[0]}, so more time at each place and less on the road." if len(areas) == 1
            else f"Focused on {' and '.join(areas)} so the drives between places stay short.")
    summary = {
        "total_stops": len(all_stops),
        "total_travel_hours": round(sum(d["travel_hours"] for d in days), 1),
        "entry_fees_per_person_inr": round(sum(s["entry_fee_inr"] for s in all_stops)),
        "places_with_unknown_fee": sum(1 for s in all_stops if not s["fee_known"]),
        "clusters": sorted({s["cluster"] for s in all_stops}),
    }
    return {"days": days, "unassigned": [it["place"]["name"] for it in remaining[:9]],
            "warnings": warnings, "summary": summary, "region": {"clusters": areas, "note": note}}


# ------------------------------------------------------------------ CLI demo
def print_itinerary(plan):
    if plan.get("region"):
        print("Region:", plan["region"]["note"])
    for d in plan["days"]:
        print(f"\n=== Day {d['day']}  ({d['base_cluster']})  {d['start_time']} - {d['end_time']}  "
              f"driving {d['travel_hours']} h ===")
        if d["transfer_from_previous_day"]:
            t = d["transfer_from_previous_day"]
            print(f"   (transfer from previous day: {t['km']} km, {t['hours']} h)")
        if d["from_origin"]:
            print(f"   (from start area: {d['from_origin']['km']} km, {d['from_origin']['hours']} h)")
        for i, s in enumerate(d["stops"]):
            leg_txt = f"  <- {s['travel_from_prev_km']} km / {s['travel_from_prev_hours']} h" if i else ""
            extra = f"  [waits {s['wait_minutes']} min]" if s["wait_minutes"] else ""
            extra += f"  [cut short {s['cut_minutes']} min]" if s["cut_minutes"] else ""
            print(f"  {s['arrive']}-{s['depart']}  {s['name'][:44]:<44} typical {s['typical_hours']}{leg_txt}{extra}")
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