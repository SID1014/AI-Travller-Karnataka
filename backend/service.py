"""
service.py - the whole planner as one plain-Python object (no web framework), so it is easy to test.

    svc = PlannerService(data_path, stays_path)
    result = svc.plan({"days": 3, "interests": ["heritage"], "month": 12, "free_text": "temples and coffee"})

Flow:  clean request -> understand free text (LLM, else rules) -> merge with form
       -> retrieve + score -> route into days -> narrative (LLM, else template)
       -> price alerts + rough budget -> one JSON-friendly dict.
"""

import copy
import json
import math
import os
import threading
import time
from collections import OrderedDict

from data_loader import VALID_CATEGORIES, load_places
from llm import PACES, LLMClient, ai_status_note, build_narrative, parse_request
from retrieval import PlaceRetriever
from routing import plan_itinerary
from scoring import MONTH_NAMES, Preferences
from stays import TIERS, estimate_stays, load_stays, price_alerts


def _int(v, lo, hi, default=None):
    try:
        return max(lo, min(hi, int(v)))
    except (TypeError, ValueError):
        return default


class PlannerService:
    def __init__(self, data_path, stays_path):
        self.places, issues = load_places(data_path)
        if not self.places:
            raise RuntimeError(f"No valid places loaded from {data_path}")
        skipped = sum(1 for s, _, _ in issues if s == "ERROR")
        print(f"[service] {len(self.places)} places loaded, {skipped} skipped")
        self.retriever = PlaceRetriever(self.places, backend=os.getenv("RETRIEVAL_BACKEND", "auto"))
        self.clusters = sorted({p["cluster"] for p in self.places})
        self.stays = load_stays(stays_path)
        self.llm = LLMClient()
        self._cache, self._cache_lock = OrderedDict(), threading.Lock()
        self.cache_ttl = float(os.getenv("PLAN_CACHE_TTL", "1800") or 0)     # seconds; 0 turns the cache off
        self.cache_max = int(os.getenv("PLAN_CACHE_MAX", "100") or 100)
        self.all_names = [p["name"] for p in self.places]
        self.by_id = {p["id"]: p for p in self.places}
        print(f"[service] LLM: {self.llm.label}")

    # ------------------------------------------------------------ metadata for the UI
    def meta(self):
        return {
            "categories": sorted(VALID_CATEGORIES),
            "clusters": self.clusters,
            "months": MONTH_NAMES,
            "paces": PACES,
            "stay_types": list(TIERS),
            "places": len(self.places),
            "llm": self.llm.label,
            "retrieval": self.retriever.backend,
        }

    # ------------------------------------------------------------ request handling
    def _clean(self, req):
        req = req or {}
        interests = [i for i in (str(x).lower() for x in req.get("interests") or []) if i in VALID_CATEGORIES]
        pace = str(req.get("pace") or "").lower()
        start = req.get("start_cluster")
        stay = str(req.get("stay_type") or "").lower()
        return {
            "days": _int(req.get("days"), 1, 7),
            "interests": interests,
            "month": _int(req.get("month"), 1, 12),
            "pace": pace if pace in PACES else None,
            "start_cluster": start if start in self.clusters else None,
            "travelers": _int(req.get("travelers"), 1, 12),
            "budget_inr": _int(req.get("budget_inr"), 1000, 10_000_000),
            "stay_type": stay if stay in TIERS else None,
            "free_text": str(req.get("free_text") or "").strip()[:500],
            "use_llm": bool(req.get("use_llm", True)),
        }

    def _merge(self, form, parsed):
        """Text is more specific than form defaults, so values found in the message win."""
        from_text = []

        def pick(key, default=None):
            if parsed.get(key) not in (None, [], ""):
                from_text.append(key)
                return parsed[key]
            return form.get(key) if form.get(key) not in (None, [], "") else default

        interests = list(dict.fromkeys(form["interests"] + parsed.get("interests", [])))
        if parsed.get("interests"):
            from_text.append("interests")
        eff = {
            "days": pick("days", 3),
            "interests": interests,
            "month": pick("month"),
            "pace": pick("pace", "balanced"),
            "start_cluster": pick("start_cluster"),
            "travelers": pick("travelers", 2),
            "budget_inr": pick("budget_inr"),
            "stay_type": form["stay_type"],
        }
        eff["from_message"] = sorted(set(from_text))
        return eff

    # ------------------------------------------------------------ result cache
    @staticmethod
    def _cache_key(form):
        norm = {**form, "free_text": " ".join(form["free_text"].lower().split()),
                "interests": sorted(form["interests"])}
        return json.dumps(norm, sort_keys=True)

    def _cache_get(self, key):
        if not self.cache_ttl:
            return None
        with self._cache_lock:
            hit = self._cache.get(key)
            if not hit:
                return None
            if time.time() - hit[0] > self.cache_ttl:
                del self._cache[key]
                return None
            self._cache.move_to_end(key)
            return copy.deepcopy(hit[1])

    def _cache_put(self, key, result):
        if not self.cache_ttl:
            return
        with self._cache_lock:
            self._cache[key] = (time.time(), copy.deepcopy(result))
            self._cache.move_to_end(key)
            while len(self._cache) > self.cache_max:
                self._cache.popitem(last=False)

    # ------------------------------------------------------------ main entry
    def plan(self, req):
        t0 = time.time()
        form = self._clean(req)
        client = self.llm if form["use_llm"] else None

        key = self._cache_key(form)
        hit = self._cache_get(key)
        if hit is not None:
            hit["meta"] = {**hit["meta"], "cached": True, "seconds": round(time.time() - t0, 3)}
            return hit
        if client:
            client.reset_status()

        parsed, parsed_by = parse_request(form["free_text"], self.clusters, client)
        eff = self._merge(form, parsed)

        prefs = Preferences(days=eff["days"], interests=eff["interests"], month=eff["month"],
                            pace=eff["pace"], free_text=form["free_text"])
        plan = plan_itinerary(prefs, self.retriever, start_cluster=eff["start_cluster"])

        facts = {}
        for d in plan["days"]:
            for s in d["stops"]:
                p = self.by_id.get(s["id"], {})
                facts[s["id"]] = p.get("wiki_extract") or p.get("voyage_extract") or p.get("tags") or ""
                s["maps_url"] = f"https://www.google.com/maps/search/?api=1&query={s['lat']},{s['lng']}"
                s["source_url"] = p.get("source_url", "")

        unmet = self._coverage_notes(plan, eff)
        if unmet:
            plan["warnings"] = unmet + plan.get("warnings", [])
        note = ""
        if eff["month"]:
            note = f"Planned for {MONTH_NAMES[eff['month'] - 1]}. "
        note += " ".join(unmet)
        trip = {"pace": eff["pace"], "month_name": MONTH_NAMES[eff["month"] - 1] if eff["month"] else None,
                "travelers": eff["travelers"], "interests": eff["interests"], "unmet": unmet}
        narrative = build_narrative(plan, facts, self.all_names, client, note.strip(), trip)

        llm_on = bool(client and client.available)
        status = client.status if llm_on else None
        notes = []
        if status and narrative["source"] == "template":
            notes.append(ai_status_note(status))
        elif status and parsed_by == "rules" and form["free_text"]:
            notes.append("Your message was read by keyword matching because the AI " +
                         ("was busy." if status == "busy" else "had a problem."))
        if notes:
            narrative["note"] = (narrative.get("note", "") + " " + " ".join(notes)).strip()
        narrative["ai_status"] = status or ("ok" if llm_on else "off")

        clusters = plan.get("summary", {}).get("clusters", [])
        alerts = price_alerts(self.stays, clusters, eff["month"])
        budget = self._budget(plan, eff)

        result = {
            "understood": {**{k: v for k, v in eff.items()}, "parsed_by": parsed_by,
                           "month_name": MONTH_NAMES[eff["month"] - 1] if eff["month"] else None},
            "plan": plan,
            "coverage_notes": unmet,
            "narrative": narrative,
            "alerts": alerts,
            "budget": budget,
            "meta": {"llm": self.llm.label if client else "off", "retrieval": self.retriever.backend,
                     "ai_status": narrative["ai_status"], "cached": False,
                     "seconds": round(time.time() - t0, 2)},
        }
        # Cache only good results: never remember a template fallback caused by a busy or failing AI
        if (not llm_on) or (status is None and narrative["source"] != "template"):
            self._cache_put(key, result)
        return result

    # ------------------------------------------------------------ honest notes about unmet interests
    def _coverage_notes(self, plan, eff):
        covered = {c for d in plan["days"] for s in d["stops"] for c in s["categories"]}
        notes = []
        for interest in eff["interests"]:
            label = interest.replace("-", " ")
            if interest in covered:
                # covered, but only by places that are outside their best season?
                stops = [s for d in plan["days"] for s in d["stops"] if interest in s["categories"]]
                if eff["month"] and stops and all(any("off-season" in r for r in s["reasons"]) for s in stops):
                    names = ", ".join(s["name"].split(" (")[0] for s in stops[:2])
                    notes.append(f"The {label} places in this plan ({names}) are outside their best season in "
                                 f"{MONTH_NAMES[eff['month'] - 1]}, so expect a weaker experience.")
                continue
            pool = [p for p in self.places if interest in p["categories"]]
            if not pool:
                notes.append(f"We do not have any {label} places in the dataset yet.")
            elif eff["month"] and not any(eff["month"] in p["best_months"] for p in pool):
                notes.append(f"No {label} places are in season in {MONTH_NAMES[eff['month'] - 1]}, so none were included.")
            else:
                notes.append(f"No {label} places fitted near the rest of your route, so none were included.")
        return notes

    # ------------------------------------------------------------ budget
    def _budget(self, plan, eff):
        travelers = eff["travelers"]
        stops = [s for d in plan["days"] for s in d["stops"]]
        per_person = sum(s["entry_fee_inr"] for s in stops)
        unknown = sum(1 for s in stops if not s["fee_known"])
        entry = per_person * travelers
        stay = estimate_stays(self.stays, [d["base_cluster"] for d in plan["days"]], eff["stay_type"], travelers)
        known_total = round(entry + (stay["total_inr"] if stay else 0))

        msg = ""
        if eff["budget_inr"]:
            if known_total > eff["budget_inr"]:
                msg = (f"Entry fees and stays alone come to about ₹{known_total:,}, above your budget of "
                       f"₹{eff['budget_inr']:,}. Food and transport are extra.")
            else:
                msg = (f"Entry fees and stays come to about ₹{known_total:,} of your ₹{eff['budget_inr']:,} budget. "
                       "Food and transport still need to fit in the rest.")
        return {
            "travelers": travelers,
            "entry_fees_per_person_inr": round(per_person),
            "entry_fees_total_inr": round(entry),
            "places_with_unknown_fee": unknown,
            "stay": stay,
            "known_total_inr": round(known_total),
            "budget_inr": eff["budget_inr"],
            "over_budget": (known_total > eff["budget_inr"]) if eff["budget_inr"] else None,
            "message": msg,
            "not_included": "Food, transport, and paid activities such as safaris are not included. "
                            "Budget is checked against the plan but does not yet change which places are chosen.",
        }