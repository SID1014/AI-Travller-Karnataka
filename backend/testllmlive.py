"""
test_llm_live.py - run the REAL LLM end to end and show exactly what it did.

Needs from earlier: llm.py, stays.py, service.py (plus your data files and a filled-in .env).
Run from the backend folder:
    python test_llm_live.py
    python test_llm_live.py --days 4 --text "wildlife and waterfalls in July, packed"

What to look for in the output:
  parsed_by  = llm        -> your free text was understood by the model (not the keyword fallback)
  narrative  = llm        -> every sentence passed the grounding checks
             = mixed      -> some sentences were replaced by template text (see the note)
             = template   -> the model failed or was off; check the [llm] lines printed above
  tokens/plan             -> multiply by plans per day to see your free-tier budget
"""

import argparse
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
load_dotenv(Path(__file__).parent / ".env")

from service import PlannerService  # noqa: E402


def show(res, label):
    u, n = res["understood"], res["narrative"]
    print(f"\n=== {label}")
    print(f"parsed_by : {u['parsed_by']}   days={u['days']} interests={u['interests']} month={u['month']} "
          f"pace={u['pace']} start={u['start_cluster']} travelers={u['travelers']} budget={u['budget_inr']}")
    print(f"narrative : {n['source']}   (took {res['meta']['seconds']} s)")
    if n.get("note"):
        print("note      :", n["note"])
    print("title     :", n["title"])
    print("intro     :", n["intro"])
    d1 = res["plan"]["days"][0] if res["plan"]["days"] else None
    if d1:
        blurbs = {s["id"]: s["blurb"] for s in n["days"][0]["stops"]}
        print("day 1     :", n["days"][0]["summary"])
        for s in d1["stops"][:3]:
            print(f"   {s['name'][:40]:<40} {blurbs.get(s['id'], '')}")
    if n.get("tips"):
        print("tips      :", " | ".join(n["tips"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--text", default="3 days, temples and good local food, relaxed pace, in December, starting from Mysore")
    args = ap.parse_args()

    svc = PlannerService(os.getenv("DATA_PATH", str(ROOT / "data" / "places_v1.csv")),
                         os.getenv("STAYS_PATH", str(ROOT / "data" / "city_stay_costs.csv")))
    llm = svc.llm
    print(f"\nLLM: {llm.label}   extra_tokens={llm.extra_tokens}   reasoning_effort={llm.reasoning_effort or 'default'}")
    if not llm.available:
        sys.exit("LLM is not configured. Fill LLM_PROVIDER, LLM_API_KEY and LLM_MODEL in .env first.")

    tests = [
        ("1. free text + chips", {"days": args.days, "interests": ["heritage"], "month": 12, "free_text": args.text,
                                  "stay_type": "mid", "travelers": 2, "budget_inr": 25000}),
        ("2. free text only", {"free_text": "weekend with 4 friends, beaches and seafood, under 15k"}),
        ("3. form only (no text to parse)", {"days": 2, "interests": ["wildlife", "nature"], "month": 7, "pace": "relaxed"}),
    ]
    per_plan = []
    for label, req in tests:
        before = llm.total_tokens
        t = time.time()
        res = svc.plan(req)
        used = llm.total_tokens - before
        per_plan.append(used)
        show(res, label)
        print(f"tokens    : {used} for this plan   ({round(time.time() - t, 1)} s total)")

    avg = sum(per_plan) / len(per_plan)
    print(f"\n--- summary: {llm.calls} LLM calls, {llm.total_tokens} tokens, {llm.truncated} truncated replies")
    print(f"average {avg:.0f} tokens per plan -> about {int(200000 / avg) if avg else 0} plans per day if your daily "
          f"limit is 200,000 tokens (check your Groq console Limits page)")
    if llm.truncated:
        print("Some replies were cut off: raise LLM_EXTRA_TOKENS in .env (for example 1500) and run again.")


if __name__ == "__main__":
    main()