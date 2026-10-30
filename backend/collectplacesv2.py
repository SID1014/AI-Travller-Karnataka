#!/usr/bin/env python3
"""
collect_places_v2.py - rebuild the places dataset with trustworthy coordinates and more fields.

WHY A REWRITE: v1 took coordinates from whichever Wikipedia page ranked first, so look-alike
names ("Chennakesava Temple, Somanathapura" vs "Chennakeshava Temple, Belur") got the wrong
place, and nothing checked the result. v2 never trusts a single source.

HOW COORDINATES ARE CHOSEN (per place)
  1. Gather candidates from independent sources:
       wikivoyage (listing templates on region pages), wikidata (P625), wikipedia, nominatim (OSM)
       + your own manual_coords.csv (always wins) + coordinates already in the input CSV
  2. Sanity filters: inside Karnataka AND within a radius of the place's town
     (the town is geocoded once and used as an "anchor"; Belur is ~130 km from Somanathapura,
     so that candidate is rejected). Wikipedia pages are also rejected if the title doesn't
     match the place or names a different town from your list.
  3. Consensus: candidates within 2 km of each other agree. Two independent sources agreeing
     = confidence "high"; one trusted source = "medium"; weak/lone OSM guess = "low";
     nothing survives = "none" (fill it in manual_coords.csv).

EXTRA FIELDS (only filled when empty, never overwrites your manual edits)
  wikidata_id, official_website, wiki_title/url/extract, voyage_description,
  fee_text + entry_fee_inr (a best-effort guess from Wikivoyage, marked low confidence),
  hours_text (reference only; the planner fetches live hours), anchor_km, coord_* audit columns.

ALSO WRITES  wikivoyage_sleep_prices.csv : every priced "Sleep" listing on the region pages,
labelled Budget / Mid-range / Splurge, as a cross-check for city_stay_costs.csv.

USAGE
    pip install requests
    python collect_places_v2.py --in ../data/karnataka_mysuru_division_places.csv --out ../data/places_v2.csv
    python collect_places_v2.py --limit 5            # quick test
    python collect_places_v2.py --only mys_palace,crg_abbey
    python collect_places_v2.py --refresh-coords     # ignore coordinates already in the input
Every web response is cached in .collect_cache.json, so re-runs are fast and polite.
"""

import argparse
import csv
import json
import math
import re
import statistics
import sys
import time
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import requests

# ----------------------------------------------------------------- config
USER_AGENT = "KarnatakaTravelPlanner/2.0 (hackathon project; contact: siddhantdakhore143@gmail.com)"
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
WIKIVOYAGE_API = "https://en.wikivoyage.org/w/api.php"
WIKIDATA_API = "https://www.wikidata.org/w/api.php"
NOMINATIM_API = "https://nominatim.openstreetmap.org/search"

LAT_MIN, LAT_MAX, LNG_MIN, LNG_MAX = 11.5, 18.5, 74.0, 78.6   # Karnataka bounding box
DEFAULT_ANCHOR_KM = 30.0         # a place must be this close to its town...
WIDE_ANCHOR_KM = 70.0            # ...unless it is a park / peak / waterfall / reservoir etc.
AGREE_KM = 2.0                   # two candidates this close count as agreeing
CACHE_FILE = Path(".collect_cache.json")
WIKI_DELAY, NOMINATIM_DELAY = 0.25, 1.1
MAX_EXTRACT_CHARS = 1800

WIDE_CATEGORIES = {"wildlife", "adventure", "hill-station"}
WIDE_KEYWORDS = re.compile(r"national park|sanctuary|reserve|hills?\b|peak|falls|reservoir|backwaters|trek|dam\b|ghat", re.I)

# Wikivoyage region pages to read listings from (pages that don't exist are skipped silently)
REGION_PAGES = [
    "Mysore", "Mysuru", "Srirangapatna", "Kodagu", "Coorg", "Madikeri", "Chikmagalur",
    "Chikkamagaluru", "Hassan", "Belur", "Halebidu", "Shravanabelagola", "Mangalore",
    "Mangaluru", "Udupi", "Manipal", "Dharmasthala", "Bandipur National Park",
    "Nagarhole National Park", "Sringeri", "Kollur", "Sakleshpur",
]
LISTING_TYPES = {"see", "do", "eat", "drink", "buy", "sleep", "listing", "go"}

PRIORITY = {"manual": 0, "wikivoyage": 1, "existing": 1.5, "wikidata": 2, "wikipedia": 3, "nominatim": 4}

# Words that don't identify a place (used to find its distinctive name tokens)
GENERIC = set("""temple temples beach falls fall national park wildlife sanctuary fort palace hills hill peak museum
market church cathedral chapel monastery gardens garden dam lake the of and at in near viewpoint view point island
reservoir backwaters tour plantation elephant camp zoo zoological hotel sweet mart matha peetha trek ghat tiger
reserve bird sri shri swamy swami devi maha lighthouse and station""".split())

NEW_COLUMNS = [
    "lat", "lng", "coord_confidence", "coord_sources", "coord_notes", "anchor_km",
    "wiki_title", "wiki_url", "wiki_extract", "wikidata_id", "official_website",
    "voyage_description", "fee_text", "fee_source", "hours_text", "entry_fee_inr",
    "scrape_status", "review_notes",
]

session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT})


# ----------------------------------------------------------------- cache + http
def _load_cache():
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {}


CACHE = _load_cache()


def http_json(url, params, key, delay, retries=3):
    if key in CACHE:
        return CACHE[key]
    for attempt in range(retries):
        try:
            time.sleep(delay)
            r = session.get(url, params=params, timeout=25)
            if r.status_code == 200:
                data = r.json()
                CACHE[key] = data
                CACHE_FILE.write_text(json.dumps(CACHE, ensure_ascii=False), encoding="utf-8")
                return data
            if r.status_code in (429, 503):
                time.sleep(2 ** (attempt + 1))
                continue
            print(f"  ! HTTP {r.status_code} for {key[:70]}", file=sys.stderr)
            return None
        except (requests.RequestException, ValueError) as exc:
            print(f"  ! {type(exc).__name__} ({key[:50]}); retry {attempt + 1}/{retries}", file=sys.stderr)
            time.sleep(2 ** (attempt + 1))
    return None


# ----------------------------------------------------------------- text + geo helpers
def norm(s):
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9 ]+", " ", s.lower()).strip()


def core_name(name):
    """'Namdroling Monastery (Golden Temple), Bylakuppe' -> 'Namdroling Monastery'"""
    name = re.sub(r"\(.*?\)", "", name or "").split(",")[0]
    return re.sub(r"\s+", " ", name).strip()


def alt_names(name):
    return [a.strip() for a in re.findall(r"\((.*?)\)", name or "") if len(a.strip()) > 3]


def distinctive(name):
    return [t for t in norm(core_name(name)).split() if t not in GENERIC and len(t) > 2]


def token_match(tok, text_tokens, thr=0.82):
    return any(tok == t or SequenceMatcher(None, tok, t).ratio() >= thr for t in text_tokens)


def haversine_km(a, b):
    r = 6371.0
    p1, p2 = math.radians(a["lat"]), math.radians(b["lat"])
    h = (math.sin((p2 - p1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(math.radians(b["lng"] - a["lng"]) / 2) ** 2)
    return 2 * r * math.asin(math.sqrt(h))


def in_bbox(c):
    return LAT_MIN <= c["lat"] <= LAT_MAX and LNG_MIN <= c["lng"] <= LNG_MAX


def trim(text, limit=MAX_EXTRACT_CHARS):
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    return cut[: cut.rfind(". ") + 1] if ". " in cut else cut


def clean_wiki(text):
    text = re.sub(r"<ref.*?(</ref>|/>)", "", text or "", flags=re.S)
    text = re.sub(r"\{\{[^{}]*\}\}", "", text)
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", text)
    text = re.sub(r"\[https?://\S+ ([^\]]+)\]", r"\1", text)
    text = re.sub(r"'{2,}|<[^>]+>", "", text)
    return re.sub(r"\s+", " ", text).strip()


def anchor_radius(row):
    cats = {c.strip() for c in (row.get("suggested_categories") or "").replace(",", ";").split(";")}
    if cats & WIDE_CATEGORIES or WIDE_KEYWORDS.search(row.get("name", "")):
        return WIDE_ANCHOR_KM
    return DEFAULT_ANCHOR_KM


# ----------------------------------------------------------------- Wikivoyage parsing
def find_templates(wt):
    """Return [(start_pos, inner_text)] for every {{...}} (handles nesting)."""
    out, i, n = [], 0, len(wt)
    while True:
        j = wt.find("{{", i)
        if j == -1:
            break
        depth, k = 0, j
        while k < n - 1:
            two = wt[k:k + 2]
            if two == "{{":
                depth += 1
                k += 2
            elif two == "}}":
                depth -= 1
                k += 2
                if depth == 0:
                    break
            else:
                k += 1
        else:
            break  # unbalanced braces: stop
        out.append((j, wt[j + 2:k - 2]))
        i = j + 2
    return out


def split_top(inner):
    """Split on '|' that are not inside {{ }} or [[ ]]."""
    parts, buf, d1, d2, i = [], [], 0, 0, 0
    while i < len(inner):
        two = inner[i:i + 2]
        if two == "{{":
            d1 += 1; buf.append(two); i += 2; continue
        if two == "}}":
            d1 -= 1; buf.append(two); i += 2; continue
        if two == "[[":
            d2 += 1; buf.append(two); i += 2; continue
        if two == "]]":
            d2 -= 1; buf.append(two); i += 2; continue
        c = inner[i]
        if c == "|" and d1 == 0 and d2 == 0:
            parts.append("".join(buf)); buf = []
        else:
            buf.append(c)
        i += 1
    parts.append("".join(buf))
    return parts


PRICE_RE = re.compile(r"(?:₹|rs\.?|inr)\s*([0-9][0-9,]*)(?:\s*(?:-|–|—|to)\s*(?:₹|rs\.?|inr)?\s*([0-9][0-9,]*))?", re.I)


def parse_price(text):
    """-> (low, high, is_free). Takes the first rupee amount (usually the Indian price)."""
    t = text or ""
    free = bool(re.search(r"\bfree\b", t, re.I)) and not PRICE_RE.search(t)
    m = PRICE_RE.search(t)
    if not m:
        return (0, None, True) if free else (None, None, False)
    lo = int(m.group(1).replace(",", ""))
    hi = int(m.group(2).replace(",", "")) if m.group(2) else None
    return lo, hi, False


def sleep_tier(heading):
    h = (heading or "").lower()
    if "budget" in h:
        return "budget"
    if "mid" in h:
        return "mid"
    if any(w in h for w in ("splurge", "luxury", "upscale")):
        return "luxury"
    return ""


def parse_listings(wikitext, page_title):
    """Extract listing templates ({{see}}, {{do}}, {{sleep}}...) with their section tier."""
    headings = [(m.start(), m.group(1)) for m in re.finditer(r"^=+\s*(.+?)\s*=+\s*$", wikitext, re.M)]
    listings = []
    for pos, inner in find_templates(wikitext):
        parts = split_top(inner)
        tname = parts[0].strip().lower()
        if tname not in LISTING_TYPES:
            continue
        params = {}
        for p in parts[1:]:
            if "=" in p:
                k, v = p.split("=", 1)
                params[k.strip().lower()] = v.strip()
        name = clean_wiki(params.get("name", ""))
        if not name:
            continue
        ltype = params.get("type", tname) if tname == "listing" else tname
        heading = ""
        for hp, ht in headings:
            if hp < pos:
                heading = ht
            else:
                break
        lat = lng = None
        try:
            lat, lng = float(params.get("lat", "")), float(params.get("long", params.get("lon", "")))
        except ValueError:
            pass
        listings.append({
            "page": page_title, "type": ltype, "name": name, "lat": lat, "lng": lng,
            "hours": clean_wiki(params.get("hours", "")), "price": clean_wiki(params.get("price", "")),
            "description": clean_wiki(params.get("content", params.get("description", ""))),
            "wikidata": params.get("wikidata", ""), "url": params.get("url", ""),
            "tier": sleep_tier(heading) if ltype == "sleep" else "",
        })
    return listings


def fetch_wikivoyage_listings(pages):
    listings, seen = [], set()
    for title in pages:
        data = http_json(WIKIVOYAGE_API, {"action": "parse", "page": title, "prop": "wikitext",
                                          "format": "json", "redirects": 1},
                         f"wv|{title}", WIKI_DELAY)
        if not data or "parse" not in data:
            continue
        real = data["parse"]["title"]
        if real in seen:
            continue
        seen.add(real)
        found = parse_listings(data["parse"]["wikitext"]["*"], real)
        print(f"  wikivoyage '{real}': {len(found)} listings")
        listings.extend(found)
    return listings


def match_listing(row, listings):
    """Best Wikivoyage listing for this place (non-sleep), or None."""
    want = [norm(core_name(row["name"]))] + [norm(a) for a in alt_names(row["name"])]
    dist = distinctive(row["name"])
    best, best_s = None, 0.0
    for L in listings:
        if L["type"] == "sleep":
            continue
        ln = norm(L["name"])
        ltoks = ln.split()
        s = max(SequenceMatcher(None, w, ln).ratio() for w in want)
        if dist and all(token_match(t, ltoks) for t in dist):
            s = max(s, 0.85)
        if s > best_s:
            best, best_s = L, s
    return best if best_s >= 0.82 else None


# ----------------------------------------------------------------- Wikipedia / Wikidata
def wikipedia_search(query):
    params = {
        "action": "query", "format": "json", "generator": "search", "gsrsearch": query, "gsrlimit": 5,
        "prop": "extracts|coordinates|info|pageprops", "ppprop": "wikibase_item", "inprop": "url",
        "exintro": 1, "explaintext": 1, "exlimit": "max", "colimit": "max", "redirects": 1,
    }
    data = http_json(WIKIPEDIA_API, params, f"wp|{query}", WIKI_DELAY)
    if not data or "query" not in data:
        return []
    pages = []
    for p in data["query"]["pages"].values():
        co = (p.get("coordinates") or [{}])[0]
        pages.append({
            "title": p.get("title", ""), "url": p.get("fullurl", ""), "extract": p.get("extract", ""),
            "lat": co.get("lat"), "lng": co.get("lon"), "rank": p.get("index", 99),
            "wikidata": (p.get("pageprops") or {}).get("wikibase_item", ""),
        })
    return sorted(pages, key=lambda x: x["rank"])


def title_ok(row, title, extract, known_towns):
    """Is this Wikipedia page really about THIS place (not a look-alike)?"""
    ttoks = norm(title).split()
    dist = distinctive(row["name"])
    if dist and not all(token_match(t, ttoks) or t in norm(extract[:200]).split() for t in dist):
        return False, "title lacks the place's distinctive words"
    own = set(norm(row.get("town", "")).split()) | set(norm(row["name"]).split())
    for town in known_towns:
        tt = norm(town)
        if tt and tt not in own and tt in ttoks:
            return False, f"title names a different town ({town})"
    return True, ""


def best_wikipedia(row, known_towns):
    base = core_name(row["name"])
    town = row.get("town", "")
    variants = [f"{base} {town} Karnataka", f"{base} Karnataka", base] + [f"{a} Karnataka" for a in alt_names(row["name"])]
    seen, best, best_s, why = set(), None, 0.0, ""
    for q in variants:
        if q.lower() in seen:
            continue
        seen.add(q.lower())
        for p in wikipedia_search(q):
            ok, reason = title_ok(row, p["title"], p["extract"], known_towns)
            if not ok:
                why = reason
                continue
            s = SequenceMatcher(None, norm(base), norm(p["title"])).ratio()
            if p["lat"] is not None and in_bbox({"lat": p["lat"], "lng": p["lng"]}):
                s += 0.5
            if "karnataka" in (p["extract"] or "").lower():
                s += 0.3
            if s > best_s:
                best, best_s = p, s
        if best_s >= 1.2:
            break
    return (best, "") if best and best_s >= 0.7 else (None, why)


def wikidata_entity(qid):
    if not qid:
        return {}
    data = http_json(WIKIDATA_API, {"action": "wbgetentities", "ids": qid, "props": "claims", "format": "json"},
                     f"wd|{qid}", WIKI_DELAY)
    try:
        claims = data["entities"][qid]["claims"]
    except (TypeError, KeyError):
        return {}
    out = {}
    try:
        v = claims["P625"][0]["mainsnak"]["datavalue"]["value"]
        out["lat"], out["lng"] = float(v["latitude"]), float(v["longitude"])
    except (KeyError, IndexError, ValueError, TypeError):
        pass
    try:
        out["website"] = claims["P856"][0]["mainsnak"]["datavalue"]["value"]
    except (KeyError, IndexError, TypeError):
        pass
    return out


# ----------------------------------------------------------------- Nominatim
def nominatim(query, limit=3):
    params = {"q": query, "format": "json", "limit": limit, "countrycodes": "in",
              "viewbox": f"{LNG_MIN},{LAT_MAX},{LNG_MAX},{LAT_MIN}", "bounded": 1}
    return http_json(NOMINATIM_API, params, f"nom|{query}", NOMINATIM_DELAY) or []


def geocode_anchor(town, cache_town):
    if not town:
        return None
    if town in cache_town:
        return cache_town[town]
    res = nominatim(f"{town}, Karnataka, India", limit=1)
    anchor = None
    if res:
        c = {"lat": float(res[0]["lat"]), "lng": float(res[0]["lon"])}
        if in_bbox(c):
            anchor = c
    cache_town[town] = anchor
    return anchor


def nominatim_candidates(row):
    base, town = core_name(row["name"]), row.get("town", "")
    dist = distinctive(row["name"])
    out = []
    for q in (f"{base}, {town}, Karnataka", f"{base}, Karnataka"):
        for r in nominatim(q):
            c = {"lat": float(r["lat"]), "lng": float(r["lon"]), "source": "nominatim",
                 "note": r.get("display_name", "")[:60]}
            dtoks = norm(r.get("display_name", "")).split()
            c["weak"] = bool(dist) and not any(token_match(t, dtoks) for t in dist)
            out.append(c)
        if any(not c["weak"] for c in out):
            break
    return out


# ----------------------------------------------------------------- consensus
def choose_coordinate(cands, anchor, radius_km, anchor_name=""):
    """-> (result dict or None, notes list). See module docstring for the rules."""
    notes, ok = [], []
    for c in cands:
        if not in_bbox(c):
            notes.append(f"{c['source']}: outside Karnataka")
            continue
        if anchor and c["source"] != "manual":
            d = haversine_km(anchor, c)
            c["anchor_km"] = d
            if d > radius_km:
                notes.append(f"{c['source']} rejected: {d:.0f} km from {anchor_name} (limit {radius_km:.0f})")
                continue
        ok.append(c)
    if not ok:
        return None, notes

    groups = []
    for c in sorted(ok, key=lambda x: PRIORITY[x["source"]]):
        for g in groups:
            if haversine_km(g[0], c) <= AGREE_KM:
                g.append(c)
                break
        else:
            groups.append([c])

    def strong_sources(g):
        return {m["source"] for m in g if not m.get("weak")}

    groups.sort(key=lambda g: (-len(strong_sources(g)), PRIORITY[g[0]["source"]]))
    best = groups[0]
    strong = sorted(strong_sources(best), key=lambda s: PRIORITY[s])
    lead = next((m for m in best if not m.get("weak")), best[0])

    if lead["source"] == "manual":
        conf = "manual"
    elif len(strong) >= 2:
        conf = "high"
    elif strong:
        conf = "medium"      # one non-weak source that passed the town-distance check
    else:
        conf = "low"
    if len(groups) > 1 and len(strong) < 2 and conf != "manual":
        conf = "low"
        notes.append("sources disagree: " + " vs ".join(f"{g[0]['source']}({g[0]['lat']:.3f},{g[0]['lng']:.3f})" for g in groups[:3]))
    if not anchor and conf in ("medium", "high") and len(strong) < 2:
        conf = "low"
        notes.append("no town anchor to sanity-check against")
    return {"lat": lead["lat"], "lng": lead["lng"], "confidence": conf,
            "sources": "+".join(sorted({m["source"] for m in best}, key=lambda s: PRIORITY[s])),
            "anchor_km": lead.get("anchor_km")}, notes


# ----------------------------------------------------------------- per-row pipeline
def set_empty(row, key, value, refresh=False):
    if value in (None, ""):
        return
    if refresh or not str(row.get(key, "")).strip():
        row[key] = value


def process_row(row, ctx):
    notes = []
    cands = []
    town = row.get("town", "")
    anchor = geocode_anchor(town, ctx["anchors"])
    radius = ctx["anchor_km"] or anchor_radius(row)

    manual = ctx["manual"].get(row["id"])
    if manual:
        cands.append({"lat": manual[0], "lng": manual[1], "source": "manual", "note": "manual_coords.csv"})

    try:
        ex = {"lat": float(row.get("lat", "")), "lng": float(row.get("lng", ""))}
        if not ctx["refresh_coords"] and not manual:
            cands.append({**ex, "source": "existing", "note": "from input CSV"})
    except ValueError:
        pass

    # Wikivoyage
    L = match_listing(row, ctx["listings"]) if ctx["listings"] else None
    if L:
        if L["lat"] is not None:
            cands.append({"lat": L["lat"], "lng": L["lng"], "source": "wikivoyage", "note": L["page"]})
        set_empty(row, "voyage_description", trim(L["description"], 600), ctx["refresh"])
        set_empty(row, "hours_text", L["hours"][:200], ctx["refresh"])
        if L["price"]:
            set_empty(row, "fee_text", L["price"][:200], ctx["refresh"])
            lo, _hi, free = parse_price(L["price"])
            if free or lo is not None:
                set_empty(row, "entry_fee_inr", 0 if free else lo, ctx["refresh"])
                set_empty(row, "fee_source", "wikivoyage (unverified, may be outdated)", ctx["refresh"])
        if L["wikidata"]:
            set_empty(row, "wikidata_id", L["wikidata"], ctx["refresh"])

    # Wikipedia (+ Wikidata)
    page, why = best_wikipedia(row, ctx["towns"])
    if page:
        set_empty(row, "wiki_title", page["title"], ctx["refresh"])
        set_empty(row, "wiki_url", page["url"], ctx["refresh"])
        set_empty(row, "wiki_extract", trim(page["extract"]), ctx["refresh"])
        set_empty(row, "wikidata_id", page["wikidata"], ctx["refresh"])
        if page["lat"] is not None:
            cands.append({"lat": page["lat"], "lng": page["lng"], "source": "wikipedia", "note": page["title"]})
        if len(page["extract"] or "") < 150:
            notes.append("very short Wikipedia text")
    else:
        notes.append("no trustworthy Wikipedia page" + (f" ({why})" if why else ""))

    qid = row.get("wikidata_id", "")
    ent = wikidata_entity(qid) if qid else {}
    if ent.get("lat") is not None:
        cands.append({"lat": ent["lat"], "lng": ent["lng"], "source": "wikidata", "note": qid})
    set_empty(row, "official_website", ent.get("website", ""), ctx["refresh"])

    # Nominatim last (rate-limited): always ask, it is the independent cross-check
    cands.extend(nominatim_candidates(row))

    result, cnotes = choose_coordinate(cands, anchor, radius, anchor_name=town)
    notes.extend(cnotes)
    if result:
        row["lat"], row["lng"] = f"{result['lat']:.5f}", f"{result['lng']:.5f}"
        row["coord_confidence"], row["coord_sources"] = result["confidence"], result["sources"]
        row["anchor_km"] = f"{result['anchor_km']:.1f}" if result["anchor_km"] is not None else ""
    else:
        if not manual:
            row["lat"] = row["lng"] = ""
        row["coord_confidence"], row["coord_sources"], row["anchor_km"] = "none", "", ""
        notes.append("NO trustworthy coordinates: add to manual_coords.csv")
    if anchor is None:
        notes.append(f"could not geocode town '{town}' (distance check skipped)")
    row["coord_notes"] = "; ".join(n for n in cnotes)[:300]

    has_text = bool(row.get("wiki_extract") or row.get("voyage_description"))
    good_coord = row["coord_confidence"] in ("high", "medium", "manual")
    row["scrape_status"] = "ok" if (good_coord and has_text) else ("partial" if (good_coord or has_text) else "failed")
    if row["coord_confidence"] == "low":
        notes.append("low-confidence coordinates: check on a map")
    row["review_notes"] = "; ".join(notes)[:400]
    return row


# ----------------------------------------------------------------- io + main
def read_manual(path):
    out = {}
    p = Path(path)
    if not p.exists():
        return out
    for r in csv.DictReader(open(p, newline="", encoding="utf-8")):
        try:
            out[r["id"]] = (float(r["lat"]), float(r["lng"]))
        except (ValueError, KeyError):
            continue
    return out


def write_sleep_prices(listings, path):
    rows = []
    for L in listings:
        if L["type"] != "sleep" or not L["price"]:
            continue
        lo, hi, _ = parse_price(L["price"])
        rows.append({"page": L["page"], "tier": L["tier"], "name": L["name"], "price_text": L["price"][:120],
                     "price_min_inr": lo if lo is not None else "", "price_max_inr": hi if hi is not None else ""})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["page", "tier", "name", "price_text", "price_min_inr", "price_max_inr"])
        w.writeheader()
        w.writerows(rows)
    by = {}
    for r in rows:
        if r["tier"] and r["price_min_inr"] != "":
            by.setdefault((r["page"], r["tier"]), []).append(r["price_min_inr"])
    for (page, tier), vals in sorted(by.items()):
        print(f"  sleep prices {page:<14} {tier:<7} median from-price INR {statistics.median(vals):.0f}  (n={len(vals)})")
    return len(rows)


def main():
    ap = argparse.ArgumentParser(description="Collect places with sanity-checked coordinates and extra fields.")
    ap.add_argument("--in", dest="infile", default="../data/karnataka_mysuru_division_places.csv")
    ap.add_argument("--out", dest="outfile", default="../data/places_v2.csv")
    ap.add_argument("--manual", default="../data/manual_coords.csv", help="CSV with id,lat,lng that always wins")
    ap.add_argument("--only", help="comma-separated ids to process")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--refresh-coords", action="store_true", help="ignore coordinates already in the input CSV")
    ap.add_argument("--refresh", action="store_true", help="overwrite text fields that are already filled")
    ap.add_argument("--no-wikivoyage", action="store_true")
    ap.add_argument("--anchor-km", type=float, default=None, help="override the town-distance limit for all places")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.infile, newline="", encoding="utf-8")))
    fields = list(rows[0].keys())
    for c in NEW_COLUMNS:
        if c not in fields:
            fields.append(c)
    for r in rows:
        for c in fields:
            r.setdefault(c, "")

    todo = rows
    if args.only:
        wanted = set(args.only.split(","))
        todo = [r for r in rows if r["id"] in wanted]
    if args.limit:
        todo = todo[: args.limit]

    listings = []
    if not args.no_wikivoyage:
        print("Reading Wikivoyage region pages...")
        listings = fetch_wikivoyage_listings(REGION_PAGES)
        n = write_sleep_prices(listings, Path(args.outfile).with_name("wikivoyage_sleep_prices.csv"))
        print(f"  {len(listings)} listings, {n} priced sleep listings saved\n")

    ctx = {
        "anchors": {}, "listings": listings, "manual": read_manual(args.manual),
        "towns": sorted({r["town"] for r in rows if r.get("town")}),
        "refresh": args.refresh, "refresh_coords": args.refresh_coords, "anchor_km": args.anchor_km,
    }
    print(f"Processing {len(todo)} of {len(rows)} places ({len(ctx['manual'])} manual coordinates loaded)\n")

    for i, row in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {row['name']}")
        try:
            process_row(row, ctx)
        except Exception as exc:  # one bad row must never kill the run
            row["scrape_status"], row["review_notes"] = "failed", f"exception: {type(exc).__name__}: {exc}"
        print(f"    -> {row['scrape_status']:<8} coords={row['coord_confidence']:<7} {row['coord_sources']:<28} {row['review_notes'][:90]}")
        with open(args.outfile, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    # report
    conf = {}
    for r in todo:
        conf[r["coord_confidence"]] = conf.get(r["coord_confidence"], 0) + 1
    lines = [f"Coordinate confidence: {conf}", ""]
    need = [r for r in todo if r["coord_confidence"] in ("none", "low") or r["scrape_status"] != "ok"]
    lines.append(f"{len(need)} places need a human look (add coordinates to {args.manual}):")
    for r in need:
        lines.append(f"  {r['id']:<22} coords={r['coord_confidence']:<6} status={r['scrape_status']:<8} {r['review_notes'][:110]}")
    report = "\n".join(lines)
    Path(args.outfile).with_name("collection_report.txt").write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"\nSaved {args.outfile}")


if __name__ == "__main__":
    main()