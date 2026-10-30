"""
apply_manual_fixes.py - repair the rows the scraper got wrong.

What it does (safe to re-run, makes a .bak backup first):
  1. Clears the text of places matched to the WRONG Wikipedia page (BAD_WIKI_MATCH)
  2. Adds a `tags` column with hand-written keywords for places with thin/no text
     (data_loader.py puts these into the embedded search text)
  3. Applies manual coordinates from COORDS below for places the scraper couldn't locate

Usage:
    1. Fill in COORDS (lat, lng) from Google Maps: right-click the place -> click the numbers to copy.
    2. python apply_manual_fixes.py ../data/places_v1.csv
"""

import csv
import shutil
import sys

# ---- 1. Fill these in from Google Maps (leave None to skip) -------------------------
COORDS = {
    # --- missing coordinates (skipped by the loader before)
    "kab_mmhills":       None,   # e.g. (11.xxxx, 77.xxxx)  Male Mahadeshwara Hills
    "crg_mandalpatti":   None,   # Mandalpatti Viewpoint, Kodagu
    "chk_bhadra":        None,   # Bhadra Wildlife Sanctuary (Lakkavalli entrance)
    # --- WRONG coordinates (scraped from the wrong Wikipedia page): re-enter from Google Maps
    "mys_somanathapura": None,   # Chennakesava Temple, Somanathapura (had Belur's coordinates)
    "mng_sultanbattery": None,   # Sultan Battery, Boloor, Mangaluru
    "mys_gurusweets":    None,   # Guru Sweet Mart, Mysuru
    "mys_mylari":        None,   # Mylari Hotel, Nazarbad, Mysuru
    "crg_coffee":        None,   # pick a real estate/town you want, e.g. near Madikeri or Virajpet
}

# ---- 2. Places whose scraped Wikipedia text is about the wrong thing ----------------
# collect_places_v2.py already rejects wrong matches. Leave this False when working on v2 output,
# otherwise coordinates of the ids below would be blanked. Set True only for the old v1 file.
CLEAR_BAD_MATCHES = False
BAD_WIKI_MATCH = ["mng_sultanbattery", "mys_gurusweets", "mys_mylari", "crg_coffee", "mys_somanathapura"]

# ---- 3. Hand-written keywords (factual, short). Add more for any thin-text place. ---
TAGS = {
    "mng_sultanbattery": "Tipu Sultan watchtower fort, 18th century battery overlooking the Gurupura river, "
                         "Boloor Mangaluru, history and heritage viewpoint",
    "mys_somanathapura": "13th century Hoysala temple, star-shaped plan, intricate carved friezes, "
                         "Keshava temple near Mysuru, architecture and history",
    "mys_gurusweets":    "Mysore Pak, traditional sweet shop, famous Mysuru sweets, local food and souvenirs",
    "mys_mylari":        "Mysore dosa, iconic breakfast restaurant, butter dosa, Nazarbad, local Mysuru food",
    "crg_coffee":        "coffee plantation tour, Arabica and Robusta estates, spice plantation walk, "
                         "Kodagu estate homestay, misty hills, filter coffee tasting",
    "crg_mandalpatti":   "viewpoint above the clouds, jeep ride, sunrise and misty hills, Kodagu hill scenery",
    "has_bisle":         "Western Ghats forest viewpoint, Sakleshpur, scenic ghat road, Kumara Parvatha views",
    "chk_bhadra":        "tiger reserve, jungle safari, Bhadra river, Lakkavalli, wildlife and birds",
    "kab_mmhills":       "Male Mahadeshwara temple hill, forest and wildlife sanctuary, pilgrimage, Chamarajanagar",
    "udp_kapu":          "Kapu beach, red-and-white lighthouse, Udupi coast, sunset, relaxing seaside",
    "udp_stmarys":       "St. Mary's Island, hexagonal basalt rock formations, ferry from Malpe, island beach",
    "udp_maravanthe":    "Maravanthe beach, highway between sea and river, sunset, quiet coast",
    "mng_someshwar":     "Someshwar beach, rocky shore, Someshwar temple, Ullal, quiet coast",
}


def main(path):
    shutil.copy(path, path + ".bak")
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = list(reader.fieldnames)
    for col in ("tags", "wiki_title", "wiki_url", "wiki_extract", "source_url",
                "lat", "lng", "scrape_status", "review_notes"):
        if col not in fields:
            fields.append(col)

    by_id = {r["id"]: r for r in rows}
    cleared = tagged = located = 0

    for pid in (BAD_WIKI_MATCH if CLEAR_BAD_MATCHES else []):
        r = by_id.get(pid)
        if not r:
            continue
        had_text = bool(r.get("wiki_title") or r.get("wiki_extract"))
        r["wiki_title"] = r["wiki_url"] = r["wiki_extract"] = ""
        r["source_url"] = ""
        # coordinates scraped from the wrong page are wrong too: clear them unless re-entered below
        if not COORDS.get(pid):
            r["lat"] = r["lng"] = ""
        r["review_notes"] = "wrong wiki match cleared; using manual tags"
        cleared += 1 if had_text else 0

    for pid, tags in TAGS.items():
        r = by_id.get(pid)
        if r:
            r["tags"] = tags
            tagged += 1

    for pid, ll in COORDS.items():
        r = by_id.get(pid)
        if r and ll:
            r["lat"], r["lng"] = f"{ll[0]:.5f}", f"{ll[1]:.5f}"
            r["scrape_status"] = "ok"
            r["review_notes"] = "coordinates set manually"
            located += 1

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print(f"cleared wrong text: {cleared}, tags added: {tagged}, coordinates set: {located}")
    still = [pid for pid in COORDS if by_id.get(pid) and not by_id[pid].get("lat")]
    if still:
        print("STILL MISSING COORDINATES (fill COORDS and re-run):", still)
    print(f"backup saved to {path}.bak")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "../data/places_v1.csv")