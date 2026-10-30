
"""
remove_places.py - delete places by id from one or more CSV files (a .bak backup is made first).

Usage:
    python remove_places.py crg_coffee ../data/places_v1.csv ../data/places_v2.csv ../data/manual_coords.csv
    python remove_places.py crg_coffee,mys_mylari ../data/places_v1.csv

First argument: one id or several separated by commas. Remaining arguments: the CSV files to clean.
Files that have no "id" column, or no matching rows, are left untouched.
"""

import csv
import shutil
import sys


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    ids = {i.strip() for i in sys.argv[1].split(",") if i.strip()}
    for path in sys.argv[2:]:
        try:
            with open(path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                fields = reader.fieldnames or []
                rows = list(reader)
        except FileNotFoundError:
            print(f"{path}: not found, skipped")
            continue
        if "id" not in fields:
            print(f"{path}: no 'id' column, skipped")
            continue
        kept = [r for r in rows if r["id"] not in ids]
        removed = len(rows) - len(kept)
        if not removed:
            print(f"{path}: none of {sorted(ids)} found, unchanged")
            continue
        shutil.copy(path, path + ".bak")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(kept)
        print(f"{path}: removed {removed} row(s), {len(kept)} left (backup: {path}.bak)")


if __name__ == "__main__":
    main()