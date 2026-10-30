"""
hours.py - TYPICAL opening windows by kind of place. These are rough defaults, not live data.

The scheduler uses them so it never plans the palace for 16:00 or a safari for noon. Real hours change
(closed days, festivals, seasons), so the page still tells travellers to check before they go. When live
opening hours are added later, they can replace typical_windows() without touching the scheduler.

A window is (open, close) in 24-hour decimal hours, e.g. (10.0, 17.5) = 10:00 to 17:30.
A place can have several windows (a wildlife park has a morning and an afternoon safari).
"""

import re

# first match on the place NAME wins
RULES = [
    (r"kudremukh|trek", [(6.0, 16.0)]),                                              # treks start early
    (r"\bdam\b", [(8.0, 19.0)]),                                                      # dams and their gardens
    (r"bird sanctuary", [(9.0, 17.5)]),                                              # boating hours
    (r"national park|wildlife sanctuary|tiger reserve|reservoir|backwaters", [(6.0, 9.5), (15.0, 18.0)]),  # safaris: a 3-hour morning or afternoon session
    (r"elephant camp", [(9.0, 17.0)]),
    (r"zoo|zoological", [(8.5, 17.5)]),
    (r"palace", [(10.0, 17.5)]),
    (r"museum|gallery", [(10.0, 17.0)]),
    (r"island", [(9.0, 17.0)]),                                                      # ferry hours
    (r"fort|battery", [(8.5, 17.5)]),
    (r"cathedral|church|chapel", [(6.0, 19.0)]),
    (r"market", [(8.0, 20.0)]),
    (r"hotel|sweet", [(7.0, 21.0)]),
    (r"monastery", [(7.0, 18.0)]),
    (r"falls", [(7.0, 18.0)]),
    (r"peak|viewpoint|end point|seat|hills|betta", [(6.0, 18.0)]),                   # daylight
    (r"beach", [(6.0, 19.0)]),
    (r"dam|gardens|garden|lake", [(8.0, 19.0)]),
]


def typical_windows(place):
    """List of (open, close) windows for this place."""
    name = place["name"].lower()
    for pattern, windows in RULES:
        if re.search(pattern, name):
            return windows
    cats = set(place.get("categories", []))
    if "spiritual" in cats and "heritage" in cats:
        return [(6.5, 18.0)]            # protected heritage temples (Hoysala etc.) close at dusk
    if "spiritual" in cats:
        return [(6.0, 20.0)]            # working temples
    return [(8.0, 18.0)]


def _fmt(h):
    return f"{int(h):02d}:{int(round((h - int(h)) * 60)):02d}"


def format_windows(windows):
    return " and ".join(f"{_fmt(o)} to {_fmt(c)}" for o, c in windows)