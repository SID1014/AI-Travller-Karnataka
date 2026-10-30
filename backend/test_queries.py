"""
test_queries.py - eyeball and measure retrieval quality before building anything on top.

For each query we list which categories we EXPECT in the results. Precision@5 is the
share of the top-5 places whose categories overlap the expected set. It's crude, but
it gives you a number for your PPT/viva ("retrieval precision@5 = 0.8 on 12 test queries")
and, more usefully, shows you which queries fail so you can fix data or text.

Usage:
    python test_queries.py ../data/places_v1.csv
    python test_queries.py ../data/places_v1.csv --backend tfidf
"""

import argparse

from data_loader import load_places
from retrieval import PlaceRetriever

TESTS = [
    ("ancient temples and Hoysala architecture",          {"heritage", "spiritual"}),
    ("tiger safari and jungle wildlife",                  {"wildlife"}),
    ("relaxing beaches and seafood",                      {"beach"}),
    ("misty hill station with coffee plantations",        {"hill-station", "nature", "food"}),
    ("trekking and adventure in the mountains",           {"adventure", "hill-station"}),
    ("waterfalls in the monsoon",                         {"nature"}),
    ("royal palaces and history",                         {"heritage", "culture"}),
    ("pilgrimage and spiritual places",                   {"spiritual"}),
    ("local food and street markets",                     {"food", "culture"}),
    ("bird watching and quiet nature",                    {"wildlife", "nature"}),
    ("Buddhist monastery and Tibetan culture",            {"spiritual", "culture"}),
    ("family friendly day trip with easy walking",        {"nature", "culture", "heritage", "beach"}),
]


def precision_at_k(results, expected, k=5):
    top = results[:k]
    hits = sum(1 for place, _ in top if expected & set(place["categories"]))
    return hits / max(len(top), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--backend", default="auto", choices=["auto", "st", "tfidf"])
    ap.add_argument("--show", type=int, default=6, help="how many results to print per query")
    args = ap.parse_args()

    places, issues = load_places(args.csv)
    print(f"{len(places)} places loaded, {sum(1 for i in issues if i[0] == 'ERROR')} skipped\n")
    retriever = PlaceRetriever(places, backend=args.backend)

    precisions = []
    for query, expected in TESTS:
        results = retriever.search(query, k=max(args.show, 5))
        p5 = precision_at_k(results, expected)
        precisions.append(p5)
        print(f'\nQ: "{query}"   expected={sorted(expected)}   P@5={p5:.2f}')
        for place, score in results[:args.show]:
            mark = "+" if expected & set(place["categories"]) else " "
            print(f"  {mark} {score:5.3f}  {place['name'][:48]:<48} {place['cluster']:<22} {','.join(place['categories'])}")

    avg = sum(precisions) / len(precisions)
    print(f"\nMean precision@5 over {len(TESTS)} queries: {avg:.2f}")
    weak = [q for (q, _), p in zip(TESTS, precisions) if p < 0.6]
    if weak:
        print("Weak queries (P@5 < 0.6):")
        for q in weak:
            print("  -", q)


if __name__ == "__main__":
    main()