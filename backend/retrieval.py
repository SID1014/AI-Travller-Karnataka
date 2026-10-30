"""
retrieval.py - semantic search over the places.

Backends:
  * "st"    : sentence-transformers embeddings (best quality, needs a one-time model download)
  * "tfidf" : scikit-learn TF-IDF (no download, works offline)
  * "auto"  : try "st", silently fall back to "tfidf"

The TF-IDF fallback is deliberate: if the model can't load on your deployment
host, the app still works instead of crashing during the demo.

With ~70 places a plain NumPy matrix + cosine similarity is all you need.
(ChromaDB/FAISS only pay off at thousands of items.)

Usage:
    from data_loader import load_places
    from retrieval import PlaceRetriever
    places, _ = load_places("../data/places_v1.csv")
    r = PlaceRetriever(places)
    for place, score in r.search("quiet temples and filter coffee", k=10):
        print(place["name"], round(score, 3))
"""

import hashlib
from pathlib import Path

import numpy as np

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# For Hindi/Kannada queries later, try: "intfloat/multilingual-e5-small"
CACHE_DIR = Path(".embedding_cache")

# Short queries embed poorly; this gives the model a little context.
QUERY_PREFIX = "Tourist trip in Karnataka: "


class PlaceRetriever:
    def __init__(self, places, backend="auto", model_name=DEFAULT_MODEL, use_cache=True):
        self.places = places
        self.model_name = model_name
        self.use_cache = use_cache
        self.backend = None
        self._model = None
        self._vectorizer = None
        self._matrix = None
        texts = [p["search_text"] for p in places]

        if backend in ("auto", "st"):
            try:
                self._init_st(texts)
                self.backend = "st"
            except Exception as exc:  # missing package, no internet, etc.
                if backend == "st":
                    raise
                print(f"[retrieval] sentence-transformers unavailable ({type(exc).__name__}); "
                      f"falling back to TF-IDF")
        if self.backend is None:
            self._init_tfidf(texts)
            self.backend = "tfidf"
        print(f"[retrieval] backend = {self.backend}, {len(places)} places indexed")

    # ------------------------------------------------------------ backends
    def _init_st(self, texts):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(self.model_name)

        key = hashlib.md5((self.model_name + "||".join(texts)).encode("utf-8")).hexdigest()[:16]
        cache_file = CACHE_DIR / f"{key}.npy"
        if self.use_cache and cache_file.exists():
            self._matrix = np.load(cache_file)
            return
        self._matrix = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        if self.use_cache:
            CACHE_DIR.mkdir(exist_ok=True)
            np.save(cache_file, self._matrix)

    def _init_tfidf(self, texts):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._vectorizer = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2), sublinear_tf=True, min_df=1)
        self._matrix = self._vectorizer.fit_transform(texts)

    # ------------------------------------------------------------ search
    def _scores(self, query):
        if self.backend == "st":
            q = self._model.encode([QUERY_PREFIX + query], normalize_embeddings=True)[0]
            return self._matrix @ q                      # cosine (vectors are normalised)
        q = self._vectorizer.transform([query])
        return (self._matrix @ q.T).toarray().ravel()    # TF-IDF vectors are L2-normalised

    def search(self, query, k=25, exclude_ids=None):
        """Return up to k (place, similarity) pairs, best first."""
        scores = self._scores(query)
        order = np.argsort(-scores)
        exclude = set(exclude_ids or [])
        results = []
        for idx in order:
            place = self.places[idx]
            if place["id"] in exclude:
                continue
            results.append((place, float(scores[idx])))
            if len(results) >= k:
                break
        return results