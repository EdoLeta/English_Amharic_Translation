"""
Offline English→Amharic translation engine.

Pipeline:
  1. Exact match against the 31 K Bible corpus.
  2. TF-IDF cosine-similarity "Translation Memory" — nearest English sentence.
  3. Word co-occurrence dictionary fallback for very dissimilar queries.

Models are built once and cached to disk so subsequent starts are fast.
"""

from __future__ import annotations
import os
import re
import unicodedata
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

_HERE        = Path(__file__).parent
_CORPUS_PATH = _HERE / "data" / "en_am.csv"
_CACHE_PATH  = _HERE / "data" / "translation_cache.pkl"

# ── text normalisation ─────────────────────────────────────────────────────

_VERSE_NUM = re.compile(r"^\d+\s+")
_REFS      = re.compile(r"[\+\*]")
_SPACES    = re.compile(r"\s+")


def _clean_en(text: str) -> str:
    text = _VERSE_NUM.sub("", str(text).strip())
    text = _REFS.sub("", text).lower()
    return _SPACES.sub(" ", text).strip()


def _clean_am(text: str) -> str:
    text = _VERSE_NUM.sub("", str(text).strip())
    text = _REFS.sub("", text)
    return _SPACES.sub(" ", unicodedata.normalize("NFC", text)).strip()


# ── corpus loading ─────────────────────────────────────────────────────────

def _load_corpus() -> list[tuple[str, str]]:
    df = pd.read_csv(_CORPUS_PATH).dropna()
    pairs = [
        (_clean_en(e), _clean_am(a))
        for e, a in zip(df["English"], df["Amharic"])
        if str(e).strip() and str(a).strip()
    ]
    return pairs


# ── word co-occurrence dictionary ─────────────────────────────────────────
# For each English word, store the single most-common Amharic word that
# appears in paired sentences. Memory: O(|en_vocab| × constant).

def _build_word_dict(pairs: list[tuple[str, str]]) -> dict[str, str]:
    co: dict[str, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))
    for en_sent, am_sent in pairs:
        en_words = set(en_sent.split())
        am_words = am_sent.split()
        for ew in en_words:
            for aw in am_words:
                co[ew][aw] += 1

    return {ew: max(aw_counts, key=aw_counts.get)
            for ew, aw_counts in co.items()}


# ── TF-IDF translation memory ──────────────────────────────────────────────

def _build_tfidf(en_sentences: list[str]):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=50_000)
    mat = vec.fit_transform(en_sentences)
    return vec, mat


# ── Engine ─────────────────────────────────────────────────────────────────

class TranslationEngine:
    def __init__(self):
        self._ready       = False
        self._pairs: list[tuple[str, str]] = []
        self._vectorizer  = None
        self._tfidf_mat   = None
        self._word_dict: dict[str, str] = {}

    # -- loading -----------------------------------------------------------

    def load(self, force_rebuild: bool = False) -> None:
        if self._ready:
            return

        if not force_rebuild and _CACHE_PATH.exists():
            print("[translator] Loading cached index …")
            with open(_CACHE_PATH, "rb") as f:
                data = pickle.load(f)
            self._pairs      = data["pairs"]
            self._vectorizer = data["vectorizer"]
            self._tfidf_mat  = data["tfidf_mat"]
            self._word_dict  = data["word_dict"]
        else:
            print("[translator] Building translation index from corpus …")
            self._pairs = _load_corpus()
            en_sents    = [p[0] for p in self._pairs]

            print(f"[translator] Building TF-IDF index ({len(en_sents):,} sentences) …")
            self._vectorizer, self._tfidf_mat = _build_tfidf(en_sents)

            print("[translator] Building word co-occurrence dictionary …")
            self._word_dict = _build_word_dict(self._pairs)

            print("[translator] Saving cache …")
            _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(_CACHE_PATH, "wb") as f:
                pickle.dump({
                    "pairs":      self._pairs,
                    "vectorizer": self._vectorizer,
                    "tfidf_mat":  self._tfidf_mat,
                    "word_dict":  self._word_dict,
                }, f)
            print("[translator] Cache saved.")

        self._ready = True

    # -- translation -------------------------------------------------------

    def translate(self, text: str) -> dict:
        """
        Returns:
          method       – "exact" | "memory" | "word-lookup"
          translation  – Amharic string
          score        – float 0-1
          matched_en   – the corpus English sentence that was matched
        """
        self.load()
        query = _clean_en(text)
        if not query:
            return {"method": "error", "translation": "", "score": 0.0, "matched_en": ""}

        # 1. exact match
        for en, am in self._pairs:
            if en == query:
                return {"method": "exact", "translation": am, "score": 1.0, "matched_en": en}

        # 2. TF-IDF nearest neighbour
        from sklearn.metrics.pairwise import cosine_similarity
        q_vec  = self._vectorizer.transform([query])
        sims   = cosine_similarity(q_vec, self._tfidf_mat)[0]
        best_i = int(np.argmax(sims))
        score  = float(sims[best_i])

        if score >= 0.20:
            en, am = self._pairs[best_i]
            return {"method": "memory", "translation": am,
                    "score": round(score, 3), "matched_en": en}

        # 3. word co-occurrence fallback
        am_words = [self._word_dict[w] for w in query.split() if w in self._word_dict]
        translation = " ".join(am_words) if am_words else "(no translation found)"
        return {"method": "word-lookup", "translation": translation,
                "score": round(score, 3), "matched_en": ""}


engine = TranslationEngine()
