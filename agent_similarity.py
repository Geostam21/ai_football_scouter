"""
agent_similarity.py — the Similarity Agent (on-demand, not part of the main flow).

Resolves a referenced player name from free text (EN / Greek / greeklish), picking
the most prominent match when several players share a surname, and surfacing the
alternatives so the user can switch. Then finds similar players via SimilarityIndex.
"""
from __future__ import annotations
import re
import unicodedata
import pandas as pd
from ml import SimilarityIndex


def _norm(s: str) -> str:
    """Lowercase and strip accents for robust matching."""
    s = unicodedata.normalize("NFKD", str(s).lower())
    return "".join(c for c in s if not unicodedata.combining(c))


# Greek letter -> Latin transliteration, so "ορμπελιν" can match "Orbelin".
_GREEK_MAP = {
    "α": "a", "β": "v", "γ": "g", "δ": "d", "ε": "e", "ζ": "z", "η": "i",
    "θ": "th", "ι": "i", "κ": "k", "λ": "l", "μ": "m", "ν": "n", "ξ": "x",
    "ο": "o", "π": "p", "ρ": "r", "σ": "s", "ς": "s", "τ": "t", "υ": "y",
    "φ": "f", "χ": "ch", "ψ": "ps", "ω": "o", "μπ": "b", "ντ": "d", "γκ": "g",
    "τσ": "ts", "τζ": "tz",
}


def _greek_to_latin(s: str) -> str:
    s = s.lower()
    # digraphs first
    for gr, lat in [("μπ", "b"), ("ντ", "d"), ("γκ", "g"), ("τσ", "ts"), ("τζ", "tz")]:
        s = s.replace(gr, lat)
    out = "".join(_GREEK_MAP.get(ch, ch) for ch in s)
    return out


_LIKE_PATTERNS = [
    r"(?:similar to|like|comparable to|alternative to|in the mould of|"
    r"σαν τον|σαν τη|σαν τη ν|παρ[οό]μοι[οα] με|αντ[ίι]στοιχ[οα] (?:με|του))\s+(.+?)"
    r"(?:\s+but\b|\s+under\b|\s+below\b|\s+cheaper\b|\s+with\b|\s+αλλ[άα]\b|"
    r"\s+με\b|\s+κ[άα]τω\b|\s+φθην|\s+\d|$)",
]

_NAME_STOP = {"me", "with", "budget", "under", "below", "cheaper", "and", "a", "the",
              "μας", "με", "budjet", "κατω", "των", "εναν", "παικτη", "που", "budget"}


def _extract_fragment(text: str) -> str | None:
    """Pull the player-name fragment out of a 'like X ...' phrase."""
    low = text.lower()
    for pat in _LIKE_PATTERNS:
        m = re.search(pat, low)
        if not m:
            continue
        raw = m.group(1).strip()
        tokens = []
        for tok in raw.split():
            if tok in _NAME_STOP or any(ch.isdigit() for ch in tok):
                break
            tokens.append(tok)
        frag = " ".join(tokens).strip()
        if len(frag) >= 3:
            return frag
    return None


def resolve_player(fragment: str, players: pd.DataFrame,
                   llm_normaliser=None) -> dict:
    """Resolve a name fragment to player(s).

    Returns {'match': name|None, 'alternatives': [names], 'ambiguous': bool}.
    - Tries the fragment as-is, its Greek->Latin transliteration, and (optionally)
      an LLM-normalised spelling.
    - When multiple players share the matched surname, returns the most valuable
      as 'match' and the rest as 'alternatives'.
    """
    names = players["Name"].tolist()
    candidates = {fragment, _greek_to_latin(fragment)}
    if llm_normaliser:
        try:
            norm = llm_normaliser(fragment)
            if norm:
                candidates.add(norm)
        except Exception:
            pass

    matched = []
    for frag in candidates:
        nf = _norm(frag)
        if len(nf) < 3:
            continue
        # full containment
        for name in names:
            n = _norm(name)
            if nf == n or (nf in n and len(nf) >= 5):
                matched.append(name)
        # surname-token match (len>=4)
        frag_words = {w for w in nf.split() if len(w) >= 4}
        if frag_words:
            for name in names:
                if frag_words & set(_norm(name).split()):
                    matched.append(name)

    matched = list(dict.fromkeys(matched))  # dedupe, keep order
    if not matched:
        return {"match": None, "alternatives": [], "ambiguous": False}

    # rank matches by value (fame proxy)
    sub = players[players["Name"].isin(matched)].copy()
    sub = sub.sort_values("value_mid", ascending=False, na_position="last")
    ordered = sub["Name"].tolist()
    return {
        "match": ordered[0],
        "alternatives": ordered[1:8],
        "ambiguous": len(ordered) > 1,
    }


def detect_similarity_query(text: str, known_names: list[str]) -> str | None:
    """Lightweight boolean-ish check kept for backward compatibility.

    Returns a rough match name or None. The richer resolve_player() is preferred
    in the app, but this keeps older call sites working.
    """
    frag = _extract_fragment(text)
    if not frag:
        return None
    for cand in (frag, _greek_to_latin(frag)):
        nf = _norm(cand)
        if len(nf) < 4:
            continue
        for name in known_names:
            n = _norm(name)
            if nf == n or (nf in n and len(nf) >= 5):
                return name
        frag_words = {w for w in nf.split() if len(w) >= 4}
        for name in known_names:
            if frag_words & set(_norm(name).split()):
                return name
    return None


def extract_reference(text: str):
    """Return the raw name fragment from a similarity query (or None)."""
    return _extract_fragment(text)


class SimilarityAgent:
    def __init__(self, players: pd.DataFrame, index: SimilarityIndex | None = None):
        self.players = players
        self.index = index or SimilarityIndex(players)

    def find_similar(self, name: str, k: int = 10,
                     max_value: float | None = None,
                     cheaper_only: bool = False,
                     same_position: bool = True,
                     sort_by: str = "similarity",
                     filters: dict | None = None) -> pd.DataFrame:
        return self.index.similar_to(
            name, k=k, max_value=max_value,
            cheaper_than_target=cheaper_only, same_position=same_position,
            sort_by=sort_by, filters=filters,
        )
