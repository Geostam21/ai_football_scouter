"""
agents_core.py — the two deterministic agents.

CandidateAgent : applies HARD constraints (position, age, budget) -> candidate pool
ScoringAgent   : applies SOFT preferences (weighted attributes) -> ranked shortlist

These are pure functions of the data + spec, so results are 100% reproducible.
The LLM agents (Requirements, Reporting) sit on either side of these.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from data import ALL_ATTR_CODES, ATTRIBUTES


# ------------------------------------------------------------------ #
# The "spec" is the shared contract between all agents.
# The Requirements Agent produces it; everyone downstream consumes it.
# ------------------------------------------------------------------ #
# spec = {
#   "position_codes": ["AML"],              # any of these satisfies the filter
#   "max_age": 25, "min_age": None,
#   "max_value": 15_000_000, "min_value": None,
#   "weights": {"Pac": 0.3, "Dri": 0.3, "Fin": 0.2, "Acc": 0.1, "Sta": 0.1},
#   "top_n": 10,
# }


class CandidateAgent:
    """Filter the full dataset down to players that meet the hard constraints."""

    def __init__(self, players: pd.DataFrame):
        self.players = players

    def run(self, spec: dict) -> pd.DataFrame:
        df = self.players
        mask = pd.Series(True, index=df.index)
        notes = []

        # position: keep rows whose role list intersects the requested codes
        codes = spec.get("position_codes") or []
        if codes:
            codeset = set(codes)
            mask &= df["positions"].apply(lambda roles: bool(codeset & set(roles)))
            notes.append(f"position in {codes}")

        # age
        if spec.get("max_age") is not None:
            mask &= df["Age"] <= spec["max_age"]
            notes.append(f"age<={spec['max_age']}")
        if spec.get("min_age") is not None:
            mask &= df["Age"] >= spec["min_age"]
            notes.append(f"age>={spec['min_age']}")

        # budget (use value_mid; keep players with unknown value only if no budget set)
        if spec.get("max_value") is not None:
            mask &= (df["value_mid"].notna()) & (df["value_mid"] <= spec["max_value"])
            notes.append(f"value<={spec['max_value']:,.0f}")
        if spec.get("min_value") is not None:
            mask &= (df["value_mid"].notna()) & (df["value_mid"] >= spec["min_value"])
            notes.append(f"value>={spec['min_value']:,.0f}")

        # league filter: 'Based' contains any of the requested league substrings
        league_subs = spec.get("league_substrings") or []
        if league_subs and "Based" in df.columns:
            based = df["Based"].fillna("")
            lg_mask = pd.Series(False, index=df.index)
            for sub in league_subs:
                lg_mask |= based.str.contains(sub, case=False, regex=False)
            mask &= lg_mask
            notes.append(f"league in {len(league_subs)} groups")

        # nationality filter: keep only players whose Nat is in the allowed set
        nat_set = spec.get("nationality_set")
        if nat_set and "Nat" in df.columns:
            mask &= df["Nat"].isin(nat_set)
            notes.append(f"nationality in set ({len(nat_set)})")

        # height
        if spec.get("min_height") is not None and "height_cm" in df.columns:
            mask &= df["height_cm"] >= spec["min_height"]
            notes.append(f"height>={spec['min_height']}cm")
        if spec.get("max_height") is not None and "height_cm" in df.columns:
            mask &= df["height_cm"] <= spec["max_height"]
            notes.append(f"height<={spec['max_height']}cm")

        # preferred foot
        foot = spec.get("foot")
        if foot and "Preferred Foot" in df.columns:
            pf = df["Preferred Foot"].fillna("").str.lower()
            if foot == "left":
                mask &= pf.str.contains("left")
            elif foot == "right":
                mask &= pf.str.contains("right")
            elif foot == "either":
                mask &= pf.str.contains("either")
            notes.append(f"{foot}-footed")

        # international experience (has national-team caps)
        if spec.get("min_caps") and "Caps" in df.columns:
            mask &= pd.to_numeric(df["Caps"], errors="coerce").fillna(0) >= spec["min_caps"]
            notes.append("international (capped)")

        # free agents: empty/missing club
        if spec.get("free_agent") and "Club" in df.columns:
            club = df["Club"].fillna("").astype(str).str.strip()
            mask &= (club == "") | (club.str.lower().isin(["", "-", "none", "nan"]))
            notes.append("free agents only")

        pool = df[mask].copy()
        pool.attrs["filter_notes"] = notes
        pool.attrs["n_before"] = len(df)
        pool.attrs["n_after"] = len(pool)
        return pool


class ScoringAgent:
    """Rank a candidate pool by a weighted sum of normalised attributes."""

    def __init__(self, players: pd.DataFrame):
        # normalise each attribute against the FULL population (0-1),
        # so scores are comparable regardless of which pool we scored.
        self.min = {c: players[c].min() for c in ALL_ATTR_CODES if c in players}
        self.max = {c: players[c].max() for c in ALL_ATTR_CODES if c in players}

    def _norm(self, pool: pd.DataFrame, code: str) -> pd.Series:
        lo, hi = self.min.get(code, 1), self.max.get(code, 20)
        if hi == lo:
            return pd.Series(0.0, index=pool.index)
        return (pool[code] - lo) / (hi - lo)

    def run(self, pool: pd.DataFrame, spec: dict) -> pd.DataFrame:
        weights = spec.get("weights") or {}
        if not weights:
            # no preferences -> can't score; return pool unranked
            pool = pool.copy()
            pool["suitability"] = np.nan
            return pool

        # normalise weights to sum to 1
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items() if k in pool.columns}

        score = pd.Series(0.0, index=pool.index)
        for code, w in weights.items():
            score += w * self._norm(pool, code)

        pool = pool.copy()
        pool["suitability"] = (score * 100).round(1)   # 0-100 scale, readable

        # financial efficiency: suitability per €1M (bonus signal, section 6)
        val_m = (pool["value_mid"] / 1e6).replace(0, np.nan)
        pool["value_efficiency"] = (pool["suitability"] / val_m).round(1)

        top_n = spec.get("top_n", 10)
        ranked = pool.sort_values("suitability", ascending=False).head(top_n)
        ranked.attrs["weights_used"] = weights
        return ranked


def summarise_player(row: pd.Series, spec: dict) -> dict:
    """Compact dict for one shortlisted player (fed to the Reporting Agent)."""
    weights = spec.get("weights") or {}
    key_attrs = {
        ATTRIBUTES.get(code, code): int(row[code])
        for code in weights.keys() if code in row.index and pd.notna(row[code])
    }
    val = row.get("value_mid")
    return {
        "name": row["Name"],
        "age": int(row["Age"]) if pd.notna(row["Age"]) else None,
        "club": row.get("Club"),
        "positions": row.get("positions"),
        "value_eur": None if pd.isna(val) else int(val),
        "suitability": None if pd.isna(row.get("suitability")) else float(row["suitability"]),
        "key_attributes": key_attrs,
    }
