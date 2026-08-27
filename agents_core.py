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

        # position: keep rows whose role list intersects the requested codes.
        # If best_pos_only is set, match against the player's BEST position only
        # (their primary role) rather than any position they can fill.
        codes = spec.get("position_codes") or []
        if codes:
            codeset = set(codes)
            if spec.get("best_pos_only") and "best_pos_codes" in df.columns:
                mask &= df["best_pos_codes"].apply(
                    lambda roles: bool(codeset & set(roles)))
                notes.append(f"best position in {codes}")
            else:
                mask &= df["positions"].apply(
                    lambda roles: bool(codeset & set(roles)))
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

        # nationality filter: match first OR second nationality. Eligibility
        # (EU / community quotas) follows any passport a player holds, so
        # checking only 'Nat' would wrongly exclude dual nationals.
        nat_set = spec.get("nationality_set")
        if nat_set and "Nat" in df.columns:
            eligible = df["Nat"].isin(nat_set)
            if "nat2_code" in df.columns:
                eligible |= df["nat2_code"].isin(nat_set)
            mask &= eligible
            notes.append(f"nationality in set ({len(nat_set)})")

        # wage budget: a transfer fee is only half the cost of a signing, and a
        # club can often afford the fee but not the salary, so this filters on
        # what the player would actually cost to keep.
        if spec.get("max_salary") is not None and "salary_eur" in df.columns:
            sal = df["salary_eur"]
            mask &= sal.notna() & (sal <= spec["max_salary"])
            notes.append(f"salary <= {spec['max_salary']:,.0f}/yr")

        # playing-style archetype: keep only players clustered into that style
        if spec.get("style") and "style" in df.columns:
            mask &= df["style"] == spec["style"]
            notes.append(f"style: {spec['style']}")

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

        # free agents / out-of-contract players.
        # Both mean "signable for nothing", but they're stored differently: a
        # free agent has no club, while an expired contract still lists the old
        # club. Requiring both at once would always return nobody, so when the
        # request implies either we match on the union.
        want_free = spec.get("free_agent")
        want_expired = spec.get("expired_contract")
        club_col = df["Club"].fillna("").astype(str).str.strip() if "Club" in df.columns else None
        if want_free or want_expired:
            avail = pd.Series(False, index=df.index)
            # a "free agent" is anyone available on a free transfer: no club on
            # file OR an expired contract (the player's old club may still be
            # listed even though the deal has run out).
            if want_free:
                if club_col is not None:
                    avail |= club_col.str.lower().isin(["", "-", "none", "nan"])
                if "contract_status" in df.columns:
                    avail |= df["contract_status"] == "expired"
            if want_expired and "contract_status" in df.columns:
                avail |= df["contract_status"] == "expired"
            mask &= avail
            notes.append("free agents / expired contracts")
        elif spec.get("max_contract_months") is not None \
                and "contract_months_left" in df.columns:
            months = df["contract_months_left"]
            # "expiring soon" means still under contract but not for long — an
            # already-expired deal has negative months left, which would slip
            # through a plain "<= N" test, so require a positive remainder.
            mask &= months.notna() & (months > 0) \
                & (months <= spec["max_contract_months"])
            notes.append(f"contract <= {spec['max_contract_months']} months left")

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

    def score(self, pool: pd.DataFrame, weights: dict) -> pd.Series:
        """Suitability (0-100) for every row, using pre-normalised weights.

        Exposed separately from run() so the same yardstick can be applied to a
        club's existing players — that's what makes "better than what we already
        have" a fair comparison rather than two different scales.
        """
        if not weights or pool.empty:
            return pd.Series(dtype=float, index=pool.index)
        total = sum(weights.values())
        w = {k: v / total for k, v in weights.items() if k in pool.columns}
        score = pd.Series(0.0, index=pool.index)
        for code, weight in w.items():
            score += weight * self._norm(pool, code)
        return (score * 100).round(1)

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
        raw = (score * 100)
        # Attribute scores alone can float veterans in semi-pro sides — whose FM
        # ratings for a few key attributes (e.g. a 35-year-old's Marking) haven't
        # fully decayed — above genuinely better players. We blend the requested-
        # attribute score with the player's overall-ability percentile. When the
        # user named specific qualities we keep their criteria dominant (65/35);
        # when they only gave a position/category ("centre back", "free agent"),
        # overall ability leads (35/65) so the strongest real players surface
        # instead of past-it names in lower divisions.
        if "overall_ability" in pool.columns:
            oa_pct = pool["overall_ability"].rank(pct=True) * 100
            if spec.get("explicit_qualities"):
                pool["suitability"] = (0.65 * raw + 0.35 * oa_pct).round(1)
            else:
                pool["suitability"] = (0.35 * raw + 0.65 * oa_pct).round(1)
        else:
            pool["suitability"] = raw.round(1)

        # financial efficiency: suitability per €1M (bonus signal, section 6)
        val_m = (pool["value_mid"] / 1e6).replace(0, np.nan)
        pool["value_efficiency"] = (pool["suitability"] / val_m).round(1)

        top_n = spec.get("top_n", 10)
        # "best/top" -> rank by overall ability rather than the attribute-weighted
        # score. Overall ability already folds in age/decline, so it surfaces the
        # genuinely strongest current players; a pure attribute score can float
        # veterans in semi-pro sides whose FM ratings haven't fully decayed (e.g.
        # a 35-year-old with still-high Marking) above better active players.
        # We keep this to "best" queries where no explicit qualities were asked
        # for, so specific attribute requests ("centre back with good passing")
        # still rank on those attributes.
        if spec.get("rank_by_quality") and not spec.get("explicit_qualities") \
                and "overall_ability" in pool.columns:
            pool = pool.sort_values("overall_ability", ascending=False,
                                    na_position="last")
            pool["suitability"] = (pool["overall_ability"].rank(pct=True)
                                   * 100).round(1)
            ranked = pool.head(top_n)
            ranked.attrs["weights_used"] = weights
            return ranked

        # Suitability ties are common when few attributes are weighted (FM ratings
        # are 1-20 integers), and a tie would otherwise be broken by row order —
        # which buried strong players under semi-pro ones on equal marks. Ties are
        # settled on overall ability rather than market value: value tracks club
        # reputation and league, so using it would push cheap players from smaller
        # leagues down even when they are the better footballer — the exact
        # bargains a scout wants surfaced.
        tiebreak = ("overall_ability" if "overall_ability" in pool.columns
                    else "value_mid" if "value_mid" in pool.columns else None)
        if tiebreak:
            pool = pool.sort_values(["suitability", tiebreak],
                                    ascending=[False, False], na_position="last")
        else:
            pool = pool.sort_values("suitability", ascending=False)
        ranked = pool.head(top_n)
        ranked.attrs["weights_used"] = weights
        return ranked


def summarise_player(row: pd.Series, spec: dict) -> dict:
    """Compact dict for one shortlisted player (fed to the Reporting Agent)."""
    weights = spec.get("weights") or {}
    # attributes are shown on a 0-100 scale (x5 the native 1-20) so the report
    # and UI don't expose raw game ratings; the underlying data stays 1-20.
    key_attrs = {
        ATTRIBUTES.get(code, code): int(round(row[code] * 5))
        for code in weights.keys() if code in row.index and pd.notna(row[code])
    }
    val = row.get("value_mid")
    return {
        "name": row["Name"],
        "age": int(row["Age"]) if pd.notna(row["Age"]) else None,
        "club": row.get("Club"),
        "positions": row.get("positions"),
        "style": (None if pd.isna(row.get("style")) else row.get("style")),
        "value_eur": None if pd.isna(val) else int(val),
        "predicted_value": (None if pd.isna(row.get("predicted_value"))
                            else int(row.get("predicted_value"))),
        "value_range": (
            None if pd.isna(row.get("value_min")) or pd.isna(row.get("value_max"))
            or row.get("value_min") == row.get("value_max")
            else (int(row["value_min"]), int(row["value_max"]))),
        "contract": row.get("contract_status"),
        "contract_expires": (None if pd.isna(row.get("Expires"))
                             else str(row.get("Expires"))),
        "suitability": None if pd.isna(row.get("suitability")) else float(row["suitability"]),
        "key_attributes": key_attrs,
    }
