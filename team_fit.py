"""
team_fit.py — team-need analysis for squad-aware suitability.

When the user scouts for a specific club ("scout for AEK, need a left back"),
this looks at that club's existing players in the target position, finds where
the squad is WEAKEST relative to the league, and boosts those attribute weights.
Result: the shortlist favours players who fill the actual gap, not just the best
generic player.
"""
from __future__ import annotations
import pandas as pd
import unicodedata
from data import ALL_ATTR_CODES, ATTRIBUTES


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s).lower())
    return "".join(c for c in s if not unicodedata.combining(c))


class TeamFitAnalyzer:
    def __init__(self, players: pd.DataFrame):
        self.players = players
        self.attrs = [c for c in ALL_ATTR_CODES if c in players.columns]
        # league-wide mean per attribute (baseline for "strong/weak")
        self._league_mean = players[self.attrs].mean()

    def find_club(self, name: str) -> str | None:
        """Resolve a user-typed club name to an actual club in the data.

        On ambiguous matches (e.g. 'AEK' -> AEK / AEK Larnakas), prefer the club
        with the larger squad in the data (usually the more prominent one).
        """
        target = _norm(name)
        clubs = self.players["Club"].dropna().unique()
        # exact match first
        for c in clubs:
            if _norm(c) == target:
                return c
        # collect all contains-matches, pick the one with the biggest squad
        matches = []
        for c in clubs:
            nc = _norm(c)
            if target in nc or nc in target:
                matches.append(c)
        if matches:
            sizes = {c: (self.players["Club"] == c).sum() for c in matches}
            return max(matches, key=lambda c: sizes[c])
        return None

    # position groups used for squad-need analysis
    _NEED_GROUPS = {
        "GK": ["GK"],
        "Centre-back": ["DC"],
        "Full-back": ["DL", "DR", "WBL", "WBR"],
        "Defensive mid": ["DM"],
        "Central mid": ["MC"],
        "Attacking mid": ["AMC"],
        "Winger": ["AML", "AMR", "ML", "MR"],
        "Striker": ["STC"],
    }

    def squad_needs(self, club: str, top: int = 3) -> dict:
        """Analyse the whole squad and rank positions by need.

        Need combines two signals: quality (how far the club's players in a
        position sit below the league mean) and depth (how few players it has
        there). Returns {'club', 'needs': [{position, codes, need, depth,
        quality_gap, note}], 'summary'}.
        """
        squad = self.players[self.players["Club"] == club]
        if len(squad) == 0:
            return {"club": club, "needs": [], "summary": f"No squad data for {club}."}

        gk_codes = {"Aer", "Cmd", "Com", "Ecc", "Han", "Kic", "1v1", "Pun", "Ref", "TRO", "Thr"}
        results = []
        for label, codes in self._NEED_GROUPS.items():
            cset = set(codes)
            in_pos = squad[squad["positions"].apply(lambda L: bool(cset & set(L)))]
            depth = len(in_pos)
            is_gk = "GK" in cset
            relevant = [c for c in self.attrs if (c in gk_codes) == is_gk]
            if depth > 0:
                squad_mean = in_pos[relevant].mean().mean()
                league_mean = self._league_mean[relevant].mean()
                quality_gap = float(league_mean - squad_mean)  # positive = below league
            else:
                quality_gap = 2.0  # no player at all = big gap

            # depth score: 0 players = 1.0, 1 = 0.6, 2 = 0.3, 3+ = 0.1
            depth_score = {0: 1.0, 1: 0.6, 2: 0.3}.get(depth, 0.1)
            # normalise quality gap to ~0-1 (gaps beyond +2 are severe)
            quality_score = max(0.0, min(1.0, quality_gap / 2.0))
            need = round(0.6 * quality_score + 0.4 * depth_score, 3)

            results.append({
                "position": label, "codes": codes, "depth": depth,
                "quality_gap": round(quality_gap, 2), "need": need,
            })

        results.sort(key=lambda r: -r["need"])
        top_needs = results[:top]

        # readable summary
        lines = []
        for r in top_needs:
            sev = "critical" if r["need"] > 0.6 else "notable" if r["need"] > 0.4 else "minor"
            depth_txt = ("no specialist" if r["depth"] == 0
                         else f"{r['depth']} player" + ("s" if r["depth"] != 1 else ""))
            lines.append(f"{r['position']} ({sev}): {depth_txt}, "
                         + ("below league level" if r["quality_gap"] > 0.3 else "around league level"))
        summary = f"{club} squad needs — " + "; ".join(lines)
        return {"club": club, "needs": top_needs, "summary": summary}

    def squad_gaps(self, club: str, position_codes: list[str]) -> dict:
        """Return the club's current players in position + which attributes lag.

        Returns {'players': [...], 'weak_attrs': {code: deficit}, 'note': str}.
        """
        codeset = set(position_codes)
        squad = self.players[
            (self.players["Club"] == club)
            & (self.players["positions"].apply(lambda L: bool(codeset & set(L))))
        ]
        if len(squad) == 0:
            return {"players": [], "weak_attrs": {}, "note": "no current players in that position"}

        # only consider attributes relevant to the position type
        is_gk = "GK" in codeset
        gk_codes = {"Aer", "Cmd", "Com", "Ecc", "Han", "Kic", "1v1", "Pun", "Ref", "TRO", "Thr"}
        if is_gk:
            relevant = [c for c in self.attrs if c in gk_codes]
        else:
            relevant = [c for c in self.attrs if c not in gk_codes]

        squad_mean = squad[relevant].mean()
        deficit = (self._league_mean[relevant] - squad_mean)
        weak = deficit[deficit > 0.5].sort_values(ascending=False)
        weak_attrs = {code: round(float(v), 1) for code, v in weak.items()}

        return {
            "players": squad["Name"].tolist(),
            "squad_size": len(squad),
            "weak_attrs": weak_attrs,
            "note": f"{len(squad)} current player(s) in position",
        }

    def adjust_weights(self, base_weights: dict, gaps: dict,
                       boost: float = 0.5) -> dict:
        """Blend the user's weights with the squad's weak spots.

        Attributes where the squad is weak get boosted, so players who fill the
        gap rank higher. Keeps the user's original priorities but tilts toward need.
        """
        if not gaps.get("weak_attrs"):
            return dict(base_weights)

        adjusted = dict(base_weights)
        # take the top few weak attributes and add/boost them
        top_weak = list(gaps["weak_attrs"].items())[:5]
        max_def = max(v for _, v in top_weak) or 1.0
        for code, deficit in top_weak:
            extra = boost * (deficit / max_def)  # scale by how weak it is
            adjusted[code] = adjusted.get(code, 0.0) + extra
        return adjusted

    def summary(self, club: str, gaps: dict) -> str:
        """Human-readable note about the squad's needs."""
        if not gaps.get("weak_attrs"):
            if not gaps.get("players"):
                return f"{club} has no players in this position — any signing fills a gap."
            return f"{club}'s players here are at/above league level; ranking by your priorities."
        weak_names = ", ".join(ATTRIBUTES.get(c, c) for c, _ in list(gaps["weak_attrs"].items())[:3])
        return (f"{club} has {gaps['squad_size']} player(s) here; the squad is weakest in "
                f"{weak_names}. Boosting those so candidates who fill the gap rank higher.")
