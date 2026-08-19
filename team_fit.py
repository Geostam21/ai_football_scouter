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


# Greek -> Latin, so a club typed in Greek ("ΑΕΚ", "Ολυμπιακος") still resolves
# against the Latin-script club names used in the dataset.
_GREEK_TO_LATIN = {
    "α": "a", "β": "v", "γ": "g", "δ": "d", "ε": "e", "ζ": "z", "η": "i",
    "θ": "th", "ι": "i", "κ": "k", "λ": "l", "μ": "m", "ν": "n", "ξ": "x",
    "ο": "o", "π": "p", "ρ": "r", "σ": "s", "ς": "s", "τ": "t", "υ": "y",
    "φ": "f", "χ": "ch", "ψ": "ps", "ω": "o",
}


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s).lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    if any(c in _GREEK_TO_LATIN for c in s):
        s = "".join(_GREEK_TO_LATIN.get(c, c) for c in s)
    return s


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
                league_mean = self._baseline(club)[relevant].mean()
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

    def incumbents(self, club: str, position_codes: list[str],
                   best_pos: bool = False) -> pd.DataFrame:
        """The club's current players who cover the requested position."""
        codeset = set(position_codes)
        col = "best_pos_codes" if (best_pos and "best_pos_codes" in self.players.columns) \
            else "positions"
        return self.players[
            (self.players["Club"] == club)
            & (self.players[col].apply(lambda L: bool(codeset & set(L))))
        ]

    def upgrade_benchmark(self, club: str, position_codes: list[str],
                          scorer, weights: dict, best_pos: bool = False) -> dict:
        """Score the club's current players in a position on the user's criteria.

        Returns the bar a signing has to clear to actually be an upgrade, plus
        the incumbents themselves so the report can name who'd be displaced.
        Scoring the squad with the *same* weights as the search is what makes
        "better than what we have" meaningful — a generic quality rating would
        ignore what the user actually asked for.
        """
        squad = self.incumbents(club, position_codes, best_pos=best_pos)
        if squad.empty or not weights:
            return {"club": club, "threshold": None, "incumbents": [],
                    "note": f"{club} has no current player in that position — "
                            f"any signing fills a gap."}
        scores = scorer.score(squad, weights)
        ranked = sorted(
            ({"name": n, "age": (int(a) if pd.notna(a) else None), "score": float(s)}
             for n, a, s in zip(squad["Name"], squad["Age"], scores)),
            key=lambda r: -r["score"],
        )
        best = ranked[0]
        return {
            "club": club,
            "threshold": best["score"],
            "incumbents": ranked[:5],
            "note": (f"{club}'s best current option there is {best['name']} "
                     f"({best['score']:.1f}/100 on your criteria). "
                     f"Showing only players who beat that."),
        }

    def _baseline(self, club: str) -> pd.Series:
        """Attribute baseline to judge a squad against: its own league.

        The global mean across the whole dataset is useless as a yardstick here
        — it includes semi-pro and youth players, so every top-flight club looks
        strong everywhere and no real gap is ever found. Comparing a club to the
        division it actually competes in is what surfaces genuine weaknesses.
        """
        league = self.players.loc[self.players["Club"] == club, "Based"]
        if len(league):
            peers = self.players[self.players["Based"] == league.iloc[0]]
            if len(peers) >= 50:
                return peers[self.attrs].mean()
        return self._league_mean

    def club_profile(self, club: str) -> dict:
        """Financial and sporting level of a club, inferred from its own squad."""
        squad = self.players[self.players["Club"] == club]
        if squad.empty:
            return {}
        top10 = squad.nlargest(10, "overall_ability")["overall_ability"].mean() \
            if "overall_ability" in squad.columns else float("nan")
        return {
            "squad_size": len(squad),
            "top_value": float(squad["value_mid"].max() or 0),
            "median_value": float(squad["value_mid"].median() or 0),
            "top_wage": float(squad["salary_eur"].max() or 0),
            "level": float(top10),
            "league": (squad["Based"].mode().iloc[0]
                       if squad["Based"].notna().any() else None),
        }

    def player_fit(self, player_row: pd.Series, club: str) -> dict:
        """How well a specific player would fit a specific club.

        Deliberately split into separate, explainable axes rather than one
        opaque number, because "fit" is really two different questions that can
        disagree: whether he would improve the team on the pitch, and whether
        the club could realistically sign him. Haaland improves everyone, but
        that is not useful advice for a mid-table side.

        Axes:
          need     - does the club lack depth/quality in his position?
          upgrade  - is he better than who they already have there?
          gap_fit  - is he strong exactly where the squad is weak?
          feasible - is the fee/wage within reach of this club?
        """
        prof = self.club_profile(club)
        if not prof:
            return {"club": club, "error": f"No squad data for {club}."}

        pos = list(player_row.get("best_pos_codes")
                   or player_row.get("positions") or [])
        squad = self.players[self.players["Club"] == club]

        # --- 1. positional need (depth + quality of incumbents) ---
        cset = set(pos)
        # Count only players whose BEST position is this one. Counting anyone who
        # can fill in there makes every squad look deep — a winger who can cover
        # striker is not a striker — and the need score collapses to a constant.
        col = "best_pos_codes" if "best_pos_codes" in squad.columns else "positions"
        incumbents = squad[squad[col].apply(lambda L: bool(cset & set(L)))]
        # FM squads carry the whole academy, so raw counts overstate depth badly
        # (AEK look like they have 7 strikers; 5 are teenagers). Depth is judged
        # on first-team-calibre players only.
        senior = incumbents[
            (incumbents["Age"] >= 19)
            & (incumbents["overall_ability"] >= incumbents["overall_ability"].max() - 3.0)
        ] if len(incumbents) else incumbents
        depth = len(senior) if len(senior) else len(incumbents)
        need = {0: 100, 1: 85, 2: 65, 3: 45}.get(depth, 30)

        # --- 2. upgrade over incumbents (on overall ability) ---
        p_ovr = float(player_row.get("overall_ability") or 0)
        pool_inc = senior if len(senior) else incumbents
        if len(pool_inc) and "overall_ability" in pool_inc.columns:
            best_inc = float(pool_inc["overall_ability"].max())
            # +2.0 overall over the best incumbent is a decisive upgrade
            upgrade = max(0.0, min(100.0, 50 + (p_ovr - best_inc) * 25))
        else:
            best_inc, upgrade = None, 100.0

        # --- 3. does he fix what the squad is actually weak at? ---
        gaps = self.squad_gaps(club, pos) if pos else {"weak_attrs": {}}
        weak = list(gaps.get("weak_attrs", {}).items())[:5]
        if weak and depth:
            # He "fills" a weak spot only if he is clearly better there than the
            # players already in that position — being above league average is
            # not enough, since the incumbents may already be too. This is what
            # separates a genuine fix from a sideways move.
            inc_mean = incumbents[[c for c, _ in weak]].mean()
            hits = [1.0 for code, _ in weak
                    if code in player_row.index and pd.notna(player_row[code])
                    and player_row[code] >= inc_mean.get(code, 0) + 1.0]
            gap_fit = round(100 * len(hits) / len(weak))
        elif weak:
            gap_fit = 100  # no incumbents -> anyone is an improvement
        else:
            gap_fit = 50

        # --- 4. can the club afford him? ---
        val = float(player_row.get("value_mid") or 0)
        wage = float(player_row.get("salary_eur") or 0)
        ceil_v = prof["top_value"] * 1.2 or 1
        ceil_w = prof["top_wage"] * 1.2 or 1
        fee_ok = 100 if val <= ceil_v else max(0, 100 - (val / ceil_v - 1) * 100)
        wage_ok = 100 if wage <= ceil_w else max(0, 100 - (wage / ceil_w - 1) * 100)
        feasible = round(min(fee_ok, wage_ok))

        # Upgrade dominates: a clearly better player is worth signing even if he
        # doesn't happen to patch a specific weak trait. Need and gap-fit refine
        # the picture but shouldn't override raw quality — that mistake made a
        # like-for-like star (Bruno Guimaraes) score below a mediocre incumbent.
        sporting = round(0.55 * upgrade + 0.30 * need + 0.15 * gap_fit)
        return {
            "club": club, "player": player_row.get("Name"),
            "positions": pos, "depth": depth,
            "need": round(need), "upgrade": round(upgrade),
            "gap_fit": gap_fit, "feasible": feasible,
            "sporting_fit": sporting,
            "best_incumbent": (None if best_inc is None else round(best_inc, 2)),
            "player_overall": round(p_ovr, 2),
            "weak_attrs": [c for c, _ in weak],
            "weak_attr_names": [ATTRIBUTES.get(c, c) for c, _ in weak],
            "club_level": round(prof["level"], 2),
        }
    def _relevant_attrs(self, position_codes) -> list[str]:
        """Attributes that actually matter for a position type.

        Averaging over ALL attributes makes gap analysis meaningless: for a
        central midfielder it would flag Jumping Reach or Long Throws as
        'weaknesses', which no scout would act on. Restricting to the traits the
        role is judged on is what makes the weak-spot readout trustworthy.
        """
        codeset = set(position_codes)
        gk = {"Aer", "Cmd", "Com", "Ecc", "Han", "Kic", "1v1", "Pun", "Ref",
              "TRO", "Thr"}
        # mental attrs matter everywhere
        mental = ["Dec", "Ant", "Cmp", "Cnt", "Wor", "Tea", "Vis", "Pos"]
        defend = ["Tck", "Mar", "Pos", "Str", "Hea", "Bra", "Agg"]
        create = ["Pas", "Vis", "Dri", "Tec", "Fir", "Fla", "OtB", "Cro"]
        attack = ["Fin", "OtB", "Cmp", "Fir", "Dri", "Hea", "Ant"]
        pace = ["Pac", "Acc", "Agi", "Bal", "Sta"]

        if "GK" in codeset:
            want = list(gk) + ["Cmp", "Cnt", "Dec", "Ant", "Pos"]
        elif codeset & {"DC"}:
            want = defend + mental + ["Jum", "Pac", "Acc"]
        elif codeset & {"DR", "DL", "WBR", "WBL"}:
            want = ["Tck", "Mar", "Pos"] + pace + ["Cro", "Dri", "Sta", "Wor"] + mental
        elif codeset & {"DM"}:
            want = ["Tck", "Mar", "Pos"] + create + mental + ["Str"]
        elif codeset & {"MC", "MR", "ML"}:
            want = create + defend[:3] + mental + pace[:3]
        elif codeset & {"AMC", "AMR", "AML"}:
            want = create + attack + pace + ["Cmp"]
        elif codeset & {"STC", "ST"}:
            want = attack + pace + ["Str", "Jum", "Fin"]
        else:
            want = create + mental
        # keep only attrs present, deduped, order-preserving
        seen, out = set(), []
        for c in want:
            if c in self.attrs and c not in seen:
                seen.add(c)
                out.append(c)
        return out

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
        relevant = self._relevant_attrs(codeset)
        if not relevant:
            is_gk = "GK" in codeset
            gk_codes = {"Aer", "Cmd", "Com", "Ecc", "Han", "Kic", "1v1", "Pun",
                        "Ref", "TRO", "Thr"}
            relevant = [c for c in self.attrs
                        if (c in gk_codes) == is_gk]

        squad_mean = squad[relevant].mean()
        baseline = self._baseline(club)[relevant]
        deficit = (baseline - squad_mean)
        # First choice: attributes genuinely below the league. If the squad is
        # strong everywhere (deficit never clears the bar), fall back to its own
        # relatively weakest traits so "fills weak spots" still says something
        # meaningful instead of going blank for good teams.
        weak = deficit[deficit > 0.5].sort_values(ascending=False)
        if weak.empty:
            weak = deficit.sort_values(ascending=False).head(3)
            weak = weak[weak > -1.0]  # ignore areas where they're clearly strong
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
