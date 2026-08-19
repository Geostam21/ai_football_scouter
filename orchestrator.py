"""
orchestrator.py — wire the four agents into one scouting pipeline.

    request (text)
        -> RequirementsAgent (LLM)  : text -> spec
        -> CandidateAgent    (code) : hard filters -> pool
        -> ScoringAgent      (code) : weighted rank -> shortlist
        -> ReportingAgent    (LLM)  : shortlist -> briefing
"""
from __future__ import annotations
import re
import pandas as pd
from data import load_players, ATTRIBUTES
from agents_core import CandidateAgent, ScoringAgent, summarise_player
from agents_llm import RequirementsAgent, ReportingAgent
from agent_dashboard import DashboardAgent
from team_fit import TeamFitAnalyzer


class ScoutingPipeline:
    def __init__(self, players: pd.DataFrame | None = None):
        self.players = players if players is not None else load_players()
        # add playing-style archetypes (Poacher, Target Man, ...) once up front
        from roles import RoleClusterer
        self.roles = RoleClusterer(self.players)
        self.players = self.roles.attach(self.players)
        self.requirements = RequirementsAgent()
        self.candidates = CandidateAgent(self.players)
        self.scoring = ScoringAgent(self.players)
        self.reporting = ReportingAgent()
        self.dashboard = DashboardAgent(self.players)
        self.team_fit = TeamFitAnalyzer(self.players)

    def player_fit_query(self, player_name: str, club_name: str) -> dict:
        """Answer 'how well would <player> fit at <club>?' for a named pair."""
        # accent-insensitive: users type "Vinicius", the data holds "Vinícius"
        import unicodedata

        def _fold(s):
            s = unicodedata.normalize("NFKD", str(s).lower())
            return "".join(c for c in s if not unicodedata.combining(c))

        target = _fold(player_name)
        folded = self.players["Name"].map(_fold)
        # Match on substring rather than equality: a short surname is often both
        # an exact name for an obscure player and part of the famous one's full
        # name ("Vinicius" vs "Vinícius Júnior"). Sorting by value then picks the
        # player the user almost certainly meant.
        rows = self.players[folded.str.contains(re.escape(target), na=False)]
        if rows.empty:
            return {"error": f"Couldn't find a player called '{player_name}'."}
        club = self.team_fit.find_club(club_name)
        if not club:
            return {"error": f"Couldn't find a club called '{club_name}'."}
        # if the name is ambiguous, take the most prominent player
        row = rows.sort_values("value_mid", ascending=False).iloc[0]
        fit = self.team_fit.player_fit(row, club)
        fit["explanation"] = _explain_fit(fit)
        return fit

    def run(self, request: str, verbose: bool = False) -> dict:
        # 1. understand the request
        spec = self.requirements.run(request)

        # 1b. team-fit: if scouting for a club, tilt weights toward squad gaps
        team_note = None
        club_name = spec.get("team_fit_club")
        if club_name:
            club = self.team_fit.find_club(club_name)
            if club:
                gaps = self.team_fit.squad_gaps(club, spec["position_codes"])
                spec["weights"] = self.team_fit.adjust_weights(spec["weights"], gaps)
                team_note = self.team_fit.summary(club, gaps)
            else:
                team_note = f"Couldn't find '{club_name}' in the loaded data; ranking normally."

        if verbose:
            print("SPEC:", {k: v for k, v in spec.items() if k != "_request"})
            if team_note:
                print("TEAM-FIT:", team_note)

        # 2. hard filters
        pool = self.candidates.run(spec)
        if verbose:
            print(f"POOL: {pool.attrs['n_before']} -> {pool.attrs['n_after']}"
                  f" ({', '.join(pool.attrs['filter_notes'])})")

        # 3. weighted ranking
        upgrade_info = None
        if club_name and spec.get("upgrade_only"):
            club = self.team_fit.find_club(club_name)
            if club:
                upgrade_info = self.team_fit.upgrade_benchmark(
                    club, spec["position_codes"], self.scoring, spec["weights"],
                    best_pos=spec.get("best_pos_only", False))
                bar = upgrade_info.get("threshold")
                if bar is not None and not pool.empty:
                    # drop the club's own players, then keep only genuine upgrades
                    pool = pool[pool["Club"] != club]
                    pool = pool[self.scoring.score(pool, spec["weights"]) > bar]
                    pool.attrs["n_after"] = len(pool)
                team_note = upgrade_info["note"]

        ranked = self.scoring.run(pool, spec)
        shortlist = [summarise_player(r, spec) for _, r in ranked.iterrows()]

        # 4. explanation
        report = self.reporting.run(request, shortlist)

        return {
            "spec": spec,
            "pool_size": pool.attrs.get("n_after", 0),
            "shortlist": shortlist,
            "ranked_df": ranked,
            "report": report,
            "team_note": team_note,
            "upgrade": upgrade_info,
        }

    def build_dashboards(self, result: dict) -> list[dict]:
        """Build a full profile dashboard for every player in the shortlist."""
        return [self.dashboard.build(row)
                for _, row in result["ranked_df"].iterrows()]


def readable_spec(spec: dict) -> str:
    """Human-readable one-liner of a spec (for UI / debugging)."""
    parts = []
    if spec["position_codes"]:
        parts.append("pos " + "/".join(spec["position_codes"]))
    if spec.get("max_age"):
        parts.append(f"age≤{spec['max_age']}")
    if spec.get("max_value"):
        parts.append(f"≤€{spec['max_value']/1e6:.0f}M")
    if spec.get("league_substrings"):
        parts.append("selected leagues")
    if spec.get("nationality_set"):
        n = spec["nationality_set"]
        if len(n) == 1:
            parts.append(list(n)[0])
        elif len(n) == 27:
            parts.append("EU nationals")
        elif len(n) > 40:
            parts.append("community/European")
        else:
            parts.append("nat-filtered")
    if spec.get("free_agent"):
        parts.append("free agents")
    if spec.get("min_height"):
        parts.append(f"≥{spec['min_height']}cm")
    if spec.get("max_height"):
        parts.append(f"≤{spec['max_height']}cm")
    if spec.get("foot"):
        parts.append(f"{spec['foot']}-footed")
    if spec.get("min_caps"):
        parts.append("international")
    w = ", ".join(f"{ATTRIBUTES.get(c, c)}×{v:.1f}" for c, v in spec["weights"].items())
    parts.append("weights: " + w)
    return " | ".join(parts)


def format_value(eur) -> str:
    """Format a euro amount as a readable string: 11250000 -> '€11.25M'."""
    if eur is None or (isinstance(eur, float) and eur != eur):  # None or NaN
        return "n/a"
    eur = float(eur)
    if eur >= 1e6:
        return f"€{eur/1e6:.2f}M".replace(".00M", "M")
    if eur >= 1e3:
        return f"€{eur/1e3:.0f}K"
    return f"€{eur:.0f}"


def shortlist_table(result: dict) -> "pd.DataFrame":
    """Readable shortlist DataFrame with a formatted value column."""
    import pandas as pd
    rows = []
    for p in result["shortlist"]:
        rows.append({
            "Player": p["name"],
            "Age": p["age"],
            "Club": p.get("club"),
            "Value": format_value(p["value_eur"]),
            "Suitability": p["suitability"],
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pipe = ScoutingPipeline()
    req = "left winger under 25 with a budget of 15M, very fast, good dribbling and decent finishing"
    print("REQUEST:", req, "\n")
    result = pipe.run(req, verbose=True)
    print("\n--- REPORT ---")
    print(result["report"])


def _explain_fit(f: dict) -> str:
    """Turn the fit axes into a plain-language verdict."""
    if f.get("error"):
        return f["error"]
    name, club = f["player"], f["club"]
    parts = []
    # sporting verdict
    if f["upgrade"] >= 85:
        parts.append(f"{name} would be a clear upgrade on {club}'s current options "
                     f"({f['player_overall']} overall vs {f['best_incumbent']} for their best).")
    elif f["upgrade"] >= 55:
        parts.append(f"{name} would modestly improve {club} "
                     f"({f['player_overall']} vs {f['best_incumbent']}).")
    else:
        parts.append(f"{name} would not improve {club} — they already have "
                     f"{f['best_incumbent']} overall in that position.")
    # need
    if f["need"] >= 80:
        parts.append(f"The position is a genuine hole ({f['depth']} senior option(s)).")
    elif f["need"] >= 60:
        parts.append(f"They are thin there ({f['depth']} senior options).")
    else:
        parts.append(f"They are already well stocked there ({f['depth']} senior options).")
    # gap fit
    if f["weak_attrs"]:
        if f["gap_fit"] >= 70:
            parts.append("He is strong in exactly what the squad lacks ("
                         + ", ".join(f.get("weak_attr_names", f["weak_attrs"])[:3]) + ").")
        elif f["gap_fit"] <= 30:
            parts.append("He does not address their weak spots ("
                         + ", ".join(f.get("weak_attr_names", f["weak_attrs"])[:3]) + ").")
    # feasibility
    if f["feasible"] >= 80:
        parts.append("Financially this is realistic for them.")
    elif f["feasible"] >= 30:
        parts.append("The fee/wages would stretch them.")
    else:
        parts.append("Financially it is out of reach — fee and wages sit far above "
                     "anything on their books.")
    return " ".join(parts)


# "how would X fit at Y" / "θα ταιριαζε ο X στην Y" -> (player, club)
_FIT_PATTERNS = [
    r"(?:how (?:well )?would|would|does)\s+(.+?)\s+(?:fit|suit|work)\s+"
    r"(?:in|at|for|with)?\s*(.+?)[\?\.]*$",
    r"(?:fit|suitability)\s+of\s+(.+?)\s+(?:in|at|for)\s+(.+?)[\?\.]*$",
    r"(?:ταιριαζει|ταιριαζε|θα ταιριαζε|καναι|κανει)\s+(?:ο|η|τον|την)?\s*(.+?)\s+"
    r"(?:στην|στον|στη|για την|για τον)\s+(.+?)[\?\.]*$",
]


def detect_fit_query(request: str):
    """Return (player, club) if the request is a player-to-club fit question."""
    text = request.strip()
    low = text.lower()
    if not any(k in low for k in ["fit", "suit", "ταιριαζ", "tairiaz", "καναι", "work at",
                                  "work in", "work for"]):
        return None
    for pat in _FIT_PATTERNS:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            player = m.group(1).strip(" ,")
            club = m.group(2).strip(" ,?")
            # strip filler that survives the pattern
            for junk in ("well ", "the ", "a ", "at ", "in ", "for "):
                if player.lower().startswith(junk):
                    player = player[len(junk):]
                if club.lower().startswith(junk):
                    club = club[len(junk):]
            if len(player) >= 3 and len(club) >= 2:
                return player, club
    return None
