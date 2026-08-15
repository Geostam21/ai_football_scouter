"""
orchestrator.py — wire the four agents into one scouting pipeline.

    request (text)
        -> RequirementsAgent (LLM)  : text -> spec
        -> CandidateAgent    (code) : hard filters -> pool
        -> ScoringAgent      (code) : weighted rank -> shortlist
        -> ReportingAgent    (LLM)  : shortlist -> briefing
"""
from __future__ import annotations
import pandas as pd
from data import load_players, ATTRIBUTES
from agents_core import CandidateAgent, ScoringAgent, summarise_player
from agents_llm import RequirementsAgent, ReportingAgent
from agent_dashboard import DashboardAgent
from team_fit import TeamFitAnalyzer


class ScoutingPipeline:
    def __init__(self, players: pd.DataFrame | None = None):
        self.players = players if players is not None else load_players()
        self.requirements = RequirementsAgent()
        self.candidates = CandidateAgent(self.players)
        self.scoring = ScoringAgent(self.players)
        self.reporting = ReportingAgent()
        self.dashboard = DashboardAgent(self.players)
        self.team_fit = TeamFitAnalyzer(self.players)

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
