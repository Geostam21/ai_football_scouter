"""
agent_dashboard.py — the Dashboard Agent (5th agent).

Takes one player + the full population and builds a complete profile:
attributes grouped by category, percentile vs positional peers, physical data,
value + efficiency, and an LLM-written strengths/weaknesses summary.

The heavy lifting (percentiles, grouping) is deterministic; only the prose
summary uses the LLM.
"""
from __future__ import annotations
import pandas as pd
from data import ATTRIBUTES, ALL_ATTR_CODES
from llm import call_llm

# Attributes are stored on FM's native 1-20 scale but shown to the user on a
# more intuitive 0-100 scale (x5), so a top rating reads as ~100 rather than 20
# and the numbers don't look like a raw game export. This is display-only —
# the models and filters keep using the underlying 1-20 values.
ATTR_DISPLAY_SCALE = 5


def scale_attr(v):
    """Convert a raw 1-20 attribute to the displayed 0-100 scale."""
    if v is None or (isinstance(v, float) and v != v):
        return v
    return int(round(v * ATTR_DISPLAY_SCALE))


# attribute categories for a clean profile layout
CATEGORIES = {
    "Technical": ["Cor", "Cro", "Dri", "Fin", "Fir", "Fre", "Hea", "Lon",
                  "L Th", "Mar", "Pas", "Pen", "Tck", "Tec"],
    "Mental": ["Agg", "Ant", "Bra", "Cmp", "Cnt", "Dec", "Det", "Fla",
               "Ldr", "OtB", "Pos", "Tea", "Vis", "Wor"],
    "Physical": ["Acc", "Agi", "Bal", "Jum", "Pac", "Sta", "Str"],
    "Goalkeeping": ["Aer", "Cmd", "Com", "Ecc", "Han", "Kic", "1v1",
                    "Pun", "Ref", "TRO", "Thr"],
}


class DashboardAgent:
    def __init__(self, players: pd.DataFrame):
        self.players = players

    def _peers(self, player_row: pd.Series) -> pd.DataFrame:
        """Players sharing at least one position code (for percentile comparison)."""
        codes = set(player_row["positions"])
        mask = self.players["positions"].apply(lambda L: bool(codes & set(L)))
        return self.players[mask]

    def build(self, player_row: pd.Series, is_gk: bool | None = None,
              with_summary: bool = True) -> dict:
        peers = self._peers(player_row)
        n_peers = len(peers)

        if is_gk is None:
            is_gk = "GK" in player_row["positions"]

        # grouped attributes with percentile vs peers
        groups = {}
        for cat, codes in CATEGORIES.items():
            if cat == "Goalkeeping" and not is_gk:
                continue
            if cat == "Physical" and False:
                pass
            rows = []
            for code in codes:
                if code not in player_row.index or pd.isna(player_row[code]):
                    continue
                val = int(player_row[code])
                # percentile: % of peers this player is >= to
                pct = int((peers[code] <= val).mean() * 100) if n_peers else None
                rows.append({
                    "attr": ATTRIBUTES.get(code, code),
                    "value": scale_attr(val),
                    "percentile": pct,
                })
            # sort strongest first
            rows.sort(key=lambda r: r["value"], reverse=True)
            if rows:
                groups[cat] = rows

        # overall standout attributes (top by value)
        relevant = [c for cat, cs in CATEGORIES.items()
                    for c in cs if (cat != "Goalkeeping") or is_gk]
        scored = []
        for c in relevant:
            if c in player_row.index and pd.notna(player_row[c]):
                val = int(player_row[c])
                pct = int((peers[c] <= val).mean() * 100) if n_peers else 50
                scored.append((ATTRIBUTES.get(c, c), val, pct))

        # strengths: highest percentile (best vs peers), then highest raw value
        strengths_ranked = sorted(scored, key=lambda t: (t[2], t[1]), reverse=True)
        top_attrs = [(a, scale_attr(v)) for a, v, p in strengths_ranked[:6]]

        # weaknesses: lowest percentile AND genuinely low (below peer median).
        # Only flag attributes where the player is in the bottom third of peers.
        weak_candidates = [t for t in scored if t[2] <= 33]
        weak_ranked = sorted(weak_candidates, key=lambda t: (t[2], t[1]))
        weaknesses = [(a, scale_attr(v), p) for a, v, p in weak_ranked[:5]]

        val = player_row.get("value_mid")
        profile = {
            "name": player_row["Name"],
            "age": int(player_row["Age"]) if pd.notna(player_row["Age"]) else None,
            "club": player_row.get("Club"),
            "nationality": player_row.get("Nat"),
            "contract_status": player_row.get("contract_status"),
            "contract_expires": (None if pd.isna(player_row.get("Expires"))
                                 else str(player_row.get("Expires"))),
            "positions": player_row["positions"],
            "height_cm": None if pd.isna(player_row.get("height_cm")) else int(player_row["height_cm"]),
            "weight_kg": None if pd.isna(player_row.get("weight_kg")) else int(player_row["weight_kg"]),
            "foot": player_row.get("foot"),
            "value_eur": None if pd.isna(val) else int(val),
            "tagline": player_row.get("Media Description"),
            "attribute_groups": groups,
            "top_attributes": top_attrs,
            "weaknesses": weaknesses,
            "n_peers": n_peers,
            # raw code->value map (scaled for display), used by the radar chart
            "attr_codes": {c: (None if pd.isna(player_row.get(c))
                               else scale_attr(float(player_row.get(c))))
                           for c in ALL_ATTR_CODES if c in player_row.index},
        }
        if with_summary:
            profile["summary"] = self._summary(profile)
        else:
            profile["summary"] = _mock_dashboard_summary(profile)
        return profile

    def _summary(self, profile: dict) -> str:
        sys = ("You are a football scout. Write a 3-4 sentence profile: playing "
               "style, main strengths, and clear weaknesses. Be balanced and honest "
               "about limitations. Plain text, no markdown.")
        top = ", ".join(f"{a} {v}" for a, v in profile["top_attributes"])
        weak = ", ".join(f"{a} {v}" for a, v, p in profile["weaknesses"]) or "none notable"
        prompt = (f"Player: {profile['name']}, age {profile['age']}, "
                  f"{'/'.join(profile['positions'])}, {profile['club']}. "
                  f"Strengths: {top}. Weaknesses (low vs peers): {weak}. "
                  f"Tagline: {profile.get('tagline')}. Write the scouting profile.")
        return call_llm(prompt, system=sys, json_mode=False)


def _mock_dashboard_summary(profile: dict) -> str:
    """Fallback summary if no LLM (used by the mock path)."""
    top = profile["top_attributes"]
    style = profile.get("tagline") or "versatile player"
    strengths = ", ".join(a for a, _ in top[:3])
    club = profile.get("club")
    where = f"at {club}" if isinstance(club, str) and club.strip() else "currently a free agent"
    return (f"{profile['name']} is a {profile['age']}-year-old {style.lower()} "
            f"{where}. Strongest in {strengths}. "
            f"Best suited to a role leveraging these qualities.")
