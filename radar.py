"""
radar.py — compact radar (spider) chart for a player's attribute profile.

Returns a matplotlib figure sized like a small card element, themed for the
dark UI. Groups the many attributes into 6-8 summary axes so the chart stays
readable (a "player card" look rather than 40 bars).
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from data import ATTRIBUTES

# summary axes -> the attribute codes that feed each (averaged)
OUTFIELD_AXES = {
    "Pace": ["Pac", "Acc"],
    "Shooting": ["Fin", "Lon", "Pen"],
    "Passing": ["Pas", "Vis", "Cro"],
    "Dribbling": ["Dri", "Fir", "Tec", "Agi"],
    "Defending": ["Tck", "Mar", "Pos", "Ant"],
    "Physical": ["Str", "Sta", "Jum", "Bal"],
}
GK_AXES = {
    "Reflexes": ["Ref", "1v1"],
    "Handling": ["Han", "Aer", "Pun"],
    "Command": ["Cmd", "Com"],
    "Distribution": ["Kic", "Thr", "Pas"],
    "Positioning": ["Pos", "Ant", "Cnt"],
    "Composure": ["Cmp", "Dec", "Cnt"],
}

# theme colours (match the dark/gold UI)
_GOLD = "#e8c766"
_BG = "#141416"
_GRID = "#3a3a40"
_TEXT = "#c8c8c0"


def player_radar(row, is_gk: bool | None = None, accent: str = _GOLD):
    """Build a radar figure for one player row (a pandas Series)."""
    if is_gk is None:
        is_gk = "GK" in (row.get("positions") or [])
    axes_def = GK_AXES if is_gk else OUTFIELD_AXES

    labels = list(axes_def.keys())
    values = []
    for codes in axes_def.values():
        vals = [row[c] for c in codes if c in row.index and row[c] == row[c]]
        # x5 to match the 0-100 scale shown everywhere else in the UI
        values.append(np.mean(vals) * 5 if vals else 0)

    # close the loop
    values = values + values[:1]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(3.2, 3.2), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_BG)

    ax.plot(angles, values, color=accent, linewidth=2)
    ax.fill(angles, values, color=accent, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, color=_TEXT, fontsize=9)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(["25", "50", "75", "100"], color=_GRID, fontsize=7)
    ax.set_ylim(0, 100)
    ax.grid(color=_GRID, linewidth=0.6)
    ax.spines["polar"].set_color(_GRID)
    fig.tight_layout(pad=0.5)
    return fig
