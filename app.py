"""
app.py — AI Football Scouter (Streamlit chatbot).

Dark/gold multi-agent scouting UI: NL search (EN/GR/greeklish), editable weights,
team-fit, rich filters, radar dashboards (top 3), quality/bargain/fit badges,
value-prediction bargains, smart similarity search, PDF + CSV export.

Run:  streamlit run app.py
"""
import streamlit as st
import pandas as pd
from data import load_players, ATTRIBUTES
from agents_core import CandidateAgent, ScoringAgent, summarise_player
from agents_llm import RequirementsAgent, ReportingAgent
from agent_dashboard import DashboardAgent
from agent_similarity import (SimilarityAgent, extract_reference, resolve_player)
from ml import ValueModel, SimilarityIndex
from team_fit import TeamFitAnalyzer
from orchestrator import readable_spec, format_value
from radar import player_radar
from report_pdf import build_report
try:
    from llm import normalise_player_name
except Exception:
    normalise_player_name = None

import os as _os0
_pi = _os0.path.join(_os0.path.dirname(__file__), "scout_icon.png")
st.set_page_config(page_title="AI Football Scouter",
                   page_icon=(_pi if _os0.path.exists(_pi) else "⚽"),
                   layout="wide")

# ---- theme (parametric: swap ACCENT to change gold -> orange) ----
ACCENT = "#e8c766"
ACCENT_DARK = "#b8973f"
BG = "#1c1c22"          # base dark grey
SURFACE = "#26262d"     # cards
SURFACE2 = "#3a3a44"    # lighter grey panels (chat, tables) - stands out but still dark
BORDER = "#3d3d46"
TEXT = "#ececec"
MUTED = "#a0a098"

# subtle SVG background: two stylised players contesting a ball (silhouette)
BG_SVG = """
<svg xmlns='http://www.w3.org/2000/svg' width='1200' height='800' viewBox='0 0 1200 800'>
  <g fill='none' stroke='%23e8c766' stroke-opacity='0.05' stroke-width='2'>
    <circle cx='600' cy='420' r='34'/>
    <path d='M480 300 q-20 60 -10 140 l30 120 M470 560 l-24 90 M470 560 l30 86
             M480 300 l40 -50 40 40 M560 290 l-30 70'/>
    <path d='M720 300 q20 60 10 140 l-30 120 M730 560 l24 90 M730 560 l-30 86
             M720 300 l-40 -50 -40 40 M640 290 l30 70'/>
  </g>
</svg>
""".replace("\n", "").replace("#", "%23")

st.markdown(f"""
<style>
    .stApp {{
        background: {BG};
        background-image: url("data:image/svg+xml;utf8,{BG_SVG}");
        background-repeat: no-repeat; background-position: center 120px;
        background-size: 780px auto; color: {TEXT};
    }}
    /* top header + bottom chat container bars -> dark */
    header[data-testid="stHeader"] {{ background: {BG}; }}
    [data-testid="stBottomBlockContainer"] {{ background: {BG}; }}
    [data-testid="stBottom"] {{ background: {BG}; }}
    section[data-testid="stSidebar"] {{ background: {SURFACE}; border-right: 1px solid {BORDER}; }}
    h1,h2,h3,h4 {{ color: {TEXT}; }}
    .stMarkdown, p, span, label, li {{ color: {TEXT}; }}
    [data-testid="stMetricValue"] {{ color: {ACCENT}; }}
    [data-testid="stMetricLabel"] {{ color: {MUTED}; }}
    [data-testid="stDataFrame"], [data-testid="stTable"] {{ background: {SURFACE2}; border-radius:8px; }}
    [data-testid="stExpander"] {{ background: {SURFACE2}; border:1px solid {BORDER}; border-radius:8px; }}
    .stAlert {{ background: {SURFACE2}; border:1px solid {BORDER}; color:{TEXT}; }}
    .stButton>button, .stDownloadButton>button {{
        background: {SURFACE2}; color: {ACCENT}; border:1px solid {ACCENT_DARK};
        border-radius:8px; font-weight:600;
    }}
    .stButton>button:hover, .stDownloadButton>button:hover {{
        background: {ACCENT}; color: {BG}; border-color:{ACCENT};
    }}
    /* chat input: force dark grey box + visible white text across all inner wrappers */
    [data-testid="stChatInput"],
    [data-testid="stChatInput"] > div,
    [data-testid="stChatInput"] div[data-baseweb="base-input"],
    [data-testid="stChatInput"] div[data-baseweb="input"] {{
        background: {SURFACE2} !important; border-color: {ACCENT_DARK} !important;
        border-radius: 10px;
    }}
    [data-testid="stChatInput"] textarea {{
        background: {SURFACE2} !important; color: #ffffff !important; caret-color: {ACCENT};
    }}
    [data-testid="stChatInput"] textarea::placeholder {{ color: {MUTED} !important; opacity:1; }}
    [data-testid="stChatInputSubmitButton"] {{ color: {ACCENT}; }}
    .badge {{ display:inline-block; padding:2px 8px; border-radius:5px; font-size:11px;
             font-weight:700; margin-left:6px; }}
    .badge-quality {{ background:{ACCENT}; color:{BG}; }}
    .badge-bargain {{ background:#2f6b47; color:#c8f0d4; }}
    .badge-fit {{ background:#3a5a8a; color:#cfe0f5; }}
    .pcard {{ background:{SURFACE}; border:1px solid {BORDER}; border-radius:12px;
             padding:14px 16px; margin-bottom:8px; }}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_system():
    players = load_players()
    return {
        "players": players,
        "requirements": RequirementsAgent(),
        "candidates": CandidateAgent(players),
        "scoring": ScoringAgent(players),
        "reporting": ReportingAgent(),
        "dashboard": DashboardAgent(players),
        "value_model": ValueModel(players),
        "similarity": SimilarityAgent(players, SimilarityIndex(players)),
        "team_fit": TeamFitAnalyzer(players),
        "names": players["Name"].tolist(),
    }


S = get_system()

import base64, os as _os
_logo_path = _os.path.join(_os.path.dirname(__file__), "scout_icon.png")
_logo_html = ""
try:
    with open(_logo_path, "rb") as _f:
        _b64 = base64.b64encode(_f.read()).decode()
    _logo_html = f"<img src='data:image/png;base64,{_b64}' width='72' height='72' style='border-radius:12px;vertical-align:middle'/>"
except Exception:
    _logo_html = ""

st.markdown(
    f"<div style='display:flex;align-items:center;gap:16px;margin-bottom:0'>"
    f"{_logo_html}"
    f"<h1 style='color:{ACCENT};margin:0'>AI Football Scouter</h1></div>",
    unsafe_allow_html=True)
st.caption("Describe the player — position, age, budget, qualities, nationality, "
           "league — or ask for players similar to a star.")

for k in ("mode", "spec", "request", "result", "similar_to", "team_note"):
    st.session_state.setdefault(k, None)


def enrich(df):
    df = df.copy()
    df["predicted_value"] = S["value_model"].predict(df)
    df["bargain_ratio"] = S["value_model"].bargain_ratio(df)
    return df


def run_search(spec):
    team_note = None
    club_name = spec.get("team_fit_club")
    if club_name:
        club = S["team_fit"].find_club(club_name)
        if club:
            gaps = S["team_fit"].squad_gaps(club, spec["position_codes"])
            spec["weights"] = S["team_fit"].adjust_weights(spec["weights"], gaps)
            team_note = S["team_fit"].summary(club, gaps)
        else:
            team_note = f"Couldn't find '{club_name}' in the data; ranking normally."
    pool = S["candidates"].run(spec)
    ranked = enrich(S["scoring"].run(pool, spec))
    # "best" with no real attribute criteria -> rank by predicted quality
    if spec.get("rank_by_quality") and set(spec.get("weights", {}).keys()) <= {"Pac", "Dri", "Fin"}:
        ranked = ranked.sort_values("predicted_value", ascending=False).head(spec.get("top_n", 10))
        ranked["suitability"] = (ranked["predicted_value"].rank(pct=True) * 100).round(1)
    shortlist = [summarise_player(r, spec) for _, r in ranked.iterrows()]
    report = S["reporting"].run(st.session_state.request, shortlist)
    return {"pool": len(pool), "ranked": ranked, "shortlist": shortlist,
            "report": report, "team_note": team_note, "pool_size": len(pool)}


def bargain_label(ratio):
    if pd.isna(ratio):
        return ""
    if ratio >= 1.3:
        return f"BARGAIN {ratio:.1f}x"
    if ratio <= 0.7:
        return f"pricey {ratio:.1f}x"
    return f"fair {ratio:.1f}x"


def club_label(club):
    """Empty/NaN club -> 'Free agent'."""
    if not isinstance(club, str) or not club.strip():
        return "Free agent"
    return club


@st.cache_data(show_spinner=False)
def _shap_cached():
    """Compute SHAP value drivers once and cache."""
    return S["value_model"].shap_importance(S["players"], top=8)


# ---------------- input ----------------
request = st.chat_input("e.g. left back under 25, budget 15M, good crossing  |  players like Pineda")
if request:
    st.session_state.request = request
    frag = extract_reference(request)
    if frag:
        res = resolve_player(frag, S["players"], llm_normaliser=normalise_player_name)
        st.session_state.mode = "similar"
        st.session_state.similar_to = res["match"]
        st.session_state["sim_alts"] = res.get("alternatives", [])
        st.session_state["sim_frag"] = frag
        # also parse the request for extra hard filters (age, nationality, etc.)
        try:
            fspec = S["requirements"].run(request)
            st.session_state["sim_filters"] = {
                "min_age": fspec.get("min_age"), "max_age": fspec.get("max_age"),
                "min_value": fspec.get("min_value"),
                "min_height": fspec.get("min_height"), "max_height": fspec.get("max_height"),
                "foot": fspec.get("foot"),
                "nationality_set": fspec.get("nationality_set"),
                "league_substrings": fspec.get("league_substrings"),
            }
            st.session_state["sim_prompt_maxval"] = fspec.get("max_value")
        except Exception:
            st.session_state["sim_filters"] = {}
            st.session_state["sim_prompt_maxval"] = None
    else:
        st.session_state.mode = "search"
        st.session_state.spec = S["requirements"].run(request)
        res = run_search(st.session_state.spec)
        st.session_state.result = res
        st.session_state.team_note = res["team_note"]

# ---------------- weight editor ----------------
if st.session_state.mode == "search" and st.session_state.spec:
    spec = st.session_state.spec
    with st.sidebar:
        st.subheader("Interpreted request")
        st.write(readable_spec(spec))
        if st.session_state.team_note:
            st.info(st.session_state.team_note)
        st.divider()
        st.subheader("Adjust weights")
        new_w = {}
        for code, w in spec["weights"].items():
            new_w[code] = st.slider(ATTRIBUTES.get(code, code), 0.0, 2.0, float(w), 0.1, key=f"w_{code}")
        if st.button("Re-run with these weights"):
            spec = dict(spec); spec["weights"] = new_w
            st.session_state.spec = spec
            st.session_state.result = run_search(spec)

        st.divider()
        with st.expander("What drives predicted value? (SHAP)"):
            st.caption("Global drivers of the value model's predictions, ranked "
                       "by mean absolute SHAP contribution.")
            for name, val in _shap_cached():
                st.markdown(f"- **{name}** · {val}")

# ---------------- results: search ----------------
if st.session_state.mode == "search" and st.session_state.result:
    res = st.session_state.result
    st.markdown(f"**Searched {res['pool']:,} matching players.**")
    st.markdown(res["report"])

    # top-3 highlight cards with badges + radar
    top = res["ranked"].head(3)
    cols = st.columns(3)
    for col, (_, r) in zip(cols, top.iterrows()):
        with col:
            badges = ""
            if r["suitability"] >= 75:
                badges += "<span class='badge badge-quality'>TOP QUALITY</span>"
            if r.get("bargain_ratio", 0) >= 1.3:
                badges += "<span class='badge badge-bargain'>BARGAIN</span>"
            if st.session_state.team_note and "weakest" in (st.session_state.team_note or ""):
                badges += "<span class='badge badge-fit'>FILLS GAP</span>"
            st.markdown(f"<div class='pcard'><b>{r['Name']}</b> {badges}<br>"
                        f"<span style='color:{MUTED};font-size:12px'>{int(r['Age'])} · "
                        f"{club_label(r.get('Club'))} · {r.get('Nat','')} · {format_value(r['value_mid'])}</span></div>",
                        unsafe_allow_html=True)
            st.pyplot(player_radar(r), use_container_width=True)

    # full table with nationality
    rows = []
    for (_, r), p in zip(res["ranked"].iterrows(), res["shortlist"]):
        rows.append({
            "Player": p["name"], "Age": p["age"],
            "Positions": "/".join(r.get("positions", [])),
            "Club": club_label(p.get("club")), "Nat": r.get("Nat", ""),
            "Value (dataset)": format_value(p["value_eur"]),
            "Value (predicted)": format_value(r["predicted_value"]),
            "Suitability": p["suitability"],
            "Value check": bargain_label(r["bargain_ratio"]),
        })
    table_df = pd.DataFrame(rows)
    st.dataframe(table_df, use_container_width=True, hide_index=True)

    # export: PDF report (all shortlisted) + CSV
    exp_cols = st.columns(2)
    dashboards_all = []
    for _, row in res["ranked"].iterrows():
        d = S["dashboard"].build(row, with_summary=False)
        d["predicted_value"] = row["predicted_value"]
        dashboards_all.append(d)
    pdf_bytes = build_report(
        {**res, "pool_size": res["pool"]}, st.session_state.request,
        readable_spec(st.session_state.spec), format_value,
        dashboards_all, st.session_state.team_note)
    exp_cols[0].download_button("⬇ Scouting report (PDF)", pdf_bytes,
                                "scouting_report.pdf", "application/pdf")
    exp_cols[1].download_button("⬇ Shortlist (CSV)", table_df.to_csv(index=False),
                                "shortlist.csv", "text/csv")

    # per-player detail (any of the 10)
    st.divider()
    names = [p["name"] for p in res["shortlist"]]
    chosen = st.selectbox("Full profile / find similar:", names)
    if chosen:
        row = res["ranked"][res["ranked"]["Name"] == chosen].iloc[0]
        prof = S["dashboard"].build(row)
        c1, c2 = st.columns([3, 2])
        with c1:
            st.subheader(prof["name"])
            st.caption(f"{prof.get('tagline','')} · {'/'.join(prof['positions'])} · {club_label(prof['club'])} · {prof.get('nationality','')}")
            st.markdown(prof["summary"])
            m = st.columns(4)
            m[0].metric("Age", prof["age"])
            m[1].metric("Height", f"{prof['height_cm']}cm" if prof["height_cm"] else "-")
            m[2].metric("Value", format_value(row["value_mid"]))
            m[3].metric("Predicted", format_value(row["predicted_value"]))
            sw = st.columns(2)
            sw[0].markdown("**Strengths**")
            for a, v in prof["top_attributes"][:5]:
                sw[0].markdown(f"- {a}: {v}")
            sw[1].markdown("**Weaknesses**")
            if prof["weaknesses"]:
                for a, v, pc in prof["weaknesses"]:
                    sw[1].markdown(f"- {a}: {v} (bottom {pc}%)")
            else:
                sw[1].markdown("- None notable")
        with c2:
            st.pyplot(player_radar(row), use_container_width=True)
        if st.button(f"Find players similar to {chosen}"):
            st.session_state.mode = "similar"
            st.session_state.similar_to = chosen
            st.session_state["sim_alts"] = []
            st.rerun()

# ---------------- results: similarity ----------------
if st.session_state.mode == "similar":
    frag = st.session_state.get("sim_frag", "")
    ref = st.session_state.similar_to
    if not ref:
        st.warning(f"Couldn't find a player matching '{frag}' in the dataset. "
                   "Try a player who appears in the data, or describe the profile instead.")
        if st.button("Back to search"):
            st.session_state.mode = "search"; st.rerun()
        st.stop()

    # disambiguation: offer same-surname alternatives
    alts = st.session_state.get("sim_alts", [])
    if alts:
        st.caption("Did you mean someone else?")
        pick_cols = st.columns(min(len(alts) + 1, 6))
        if pick_cols[0].button(f"✓ {ref}", key="keep_ref"):
            pass
        for i, alt in enumerate(alts[:5], 1):
            if pick_cols[i].button(alt, key=f"alt_{i}"):
                st.session_state.similar_to = alt
                st.session_state["sim_alts"] = [ref] + [a for a in alts if a != alt]
                st.rerun()

    ref = st.session_state.similar_to
    ref_row = S["players"][S["players"]["Name"] == ref].iloc[0]
    ref_val = ref_row["value_mid"]; ref_pos = ref_row["positions"]

    st.subheader(f"Players similar to {ref}")
    st.caption(f"{ref}: {'/'.join(ref_pos)} · {ref_row.get('Nat','')} · {format_value(ref_val)}. "
               "Same-position players by attribute similarity — for a replacement "
               "(equal, better, or cheaper).")

    sim_filters = st.session_state.get("sim_filters", {}) or {}
    # show which extra filters were parsed from the prompt
    active = []
    if sim_filters.get("min_age") or sim_filters.get("max_age"):
        lo = sim_filters.get("min_age") or ""; hi = sim_filters.get("max_age") or ""
        active.append(f"age {lo}-{hi}".replace("age -", "age ≤").replace("- ", "≥"))
    if sim_filters.get("nationality_set"):
        n = sim_filters["nationality_set"]
        active.append(list(n)[0] if len(n) == 1 else ("EU" if len(n) == 27 else "community/European"))
    if sim_filters.get("min_height"):
        active.append(f"≥{sim_filters['min_height']}cm")
    if sim_filters.get("max_height"):
        active.append(f"≤{sim_filters['max_height']}cm")
    if sim_filters.get("foot"):
        active.append(f"{sim_filters['foot']}-footed")
    if sim_filters.get("league_substrings"):
        active.append("selected leagues")
    if active:
        st.info("Extra filters from your request: " + ", ".join(active))

    col1, col2 = st.columns(2)
    with col1:
        prompt_maxval = st.session_state.get("sim_prompt_maxval")
        use_cap = st.checkbox("Set a max budget", value=bool(prompt_maxval))
        default_cap = int(prompt_maxval / 1_000_000) if prompt_maxval else 30
        cap = st.slider("Max value (€M)", 1, 150, min(default_cap, 150), disabled=not use_cap) * 1_000_000
        sort_choice = st.radio("Rank by", ["Closest match", "Best value within budget"],
                               index=0, horizontal=True)
    with col2:
        cheaper = st.checkbox(f"Only cheaper than {ref}", value=False)
        same_pos = st.checkbox("Same position only", value=True)

    sim = S["similarity"].find_similar(
        ref, k=10, max_value=(cap if use_cap else None),
        cheaper_only=cheaper, same_position=same_pos,
        sort_by=("similarity" if sort_choice.startswith("Closest") else "value"),
        filters=sim_filters)
    if len(sim):
        sim = enrich(sim)
        rows = []
        for _, r in sim.iterrows():
            rows.append({
                "Player": r["Name"], "Age": int(r["Age"]),
                "Positions": "/".join(r.get("positions", [])),
                "Club": club_label(r.get("Club")), "Nat": r.get("Nat", ""),
                "Value (dataset)": format_value(r["value_mid"]),
                "Value (predicted)": format_value(r["predicted_value"]),
                "Similarity (lower=closer)": r["similarity_distance"],
                "Value check": bargain_label(r["bargain_ratio"]),
            })

        # top-3 radar cards (closest matches)
        top = sim.head(3)
        cols = st.columns(3)
        for col, (_, r) in zip(cols, top.iterrows()):
            with col:
                badges = ""
                if r.get("bargain_ratio", 0) >= 1.3:
                    badges += "<span class='badge badge-bargain'>BARGAIN</span>"
                st.markdown(f"<div class='pcard'><b>{r['Name']}</b> {badges}<br>"
                            f"<span style='color:{MUTED};font-size:12px'>{int(r['Age'])} · "
                            f"{club_label(r.get('Club'))} · {r.get('Nat','')} · "
                            f"{format_value(r['value_mid'])}</span></div>",
                            unsafe_allow_html=True)
                st.pyplot(player_radar(r), use_container_width=True)

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # exports: PDF report + CSV
        sim_dash = []
        for _, row in sim.iterrows():
            dd = S["dashboard"].build(row, with_summary=False)
            dd["predicted_value"] = row.get("predicted_value")
            dd["value_eur"] = None if pd.isna(row.get("value_mid")) else int(row["value_mid"])
            sim_dash.append(dd)
        sim_shortlist = [{"name": r["Name"], "age": int(r["Age"]),
                          "club": r.get("Club"), "value_eur": (None if pd.isna(r["value_mid"]) else int(r["value_mid"])),
                          "suitability": 0} for _, r in sim.iterrows()]
        sim_pdf = build_report(
            {"pool_size": len(sim), "shortlist": sim_shortlist},
            f"Players similar to {ref}",
            f"similar to {ref} ({'/'.join(ref_pos)})", format_value,
            sim_dash, None)
        ecol = st.columns(2)
        ecol[0].download_button("⬇ Similar players report (PDF)", sim_pdf,
                                "similar_players.pdf", "application/pdf")
        ecol[1].download_button("⬇ Similar players (CSV)",
                                pd.DataFrame(rows).to_csv(index=False),
                                "similar_players.csv", "text/csv")

        # per-player full profile
        st.divider()
        names = sim["Name"].tolist()
        chosen = st.selectbox("Full profile:", names, key="sim_profile")
        if chosen:
            row = sim[sim["Name"] == chosen].iloc[0]
            prof = S["dashboard"].build(row)
            c1, c2 = st.columns([3, 2])
            with c1:
                st.subheader(prof["name"])
                st.caption(f"{prof.get('tagline','')} · {'/'.join(prof['positions'])} · "
                           f"{club_label(prof['club'])} · {prof.get('nationality','')}")
                st.markdown(prof["summary"])
                m = st.columns(4)
                m[0].metric("Age", prof["age"])
                m[1].metric("Height", f"{prof['height_cm']}cm" if prof["height_cm"] else "-")
                m[2].metric("Value", format_value(row["value_mid"]))
                m[3].metric("Predicted", format_value(row["predicted_value"]))
                sw = st.columns(2)
                sw[0].markdown("**Strengths**")
                for a, v in prof["top_attributes"][:5]:
                    sw[0].markdown(f"- {a}: {v}")
                sw[1].markdown("**Weaknesses**")
                if prof["weaknesses"]:
                    for a, v, pc in prof["weaknesses"]:
                        sw[1].markdown(f"- {a}: {v} (bottom {pc}%)")
                else:
                    sw[1].markdown("- None notable")
            with c2:
                st.pyplot(player_radar(row), use_container_width=True)
    else:
        st.info("No similar players with these filters. Raise the max value or uncheck 'cheaper'.")
    if st.button("Back to search"):
        st.session_state.mode = "search"; st.rerun()
