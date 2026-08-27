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
from orchestrator import (readable_spec, format_value, detect_fit_query,
                          _explain_fit)
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
    # add playing-style archetypes (Poacher, Target Man, ...) so the Style column
    # and the style filter work — the agents below all read from this frame.
    from roles import RoleClusterer
    players = RoleClusterer(players).attach(players)
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


@st.cache_resource
def get_committee():
    """The multi-agent scouting committee (built once, on top of the pipeline)."""
    from committee import ScoutingCommittee
    from orchestrator import ScoutingPipeline
    # reuse the already-loaded players via a lightweight pipeline wrapper
    pipe = ScoutingPipeline(S["players"])
    return ScoutingCommittee(pipe)

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
    f"<div style='display:flex;align-items:center;gap:16px;margin-bottom:2px'>"
    f"{_logo_html}"
    f"<h1 style='color:{ACCENT};margin:0'>AI Football Scouter</h1></div>",
    unsafe_allow_html=True)
st.markdown(
    "<p style='color:#8a8a8a;margin:0 0 14px 2px;font-size:0.95rem;'>"
    "Scout 176,000 players by role, style, value & squad fit.</p>",
    unsafe_allow_html=True)

for k in ("mode", "spec", "request", "result", "similar_to", "team_note",
          "fit_result"):
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
    if len(pool) == 0:
        # every candidate was filtered out — return an empty, well-formed result
        # so the UI can show a friendly "no matches" message instead of crashing.
        return {"pool": 0, "ranked": pool, "shortlist": [],
                "report": "No players matched all of those filters. Try relaxing "
                          "one — a wider age range, higher budget, or fewer "
                          "qualities.",
                "team_note": team_note, "pool_size": 0}
    ranked = enrich(S["scoring"].run(pool, spec))
    shortlist = [summarise_player(r, spec) for _, r in ranked.iterrows()]
    # The LLM report is the slowest step (a second model call), so it's deferred:
    # results render immediately and the report is generated lazily on display.
    return {"pool": len(pool), "ranked": ranked, "shortlist": shortlist,
            "report": None, "team_note": team_note, "pool_size": len(pool),
            "_request": st.session_state.request}


def bargain_label(ratio, hi=1.3, lo=0.7):
    if pd.isna(ratio):
        return ""
    if ratio >= hi:
        return f"BARGAIN {ratio:.1f}x"
    if ratio <= lo:
        return f"pricey {ratio:.1f}x"
    return f"fair {ratio:.1f}x"


def contract_label(status, expires):
    """Compact contract note for the table: just 'FREE' or the expiry date."""
    if not isinstance(status, str) or status == "unknown":
        return ""
    if status == "expired":
        return "FREE"
    return expires or ""


def run_player_fit(player_name, club_name):
    """Score how well a named player would fit a named club."""
    players = S["players"]
    rows = players[players["Name"].str.lower() == player_name.lower().strip()]
    if rows.empty:
        rows = players[players["Name"].str.contains(
            player_name.strip(), case=False, na=False, regex=False)]
    if rows.empty:
        return {"error": f"Couldn't find a player called '{player_name}'."}
    club = S["team_fit"].find_club(club_name)
    if not club:
        return {"error": f"Couldn't find a club called '{club_name}'."}
    row = rows.sort_values("value_mid", ascending=False).iloc[0]
    fit = S["team_fit"].player_fit(row, club)
    fit["explanation"] = _explain_fit(fit)
    return fit


def nat_label(row):
    """Show 'PER, GRE' when a player holds a second nationality, else just 'PER'.

    Dual nationals often surface in nationality-filtered searches through their
    second passport, so showing it explains why they matched and helps with
    non-EU roster planning.
    """
    nat = row.get("Nat", "") or ""
    nat2 = row.get("nat2_code")
    if isinstance(nat2, str) and nat2 and nat2 != nat:
        return f"{nat}, {nat2}"
    return nat


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
from search_builder import (POSITIONS as _POS, QUALITIES as _QUAL,
                            UI_TEXT as _UIT, build_prompt as _build_prompt)

# language toggle — the button shows the CURRENT language (EN while English,
# EL while Greek) and flips on click.
_lang = st.session_state.get("ui_lang", "en")
_tcol1, _tcol2 = st.columns([6, 1])
with _tcol2:
    if st.button(_lang.upper(), use_container_width=True,
                 help="Switch language / Άλλαξε γλώσσα"):
        st.session_state.ui_lang = "el" if _lang == "en" else "en"
        st.rerun()
T = _UIT[_lang]

with st.expander(T["title"], expanded=False):
    st.caption(T["intro"])
    gc1, gc2 = st.columns(2)
    with gc1:
        gb_pos = st.selectbox(T["position"], [T["any"]] + list(_POS.keys()),
                              key="gb_pos")
        styles = _POS.get(gb_pos, (None, []))[1] if gb_pos != T["any"] else []
        gb_style = st.selectbox(T["style"], [T["any"]] + styles,
                                key="gb_style") if styles else T["any"]
        gb_quals = st.multiselect(T["qualities"], list(_QUAL.keys()),
                                  key="gb_quals")
        gb_foot = st.radio(T["foot"], T["foot_opts"], horizontal=True,
                           key="gb_foot")
        gb_nat = st.text_input(T["nationality"], key="gb_nat",
                               placeholder="Brazil · community · EU")
    with gc2:
        gb_age = st.slider(T["age"], 16, 40, 40, key="gb_age")
        # height is opt-in: a checkbox activates it, so the default 160 doesn't
        # silently filter. Until ticked, height is ignored entirely.
        gb_height_on = st.checkbox(T["height_on"], key="gb_height_on")
        gb_height = st.slider(T["min_height"], 160, 205, 175, key="gb_height",
                              disabled=not gb_height_on)
        gb_budget = st.slider(T["budget"], 0, 200, 0, key="gb_budget")
        gb_similar = st.text_input(T["similar"], key="gb_similar")
        gb_club = st.text_input(T["club"], key="gb_club")
        gb_contract = st.radio(T["contract"], T["contract_opts"],
                               horizontal=True, key="gb_contract")

    if gb_similar:
        st.info(T["similar_note"])

    _foot_map = dict(zip(T["foot_opts"], ["Any", "Right", "Left", "Either"]))
    built = _build_prompt(
        position=None if gb_pos == T["any"] else gb_pos,
        style=None if gb_style == T["any"] else gb_style,
        qualities=gb_quals,
        max_age=None if gb_age >= 40 else gb_age,
        budget=None if gb_budget == 0 else gb_budget,
        similar_to=gb_similar or None,
        club=gb_club or None,
        contract_idx=T["contract_opts"].index(gb_contract),
        foot=None if _foot_map.get(gb_foot, "Any") == "Any" else _foot_map[gb_foot],
        nationality=gb_nat or None,
        min_height=gb_height if gb_height_on else None,
    )
    st.markdown(f"**{T['preview']}:** *{built}*")
    if st.button(T["search"], type="primary", key="gb_go"):
        st.session_state["_builder_prompt"] = built

request = st.chat_input(T["chat_placeholder"])
_committee_on = st.checkbox(
    "🧑‍⚖️ Scouting-committee mode — three specialist agents (technical, "
    "financial, tactical) assess each pick and a head scout decides",
    key="committee_mode")
# a prompt built from the guided panel is processed exactly like a typed one
if st.session_state.get("_builder_prompt"):
    request = st.session_state.pop("_builder_prompt")
if request:
    st.session_state.request = request
    fit_q = detect_fit_query(request)
    frag = None if fit_q else extract_reference(request)
    if _committee_on and not fit_q and not frag:
        st.session_state.mode = "committee"
        with st.spinner("Convening the scouting committee… (evaluating each "
                        "player from three angles)"):
            st.session_state.committee_result = get_committee().review(
                request, top_n=5)
    elif fit_q:
        st.session_state.mode = "fit"
        st.session_state.fit_result = run_player_fit(*fit_q)
    elif frag:
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
# ---------------- player-to-club fit ----------------
if st.session_state.mode == "committee" and st.session_state.get("committee_result"):
    cr = st.session_state.committee_result
    club = cr.get("club")
    st.subheader("Scouting committee verdict")
    st.caption(f"Request: \"{cr['request']}\""
               + (f" · target club: {club}" if club else "")
               + " — ranked by the committee's aggregate score.")
    _rec_color = {"PURSUE": "#2e9e5b", "CONSIDER": "#d99a2b", "PASS": "#c0433a"}
    for v in cr["verdicts"]:
        col = _rec_color.get(v["recommendation"], "#888")
        meta = ""
        if v.get("age") or v.get("positions"):
            bits = []
            if v.get("age"):
                bits.append(str(v["age"]))
            if v.get("positions"):
                bits.append("/".join(v["positions"]))
            if v.get("club"):
                bits.append(club_label(v["club"]))
            meta = " · ".join(bits)
        st.markdown(
            f"### {v['player']} &nbsp; "
            f"<span style='background:{col};color:#fff;padding:2px 10px;"
            f"border-radius:12px;font-size:0.8rem;'>{v['recommendation']}"
            f" · {v['aggregate']}/100</span>", unsafe_allow_html=True)
        if meta:
            st.caption(meta)

        # profile snapshot (radar + strengths/weaknesses) beside the arguments
        prof_col, arg_col = st.columns([1, 2])
        with prof_col:
            idx = v.get("player_index")
            if idx is not None and idx in S["players"].index:
                try:
                    st.pyplot(player_radar(S["players"].loc[idx]),
                              use_container_width=True)
                except Exception:
                    pass
            if v.get("top_attributes"):
                strengths = ", ".join(f"{a} {val}"
                                      for a, val in v["top_attributes"][:4])
                st.markdown(f"**Strengths:** {strengths}")
            if v.get("weaknesses"):
                weak = ", ".join(f"{a} {val}" for a, val, *_ in v["weaknesses"][:3])
                st.markdown(f"**Weaknesses:** {weak}")
            # acquisition snapshot — the numbers a director needs to sign him
            _club = v.get("club")
            no_club = not isinstance(_club, str) or not _club.strip()
            is_free = v.get("contract_status") == "expired" or no_club
            fee = ("Free agent" if is_free
                   else format_value(v["value_eur"]) if v.get("value_eur")
                   else "n/a")
            acq = [f"**Fee:** {fee}"]
            if v.get("predicted_eur") and not is_free:
                acq.append(f"**Model value:** {format_value(v['predicted_eur'])}")
            if v.get("salary_eur"):
                acq.append(f"**Wage:** {format_value(v['salary_eur'])}/yr")
            contract = ("FREE" if is_free
                        else v.get("contract_expires") or "—")
            acq.append(f"**Contract:** {contract}")
            nat = v.get("nat") or ""
            if v.get("nat2"):
                nat = f"{nat}, {v['nat2']}"
            if nat:
                acq.append(f"**Nationality:** {nat}")
            st.markdown("  \n".join(acq))
        with arg_col:
            acols = st.columns(3)
            _icons = {"Technical Scout": "⚽", "Financial Analyst": "💰",
                      "Tactical Fit": "🎯"}
            for ac, a in zip(acols, v["assessments"]):
                score = a["score"] if a["score"] is not None else "—"
                icon = _icons.get(a["agent"], "")
                ac.markdown(f"**{icon} {a['agent']}**  \n`{score}`  \n"
                            f"{a.get('argument', a.get('note', ''))}")
            st.markdown(f"**Head Scout:** {v['verdict']}")
        st.divider()
    st.caption("Each specialist scores from data (the value model, team-fit "
               "analyser and suitability engine); the head scout only writes the "
               "synthesis, so the evidence stays reproducible.")
    if st.button("⬇ Build committee report (PDF)"):
        with st.spinner("Building committee report…"):
            from report_pdf import build_committee_report
            pdf_bytes = build_committee_report(cr, format_value)
        st.download_button("⬇ Download PDF", pdf_bytes,
                           "committee_report.pdf", "application/pdf")


if st.session_state.mode == "fit" and st.session_state.get("fit_result"):
    f = st.session_state.fit_result
    if f.get("error"):
        st.warning(f["error"])
    else:
        def _bar(score):
            """Colour a 0-100 score green/amber/red."""
            if score >= 67:
                return "#2e9e5b"
            if score >= 40:
                return "#d99a2b"
            return "#c0433a"

        def _score_row(label, score, sub):
            colour = _bar(score)
            st.markdown(
                f"""
                <div style="margin:0.35rem 0;">
                  <div style="display:flex;justify-content:space-between;
                              font-size:0.9rem;color:#c9c9c9;">
                    <span>{label}</span><span style="color:{colour};
                    font-weight:600;">{score}</span>
                  </div>
                  <div style="background:#2a2a2a;border-radius:6px;height:9px;
                              overflow:hidden;margin-top:3px;">
                    <div style="width:{score}%;background:{colour};height:100%;
                                border-radius:6px;"></div>
                  </div>
                  <div style="font-size:0.72rem;color:#7d7d7d;margin-top:2px;">
                    {sub}</div>
                </div>""",
                unsafe_allow_html=True)

        verdict = ("Strong fit" if f["sporting_fit"] >= 67 and f["feasible"] >= 50
                   else "Unaffordable" if f["sporting_fit"] >= 67
                   else "Squad already covered" if f["upgrade"] < 55
                   else "Worth considering")
        st.markdown(f"### {f['player']} → {f['club']}")
        st.markdown(
            f"<span style='background:{_bar(f['sporting_fit'])};color:#fff;"
            f"padding:2px 10px;border-radius:12px;font-size:0.8rem;'>"
            f"{verdict}</span>", unsafe_allow_html=True)
        st.write("")

        left, right = st.columns(2)
        with left:
            st.caption("SPORTING CASE")
            _score_row("Overall sporting fit", f["sporting_fit"],
                       "blended quality, need & gap")
            _score_row("Upgrade on squad", f["upgrade"],
                       f"{f['player_overall']} overall vs "
                       f"{f['best_incumbent']} best incumbent")
            _score_row("Positional need", f["need"],
                       f"{f['depth']} senior option(s) there")
            _score_row("Fills weak spots", f["gap_fit"],
                       "improves the squad's soft areas: "
                       + (", ".join(f.get("weak_attr_names", f["weak_attrs"])[:3])
                          or "n/a"))
        with right:
            st.caption("FINANCIAL CASE")
            _score_row("Affordable for club", f["feasible"],
                       "fee + wages vs the club's ceiling")

        st.write(f["explanation"])
        st.caption("Sporting and financial cases are scored separately: a "
                   "world-class player improves any squad, but that isn't useful "
                   "advice if the club could never afford him.")


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
    if res["pool_size"] == 0:
        st.info("No players matched all of those filters. Try relaxing one — "
                "a wider age range, higher budget, or fewer qualities.")
        st.stop()
    st.markdown(f"**Searched {res['pool']:,} matching players.**")
    # generate the natural-language report lazily, so the table below shows fast
    if res.get("report") is None:
        with st.spinner("Writing the scouting briefing…"):
            res["report"] = S["reporting"].run(res.get("_request", ""),
                                                res["shortlist"])
    st.markdown(res["report"])

    # top-3 highlight cards with badges + radar
    top = res["ranked"].head(3)
    cols = st.columns(3)
    for col, (_, r) in zip(cols, top.iterrows()):
        with col:
            badges = ""
            if r["suitability"] >= 75:
                badges += "<span class='badge badge-quality'>TOP QUALITY</span>"
            if r.get("bargain_ratio", 0) >= S["value_model"].bargain_hi:
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
            "Style": p.get("style") or "",
            "Club": club_label(p.get("club")), "Nat": nat_label(r),
            "Value (dataset)": format_value(p["value_eur"]),
            "Value (predicted)": format_value(r["predicted_value"]),
            "Suitability": p["suitability"],
            "Contract": contract_label(r.get("contract_status"), r.get("Expires")),
            "Value check": bargain_label(r["bargain_ratio"], S["value_model"].bargain_hi, S["value_model"].bargain_lo),
        })
    table_df = pd.DataFrame(rows)
    # Stretch to the full container width (no empty gap on the right) while
    # keeping every column visible. use_container_width distributes the space,
    # and the per-column widths below bias that distribution so the text-heavy
    # columns get more room and the short ones stay compact — all 11 fit without
    # the last ones being pushed off-screen.
    st.dataframe(
        table_df, use_container_width=True, hide_index=True,
        column_config={
            "Player": st.column_config.TextColumn("Player", width="medium"),
            "Age": st.column_config.NumberColumn("Age", width="small"),
            "Positions": st.column_config.TextColumn("Pos", width="small"),
            "Style": st.column_config.TextColumn("Style", width="medium"),
            "Club": st.column_config.TextColumn("Club", width="medium"),
            "Nat": st.column_config.TextColumn("Nat", width="small"),
            "Value (dataset)": st.column_config.TextColumn("Value", width="small"),
            "Value (predicted)": st.column_config.TextColumn("Predicted",
                                                             width="small"),
            "Suitability": st.column_config.NumberColumn("Fit", width="small"),
            "Contract": st.column_config.TextColumn("Contract", width="small"),
            "Value check": st.column_config.TextColumn("Value check",
                                                       width="medium"),
        })

    # export: PDF report (all shortlisted) + CSV. The PDF (10 dashboards + three
    # matplotlib radar charts) is expensive, so it is built ONLY when the user
    # asks for it — otherwise every search paid that cost up front and blocked
    # the results from showing. The CSV is cheap and always ready.
    exp_cols = st.columns(2)
    if exp_cols[0].button("⬇ Build scouting report (PDF)"):
        with st.spinner("Building report with radar charts…"):
            dashboards_all = []
            for _, row in res["ranked"].iterrows():
                d = S["dashboard"].build(row, with_summary=False)
                d["predicted_value"] = row["predicted_value"]
                dashboards_all.append(d)
            pdf_bytes = build_report(
                {**res, "pool_size": res["pool"]}, st.session_state.request,
                readable_spec(st.session_state.spec), format_value,
                dashboards_all, st.session_state.team_note)
        st.download_button("⬇ Download PDF", pdf_bytes,
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
        lo, hi = sim_filters.get("min_age"), sim_filters.get("max_age")
        if lo and hi:
            active.append(f"age {int(lo)}\u2013{int(hi)}")
        elif hi:
            active.append(f"age \u2264{int(hi)}")
        elif lo:
            active.append(f"age \u2265{int(lo)}")
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
                "Style": (r.get("style") if isinstance(r.get("style"), str) else ""),
                "Club": club_label(r.get("Club")), "Nat": nat_label(r),
                "Value (dataset)": format_value(r["value_mid"]),
                "Value (predicted)": format_value(r["predicted_value"]),
                "Similarity (lower=closer)": r["similarity_distance"],
                "Value check": bargain_label(r["bargain_ratio"], S["value_model"].bargain_hi, S["value_model"].bargain_lo),
            })

        # top-3 radar cards (closest matches)
        top = sim.head(3)
        cols = st.columns(3)
        for col, (_, r) in zip(cols, top.iterrows()):
            with col:
                badges = ""
                if r.get("bargain_ratio", 0) >= S["value_model"].bargain_hi:
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
