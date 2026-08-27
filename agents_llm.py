"""
agents_llm.py — the two LLM-powered agents.

RequirementsAgent : natural-language request -> structured spec (JSON)
ReportingAgent    : ranked shortlist -> readable scouting explanation

Both go through llm.call_llm, so they work with Gemini (if a key is set) or
the built-in mock (if not).
"""
from __future__ import annotations
import json
from llm import call_llm
from data import (ATTRIBUTES, POSITION_GROUPS, resolve_position, resolve_attribute,
                  LEAGUE_GROUPS, EUROPEAN_NATIONS, COUNTRY_TO_CODE, FOOT_KEYWORDS,
                  EU27_NATIONS, COMMUNITY_NATIONS)
import re


# sensible default weights per position, used only if the LLM gives no usable ones
_POSITION_DEFAULTS = {
    "GK": {"Ref": 1.5, "Han": 1.3, "1v1": 1.2, "Cmd": 1.0, "Aer": 1.0, "Kic": 0.8},
    "DC": {"Mar": 1.4, "Tck": 1.4, "Hea": 1.3, "Pos": 1.2, "Str": 1.1, "Jum": 1.0},
    "DR": {"Tck": 1.2, "Mar": 1.1, "Pac": 1.2, "Sta": 1.1, "Cro": 1.0},
    "DL": {"Tck": 1.2, "Mar": 1.1, "Pac": 1.2, "Sta": 1.1, "Cro": 1.0},
    "DM": {"Tck": 1.3, "Pos": 1.2, "Pas": 1.2, "Wor": 1.1, "Cnt": 1.0},
    "MC": {"Pas": 1.3, "Vis": 1.2, "Tec": 1.1, "Dec": 1.1, "Wor": 1.0},
    "AMC": {"Pas": 1.2, "Vis": 1.3, "Dri": 1.2, "Tec": 1.2, "OtB": 1.1, "Fla": 1.0},
    "AML": {"Pac": 1.3, "Dri": 1.3, "Tec": 1.1, "Cro": 1.0, "Fla": 1.0},
    "AMR": {"Pac": 1.3, "Dri": 1.3, "Tec": 1.1, "Cro": 1.0, "Fla": 1.0},
    "STC": {"Fin": 1.5, "OtB": 1.3, "Cmp": 1.2, "Pac": 1.1, "Ant": 1.1, "Hea": 1.0},
}


def _default_weights(position_codes: list[str]) -> dict:
    """Pick reasonable default weights based on the requested position(s)."""
    for code in position_codes:
        if code in _POSITION_DEFAULTS:
            return dict(_POSITION_DEFAULTS[code])
    # generic outfield default
    return {"Pac": 1.0, "Dri": 1.0, "Fin": 1.0, "Pas": 1.0, "Tck": 1.0}


# Playing-style archetypes shift the emphasis WITHIN a position — a target man
# and a poacher are both strikers but judged on different traits. When a style
# is detected these weights are used instead of the plain positional defaults.
_STYLE_DEFAULTS = {
    "Poacher": {"Fin": 1.5, "OtB": 1.4, "Ant": 1.2, "Cmp": 1.2, "Acc": 1.1},
    "Target Man": {"Hea": 1.5, "Str": 1.4, "Jum": 1.3, "Fin": 1.2, "Bra": 1.1},
    "Complete Forward": {"Fin": 1.3, "Dri": 1.2, "Tec": 1.2, "OtB": 1.2, "Str": 1.1},
    "Pressing Forward": {"Wor": 1.4, "Agg": 1.3, "Sta": 1.3, "Fin": 1.1, "Ant": 1.1},
    "Inside Forward": {"Dri": 1.4, "Fin": 1.3, "Acc": 1.3, "Fla": 1.2, "Tec": 1.1},
    "Classic Winger": {"Cro": 1.5, "Pac": 1.3, "Dri": 1.2, "Acc": 1.2},
    "Wide Playmaker": {"Pas": 1.4, "Vis": 1.3, "Tec": 1.3, "Cro": 1.1, "Dri": 1.1},
    "Work-rate Wideman": {"Wor": 1.4, "Sta": 1.3, "Tck": 1.2, "Cro": 1.1, "Pac": 1.1},
    "Advanced Playmaker": {"Pas": 1.5, "Vis": 1.4, "Tec": 1.3, "Fla": 1.1, "OtB": 1.1},
    "Shadow Striker": {"Fin": 1.4, "OtB": 1.4, "Lon": 1.2, "Ant": 1.2, "Dri": 1.1},
    "Trequartista": {"Fla": 1.4, "Dri": 1.3, "Tec": 1.3, "Vis": 1.2, "Pas": 1.2},
    "Deep-lying Playmaker": {"Pas": 1.5, "Vis": 1.4, "Tec": 1.3, "Dec": 1.2, "Cmp": 1.1},
    "Box-to-Box": {"Sta": 1.4, "Wor": 1.3, "OtB": 1.2, "Tck": 1.1, "Fin": 1.1, "Pas": 1.1},
    "Ball-winner": {"Tck": 1.5, "Mar": 1.4, "Agg": 1.3, "Wor": 1.2, "Ant": 1.1},
    "Regista": {"Pas": 1.5, "Vis": 1.4, "Tec": 1.3, "Dec": 1.2, "Lon": 1.1},
    "Anchor": {"Pos": 1.5, "Mar": 1.3, "Tck": 1.3, "Cnt": 1.2, "Ant": 1.2},
    "Ball-winning DM": {"Tck": 1.5, "Agg": 1.4, "Str": 1.3, "Mar": 1.2, "Wor": 1.2},
    "Ball-playing Defender": {"Pas": 1.4, "Tec": 1.3, "Cmp": 1.3, "Mar": 1.2, "Tck": 1.2},
    "No-nonsense Stopper": {"Mar": 1.5, "Hea": 1.4, "Tck": 1.4, "Str": 1.3, "Pos": 1.2},
    "Pace CB": {"Pac": 1.4, "Acc": 1.3, "Mar": 1.3, "Tck": 1.2, "Pos": 1.1},
    "Attacking Full-back": {"Cro": 1.4, "Dri": 1.3, "Pac": 1.3, "Sta": 1.2, "OtB": 1.1},
    "Wing-back": {"Sta": 1.4, "Wor": 1.3, "Pac": 1.3, "Cro": 1.2, "Tck": 1.1},
    "Defensive Full-back": {"Tck": 1.5, "Mar": 1.4, "Pos": 1.3, "Str": 1.1, "Hea": 1.1},
}

# attribute words that signal the user asked for specific qualities (so we keep
# the LLM's / mock's chosen weights instead of overriding with role defaults)
_ATTR_KEYWORDS = None


def _mentions_attributes(request: str) -> bool:
    """True if the request names concrete qualities (not just a position)."""
    global _ATTR_KEYWORDS
    if _ATTR_KEYWORDS is None:
        # build from attribute names + common quality synonyms, EL/EN/greeklish
        base = {w.lower() for name in ATTRIBUTES.values() for w in name.split()}
        # drop words that also appear in non-attribute phrases and would cause
        # false positives — e.g. "free" (from "Free Kicks") matching "free agent",
        # "out" (from "Off the Ball" variants / "out of contract"), or common
        # filler words that show up in ordinary requests.
        base -= {"free", "kicks", "long", "work", "rate", "throw", "set",
                 "the", "of", "off", "ball", "out", "contract", "in", "on",
                 "position", "player", "players"}
        base |= {"fast", "quick", "pace", "pacey", "strong", "tall", "aerial",
                 "finishing", "clinical", "creative", "passing", "vision",
                 "tackling", "defensive", "dribbling", "technical", "crossing",
                 "physical", "composed", "leader", "workrate", "stamina",
                 "γρηγορ", "δυνατ", "ψηλ", "τελειωμα", "πασ", "ντριμπλ",
                 "τακλιν", "κρεμα", "μπαλα", "1v1", "1vs1", "αναχαιτ",
                 "grigor", "dynat", "psil", "teleioma", "ntribl"}
        _ATTR_KEYWORDS = base
    # word-boundary match so "pass" doesn't fire inside "passport" and short
    # attribute words only match as whole words
    import re
    low = request.lower()
    return any(re.search(r"\b" + re.escape(kw) + r"\b", low)
               for kw in _ATTR_KEYWORDS)

_ATTR_LIST = ", ".join(sorted(ATTRIBUTES.values()))

_REQ_SYSTEM = f"""You are a football scouting assistant. Convert the user's request
into a JSON spec. Output ONLY valid JSON, no prose.

Schema:
{{
  "position_label": string or null,   // e.g. "left winger", "striker", "goalkeeper"
  "max_age": int or null,
  "min_age": int or null,
  "max_value": number or null,        // in euros, e.g. 15000000
  "min_value": number or null,
  "weights": {{ "<attribute name>": number }},  // importance 0.5-2.0
  "top_n": int                        // default 10
}}

CRITICAL: attribute names in "weights" MUST be chosen EXACTLY from this list
(use these exact spellings, nothing else):
{_ATTR_LIST}

Pick the attributes that matter for the requested position and qualities. For a
goalkeeper use goalkeeping attributes (Reflexes, Handling, One on Ones, Command
of Area, Aerial Reach, Kicking, etc.), NOT outfield ones. Assign higher weights
to qualities the user emphasises ("very fast" > "some pace"). Never invent
attribute names outside the list.

The user may write in English, Greek, or greeklish (Greek written with Latin
letters). Understand all three and always return the spec with the English
attribute names from the list above."""


def _req_user_prompt(request: str) -> str:
    return f'User request: "{request}"\nReturn the JSON spec.'


class RequirementsAgent:
    """Parse a free-text request into a concrete, validated spec dict."""

    def run(self, request: str) -> dict:
        raw = call_llm(_req_user_prompt(request), system=_REQ_SYSTEM, json_mode=True)
        spec = self._to_spec(raw, request)
        spec["_request"] = request
        # league + nationality are detected from the raw text (robust across
        # both LLM and mock, since these are categorical filters)
        self._add_league_nationality(spec, request)
        return spec

    def _add_league_nationality(self, spec: dict, request: str):
        low = request.lower()
        # ---- league groups ----
        subs = []
        for label, league_subs in LEAGUE_GROUPS.items():
            if label in low:
                subs.extend(league_subs)
        if any(p in low for p in ["top 5", "top-5", "top five", "biggest leagues",
                                   "major leagues", "μεγαλυτερα πρωταθληματα",
                                   "μεγαλα πρωταθληματα", "megalytera protathlimata"]):
            subs.extend(LEAGUE_GROUPS["top 5"])
        spec["league_substrings"] = sorted(set(subs)) if subs else None

        # ---- nationality: specific country OR a group ----
        nat_set = None
        for word, code in sorted(COUNTRY_TO_CODE.items(), key=lambda kv: -len(kv[0])):
            if word in low:
                nat_set = {code}
                break
        if nat_set is None:
            # group keywords (order matters: community/EU before generic European)
            if any(p in low for p in ["κοινοτικ", "koinotik", "eu-eligible",
                                       "community player", "eu eligible"]):
                nat_set = COMMUNITY_NATIONS
            elif any(p in low for p in ["eu national", "eu-national", "eu passport",
                                         "ε.ε.", "ευρωπαικη ενωση", "eu citizen"]):
                nat_set = EU27_NATIONS
            elif any(p in low for p in ["european", "europe", "ευρωπαι", "eyropai",
                                         "ευρωπη", "eyropi"]):
                nat_set = EUROPEAN_NATIONS
        spec["nationality_set"] = nat_set

        # ---- free agents (empty club) ----
        spec["free_agent"] = any(p in low for p in [
            "free agent", "free transfer", "ελευθερος", "eleftheros",
            "χωρις ομαδα", "xoris omada", "χωρις συλλογο", "without a club",
        ])

        # ---- wage ceiling: "wage under 50k a week", "salary under 2m a year",
        # "μισθος κατω απο 500k". Stored annually to match the dataset. ----
        spec["max_salary"] = None
        wage_m = re.search(
            r"(?:wage|salary|wages|μισθ\w*|misth\w*)[^0-9]{0,20}"
            r"(\d+(?:[.,]\d+)?)\s*([kmκμ])?",
            low)
        if wage_m:
            amount = float(wage_m.group(1).replace(",", "."))
            unit = (wage_m.group(2) or "").lower()
            if unit in ("k", "κ"):
                amount *= 1_000
            elif unit in ("m", "μ"):
                amount *= 1_000_000
            weekly = any(p in low for p in ["per week", "a week", "p/w", "weekly",
                                            "εβδομαδ", "evdomad"])
            spec["max_salary"] = amount * 52 if weekly else amount

        # ---- playing-style archetype (Poacher, Target Man, Regista, ...) ----
        from roles import detect_style
        spec["style"] = detect_style(request)

        # ---- upgrade-only: only show players better than the club already has ----
        spec["upgrade_only"] = any(p in low for p in [
            "better than", "upgrade", "improve", "stronger than",
            "καλυτερο απο", "καλυτερους απο", "αναβαθμιση", "αναβαθμισ",
            "kalytero apo", "anavathmisi", "να βελτιωσ", "beltiosi",
        ])

        # ---- contract status: expired (free) vs expiring soon (cheap/Bosman) ----
        spec["expired_contract"] = any(p in low for p in [
            "out of contract", "contract expired", "expired contract",
            "ληγμενο συμβολαιο", "ληξε το συμβολαιο", "eleuthero symbolaio",
            "bosman", "μποσμαν",
        ])
        if any(p in low for p in ["expiring", "expires soon", "last year of contract",
                                  "final year", "συμβολαιο ληγει", "ληγει το συμβολαιο",
                                  "symvolaio ligei", "τελευταιο χρονο"]):
            spec["max_contract_months"] = 12
        else:
            spec["max_contract_months"] = None

        # ---- best-position-only: match the player's PRIMARY role, not any
        # position they can fill (e.g. "natural striker", "η βασικη του θεση") ----
        spec["best_pos_only"] = any(p in low for p in [
            "best position", "main position", "primary position",
            "natural position", "natural ", "primarily a", "genuine ",
            "βασικη θεση", "βασικη του θεση", "κυρια θεση", "φυσικη θεση",
            "vasiki thesi", "kyria thesi", "fysiki thesi", "καθαρος",
        ])

        # ---- "best" without specific qualities -> rank by predicted value ----
        # (only when the user gave no attribute words, so weights are the default)
        wants_best = any(p in low for p in [
            "best", "top", "καλυτερ", "kalyter", "κορυφαι", "koryfai",
        ])
        spec["rank_by_quality"] = wants_best

        # ---- height: range (185-190cm), then over/under, then keyword ----
        min_height = None
        max_height = None
        mr = re.search(r"(\d{3})\s*(?:-|–|to|εως|ως)\s*(\d{3})\s*cm", low)
        if mr and int(mr.group(1)) < int(mr.group(2)):
            min_height, max_height = int(mr.group(1)), int(mr.group(2))
        else:
            m = re.search(r"(?:over|above|taller than|at least|πανω απο|τουλαχιστον|min)\s*(\d(?:[.,]\d+)?)\s*m\b", low)
            if m:
                min_height = int(float(m.group(1).replace(",", ".")) * 100)
            else:
                m = re.search(r"(?:over|above|taller than|at least|πανω απο|τουλαχιστον|min)\s*(\d{3})\s*cm", low)
                if m:
                    min_height = int(m.group(1))
            if min_height is None and any(w in low for w in ["tall ", "ψηλος", "psilos", "tall,", "tall."]):
                min_height = 188
            m = re.search(r"(?:under|below|shorter than|κατω απο)\s*(\d{3})\s*cm", low)
            if m:
                max_height = int(m.group(1))
        spec["min_height"] = min_height
        spec["max_height"] = max_height

        # ---- preferred foot ----
        foot = None
        for word, f in FOOT_KEYWORDS.items():
            if word in low:
                foot = f
                break
        spec["foot"] = foot

        # ---- international experience: "capped", "international", "εθνικη" ----
        if any(p in low for p in ["international", "capped", "national team",
                                   "εθνικη ομαδα", "διεθνης", "diethnis"]):
            spec["min_caps"] = 1
        else:
            spec["min_caps"] = None

        # ---- team-fit: "scout for AEK", "σκαουτερ στην ΑΕΚ", "για την ΑΕΚ" ----
        spec["team_fit_club"] = self._detect_club(request)

    def _detect_club(self, request: str) -> str | None:
        """Detect a club the user scouts for (enables team-fit scoring)."""
        low = request.lower()
        patterns = [
            r"(?:scouting for|scout for|recruit for|playing for|for)\s+"
            r"([a-z0-9à-ÿ'.\s]+?)"
            r"(?:\s*(?:,|\.|:|and|need|looking|searching|who|that|to|$))",
            r"(?:σκαουτερ (?:στην|στον|στη)|για την|για τον|για τη)\s+"
            r"([a-zα-ωά-ώà-ÿ0-9'.\s]+?)"
            r"(?:\s*(?:,|\.|:|και|ψαχνω|θελω|που|$))",
        ]
        stop = {"a", "the", "an", "my", "club", "team", "player", "παικτη", "εναν",
                "us", "we", "our team", "the team", "the club", "the squad",
                "the dressing room", "dressing room", "the future", "now",
                "sale", "resale", "depth", "the bench", "the first team"}
        # words that clearly start a new clause, so the club name ends before them
        # ("centre back for Getafe better than what we have" -> "getafe")
        tail_cuts = [
            " better", " upgrade", " improve", " stronger", " with ", " under ",
            " aged", " budget", " who ", " which ", " cheaper", " on current",
            " current squad", " καλυτερ", " αναβαθμ", " με ", " κατω ",
        ]
        for pat in patterns:
            m = re.search(pat, low)
            if m:
                club = m.group(1).strip()
                for cut in tail_cuts:
                    idx = club.find(cut)
                    if idx > 0:
                        club = club[:idx].strip()
                if len(club) >= 2 and club not in stop:
                    return club
        return None

    def _to_spec(self, raw: str, request: str = "") -> dict:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {}

        # The mock already returns position_codes + code weights.
        # The real LLM returns human labels -> normalise both paths here.
        if "position_codes" in data:
            position_codes = data["position_codes"]
        else:
            label = data.get("position_label")
            position_codes = resolve_position(label) if label else []

        weights_in = data.get("weights", {}) or {}
        weights = {}
        for key, w in weights_in.items():
            code = key if key in ATTRIBUTES else resolve_attribute(key)
            if code:
                weights[code] = float(w)

        # If the user named no concrete qualities, prefer role-appropriate
        # defaults over whatever generic weights the LLM guessed. Otherwise a
        # bare "left back" gets scored on Pace/Dribbling/Finishing and ranks
        # attack-minded wide players above actual defenders. A detected playing
        # style refines this further (a target man vs a poacher), taking priority
        # over the plain positional defaults.
        if not _mentions_attributes(request):
            from roles import detect_style
            style = detect_style(request)
            if style and style in _STYLE_DEFAULTS:
                weights = dict(_STYLE_DEFAULTS[style])
            elif position_codes:
                weights = _default_weights(position_codes)
            elif not weights:
                weights = _default_weights(position_codes)
        elif not weights:
            weights = _default_weights(position_codes)

        return {
            "position_codes": position_codes,
            "max_age": data.get("max_age"),
            "min_age": data.get("min_age"),
            "max_value": data.get("max_value"),
            "min_value": data.get("min_value"),
            "weights": weights,
            "explicit_qualities": _mentions_attributes(request),
            "top_n": int(data.get("top_n") or 10),
        }


# ------------------------------------------------------------------ #
# Reporting Agent
# ------------------------------------------------------------------ #
_REP_SYSTEM = """You are a football scout writing a short shortlist briefing.
Given the user's original request and a ranked list of players (with suitability
scores, values and key attributes), write a concise, friendly explanation of why
the top players fit. Reference concrete attribute values and note any bargains
(high suitability, low value). Keep it under 200 words. Plain text, no markdown."""


class ReportingAgent:
    """Turn the ranked shortlist into a natural-language scouting report."""

    def run(self, request: str, shortlist: list[dict]) -> str:
        # Send the LLM only the fields it needs to write the briefing — a leaner
        # payload means fewer tokens and a faster response. Extra display fields
        # (value_range, contract_expires, raw positions) are dropped here.
        slim = [{k: p.get(k) for k in
                 ("name", "age", "club", "style", "value_eur", "contract",
                  "suitability", "key_attributes")}
                for p in shortlist]
        prompt = (
            f'Original request: "{request}"\n\n'
            f"Ranked shortlist (JSON):\n{json.dumps(slim, ensure_ascii=False)}\n\n"
            "Write the briefing."
        )
        return call_llm(prompt, system=_REP_SYSTEM, json_mode=False)
