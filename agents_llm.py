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
        spec = self._to_spec(raw)
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
        stop = {"a", "the", "an", "my", "club", "team", "player", "παικτη", "εναν"}
        for pat in patterns:
            m = re.search(pat, low)
            if m:
                club = m.group(1).strip()
                if len(club) >= 2 and club not in stop:
                    return club
        return None

    def _to_spec(self, raw: str) -> dict:
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
        if not weights:
            weights = _default_weights(position_codes)

        return {
            "position_codes": position_codes,
            "max_age": data.get("max_age"),
            "min_age": data.get("min_age"),
            "max_value": data.get("max_value"),
            "min_value": data.get("min_value"),
            "weights": weights,
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
        prompt = (
            f'Original request: "{request}"\n\n'
            f"Ranked shortlist (JSON):\n{json.dumps(shortlist, ensure_ascii=False)}\n\n"
            "Write the briefing."
        )
        return call_llm(prompt, system=_REP_SYSTEM, json_mode=False)
