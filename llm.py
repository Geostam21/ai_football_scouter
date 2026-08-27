"""
llm.py — a thin LLM wrapper.

If GEMINI_API_KEY is set, calls Google Gemini via the REST endpoint (no SDK, so
no version conflicts). Otherwise it falls back to a lightweight rule-based "mock"
so the whole pipeline runs with zero setup.
"""
from __future__ import annotations
import os, json, re, time

_last_call = [0.0]  # timestamp of last Gemini call (throttle between calls)
_MIN_GAP = 0.8      # seconds between calls, keeps us under the free RPM limit


GEMINI_MODEL = "gemini-flash-lite-latest"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def _get_key() -> str | None:
    """Return the Gemini API key from an env var (local) or Streamlit secrets (cloud)."""
    key = os.getenv("GEMINI_API_KEY")
    if key:
        return key
    # fall back to Streamlit secrets when deployed on Streamlit Cloud
    try:
        import streamlit as st
        return st.secrets.get("GEMINI_API_KEY")
    except Exception:
        return None


def _has_gemini() -> bool:
    return bool(_get_key())


_llm_cache: dict = {}   # (prompt, system, json_mode) -> response, avoids repeat calls


def call_llm(prompt: str, *, system: str = "", json_mode: bool = False) -> str:
    """Return the model's text output. Uses Gemini if a key exists, else mock.

    Responses are cached in-memory: re-running the same search (or two searches
    that parse identically) reuses the first result instead of paying for
    another Gemini round-trip.
    """
    ckey = (prompt, system, json_mode)
    if ckey in _llm_cache:
        return _llm_cache[ckey]
    if _has_gemini():
        import time
        last_err = None
        for attempt in range(3):
            try:
                out = _call_gemini(prompt, system=system, json_mode=json_mode)
                _llm_cache[ckey] = out
                return out
            except Exception as e:
                last_err = e
                time.sleep(1 * (attempt + 1))  # 1s, 2s, 3s backoff
        print(f"[llm] Gemini failed after retries ({last_err}); using mock.")
        return _mock(prompt, json_mode=json_mode)
    out = _mock(prompt, json_mode=json_mode)
    _llm_cache[ckey] = out
    return out


def _call_gemini(prompt: str, *, system: str, json_mode: bool) -> str:
    import requests
    # throttle: ensure at least _MIN_GAP seconds since the last call
    wait = _MIN_GAP - (time.time() - _last_call[0])
    if wait > 0:
        time.sleep(wait)
    _last_call[0] = time.time()
    url = GEMINI_URL.format(model=GEMINI_MODEL)
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    if system:
        payload["systemInstruction"] = {"parts": [{"text": system}]}
    if json_mode:
        payload["generationConfig"] = {"responseMimeType": "application/json"}
    r = requests.post(
        url,
        headers={"x-goog-api-key": _get_key(),
                 "Content-Type": "application/json"},
        json=payload,
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]


# ------------------------------------------------------------------ #
# MOCK fallback — good enough to demo the flow without any API key.
# ------------------------------------------------------------------ #
def _mock(prompt: str, *, json_mode: bool) -> str:
    if json_mode:
        return _mock_requirements(prompt)
    if "scouting profile" in prompt.lower() or "top attributes:" in prompt.lower() or "strengths:" in prompt.lower():
        return _mock_profile(prompt)
    return _mock_report(prompt)


def _mock_profile(prompt: str) -> str:
    name = "This player"
    m = re.search(r"Player:\s*([^,]+),", prompt)
    if m:
        name = m.group(1).strip()
    m2 = re.search(r"Strengths:\s*([^.]+)", prompt)
    strengths = m2.group(1).strip() if m2 else "several areas"
    return (f"{name} stands out for {strengths}. A well-rounded profile for the "
            "role, with clear physical and technical qualities. Worth a closer "
            "look given the value on offer.")


def _mock_requirements(prompt: str) -> str:
    from data import resolve_position, resolve_attribute, POSITION_GROUPS, SYNONYMS, ATTRIBUTES
    text = prompt.lower()

    # ---- Greek / greeklish position vocabulary ----
    GREEK_POS = {
        "τερματοφυλακ": "GK", "γκολκιπερ": "GK", "termatofylak": "GK", "goalkeeper": "GK",
        "στοπερ": "DC", "κεντρικος αμυντικος": "DC", "stoper": "DC", "senterbak": "DC",
        "δεξιος μπακ": "DR", "deksios bak": "DR", "δεξι μπακ": "DR",
        "αριστερος μπακ": "DL", "aristeros bak": "DL", "αριστερο μπακ": "DL",
        "αμυντικος μεσος": "DM", "amyntikos mesos": "DM",
        "κεντρικος μεσος": "MC", "μεσος": "MC", "mesos": "MC", "kentrikos mesos": "MC",
        "επιθετικος μεσος": "AMC", "epithetikos mesos": "AMC", "δεκαρι": "AMC",
        "αριστερος εξτρεμ": "AML", "aristeros ekstrem": "AML", "αριστερο εξτρεμ": "AML",
        "δεξιος εξτρεμ": "AMR", "deksios ekstrem": "AMR", "δεξι εξτρεμ": "AMR",
        "εξτρεμ": "AML", "ekstrem": "AML", "φτερο": "AML", "ftero": "AML",
        "επιθετικος": "STC", "σεντερ φορ": "STC", "epithetikos": "STC", "forvar": "STC",
        "φορβαρ": "STC", "κυνηγος": "STC",
    }
    position_codes = []
    # raw codes typed directly ("mc", "dc", "stc", "amr"...) — common shorthand
    _RAW_CODES = {"gk", "dc", "dr", "dl", "wbr", "wbl", "dm", "mc", "mr", "ml",
                  "amc", "amr", "aml", "stc", "st"}
    for tok in re.findall(r"\b[a-z]{2,3}\b", text):
        if tok in _RAW_CODES:
            position_codes = ["STC" if tok == "st" else tok.upper()]
            break
    if not position_codes:
        for gk_word, code in sorted(GREEK_POS.items(), key=lambda kv: -len(kv[0])):
            if gk_word in text:
                position_codes = [code]
                break
    if not position_codes:
        for label in sorted(POSITION_GROUPS, key=len, reverse=True):
            if label in text:
                position_codes = POSITION_GROUPS[label]
                break

    # ---- age: ranges first (23-27, between 23 and 27, 23 to 27), then single ----
    max_age = None
    min_age = None
    m = re.search(r"(?:between\s+)?(\d{2})\s*(?:-|–|to|and|εως|ως|μεχρι|-)\s*(\d{2})\s*(?:years|ετων|χρον|yo|y\.?o\.?)?", text)
    if m and int(m.group(1)) < int(m.group(2)) and int(m.group(1)) >= 15 and int(m.group(2)) <= 45:
        min_age, max_age = int(m.group(1)), int(m.group(2))
    else:
        # The age number must not be part of a bigger token: without these
        # guards "under 30m" (a fee) reads as age 30, and "190cm" (a height)
        # reads as age 19, silently filtering out most of the dataset.
        _not_unit = r"(?!\d)(?!\s*(?:m\b|k\b|cm|mm|million|mill|thousand|€|\$|εκατ|χιλ))"
        m = re.search(r"(?:under|below|younger than|max age|u|κατω των|κατω απο|κατω|"
                      r"kato ton|kato apo|εως|μεχρι)\s*(\d{2})" + _not_unit, text)
        if m:
            max_age = int(m.group(1))
        m2 = re.search(r"(?:over|above|older than|πανω απο|ανω των|τουλαχιστον)\s*"
                       r"(\d{2})" + _not_unit, text)
        if m2:
            min_age = int(m2.group(1))

        # keyword-based age bands when no explicit number is given
        if max_age is None and min_age is None:
            if re.search(r"\b(young|youngster|prospect|wonderkid|νεαρ|νεος|talent)",
                         text):
                max_age = 21
            elif re.search(r"\b(veteran|experienced|βετεραν|εμπειρ|ageing|"
                           r"aging)", text):
                min_age = 30

    # ---- budget: ranges first, then single upper bound ----
    max_value = None
    min_value = None
    t2 = text
    for gm in ["εκατομμυρια", "εκατομμύρια", "εκατ.", "εκατ", "εκ.", "ekatommyria", "ekat"]:
        t2 = t2.replace(gm, "m")
    for gk_thousand in ["χιλιαδες", "χιλ", "xiliades"]:
        t2 = t2.replace(gk_thousand, "k")
    # range: "5-10m", "between 5 and 10m", "5 to 10 m"
    mr = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:-|–|to|and|εως|ως)\s*(\d+(?:[.,]\d+)?)\s*(m|million|k|thousand)", t2)
    if mr:
        lo = float(mr.group(1).replace(",", ".")); hi = float(mr.group(2).replace(",", "."))
        mult = 1e6 if mr.group(3).startswith("m") else 1e3
        min_value, max_value = lo * mult, hi * mult
    else:
        m = re.search(r"(?:budget|up to|under|max|below|μεχρι|εως|budjet|μπατζετ)?\s*[€$]?\s*(\d+(?:[.,]\d+)?)\s*(m|million|k|thousand)", t2)
        if m:
            num = float(m.group(1).replace(",", "."))
            max_value = num * (1e6 if m.group(2).startswith("m") else 1e3)

    # keyword-based budget when no explicit figure is given: "cheap"/"budget"/
    # "affordable" imply a modest ceiling so the results aren't topped by the
    # most expensive superstars.
    if max_value is None and re.search(
            r"\b(cheap|bargain|budget|affordable|φθην|φτην|οικονομικ|"
            r"ftin|fthin|oikonomik)", text):
        max_value = 10e6

    # ---- attributes: English + Greek + greeklish lexicon ----
    lexicon = dict(SYNONYMS)
    for code, name in ATTRIBUTES.items():
        lexicon.setdefault(name.lower(), code)
    GREEK_ATTR = {
        "ταχυτητα": "Pac", "γρηγορ": "Pac", "taxytita": "Pac", "grigor": "Pac",
        "ντριμπλα": "Dri", "ντριπλα": "Dri", "dribla": "Dri", "ntribla": "Dri",
        "τελειωμα": "Fin", "σκοραρ": "Fin", "ευστοχ": "Fin", "teleioma": "Fin", "skorar": "Fin",
        "πασα": "Pas", "μοιρασμα": "Pas", "pasa": "Pas",
        "δυναμη": "Str", "δυνατ": "Str", "dynami": "Str", "dynat": "Str",
        "κεφαλι": "Hea", "αερα": "Hea", "kefali": "Hea", "aera": "Hea",
        "μαρκαρισμα": "Mar", "markarisma": "Mar",
        "τακλιν": "Tck", "αμυν": "Tck", "taklin": "Tck", "amyn": "Tck",
        "αντανακλαστικ": "Ref", "antanaklastik": "Ref",
        "τεχνικ": "Tec", "texnik": "Tec",
        "αντοχ": "Sta", "antox": "Sta",
        "οραμα": "Vis", "δημιουργ": "Vis", "orama": "Vis",
        "ψυχραιμ": "Cmp", "psychraim": "Cmp", "composure": "Cmp",
        "ηγετ": "Ldr", "ηγεσια": "Ldr", "igetikes": "Ldr", "leadership": "Ldr",
        "εργατικ": "Wor", "ergatik": "Wor",
        "αποφασ": "Dec", "apofas": "Dec",
        "τοποθετ": "Pos", "topothet": "Pos",
        "προβλεπ": "Ant", "anticipation": "Ant",
        "θαρρος": "Bra", "γενναι": "Bra",
        "ομαδικ": "Tea", "omadik": "Tea",
        "συγκεντρωσ": "Cnt", "sygkentrosi": "Cnt",
        "ταλεντο": "Fla", "φαντασια": "Fla", "flair": "Fla",
        # ball control / first touch
        "μπαλα στα ποδια": "Fir", "control": "Fir", "μπαλα": "Fir",
        "κοντρολ": "Fir", "kontrol": "Fir", "first touch": "Fir",
        "ball control": "Fir", "on the ball": "Fir", "στα ποδια": "Fir",
        # dribbling / taking players on / 1v1 (outfield sense)
        "1v1": "Dri", "1vs1": "Dri", "1 vs 1": "Dri", "εναντιον": "Dri",
        "παρ' το": "Dri", "take on": "Dri", "beat his man": "Dri",
        # interceptions / winning the ball back -> tackling
        "αναχαιτ": "Tck", "anachait": "Tck", "κλεψιμο": "Tck", "klepsimo": "Tck",
        "interception": "Tck", "win the ball": "Tck", "διακοπ": "Tck",
    }
    lexicon.update(GREEK_ATTR)

    weights = {}
    emphasis = {"very": 1.6, "excellent": 1.6, "high": 1.4, "great": 1.4,
                "strong": 1.3, "good": 1.2, "solid": 1.0, "decent": 0.9,
                "some": 0.7, "average": 0.7,
                # Greek / greeklish emphasis
                "πολυ": 1.5, "poly": 1.5, "εξαιρετικ": 1.6, "eksairetik": 1.6,
                "καλ": 1.2, "kal": 1.2, "αρκετα": 1.0, "arketa": 1.0,
                "μετρι": 0.8, "metri": 0.8}
    for word, code in sorted(lexicon.items(), key=lambda kv: -len(kv[0])):
        idx = text.find(word)
        if idx == -1:
            continue
        window = text[max(0, idx - 15):idx]
        w = 1.0
        for emph, mult in emphasis.items():
            if emph in window:
                w = mult
                break
        weights[code] = max(weights.get(code, 0), w)

    if not weights:
        # Position-aware defaults so a bare position query ("left back") ranks by
        # the traits that role is actually judged on, instead of always falling
        # back to Pace/Dribbling/Finishing (which buried good full-backs behind
        # quick, attack-minded players with no defensive ability).
        pc = set(position_codes)
        if "GK" in pc:
            weights = {"Ref": 1.5, "Han": 1.3, "1v1": 1.2, "Cmd": 1.0}
        elif pc & {"DC"}:
            weights = {"Mar": 1.4, "Tck": 1.4, "Hea": 1.3, "Pos": 1.2, "Str": 1.1}
        elif pc & {"DR", "DL", "WBR", "WBL"}:
            # full-backs / wing-backs: defend first, then get forward
            weights = {"Tck": 1.3, "Mar": 1.2, "Pos": 1.2, "Cro": 1.1,
                       "Pac": 1.1, "Sta": 1.1}
        elif pc & {"DM"}:
            weights = {"Tck": 1.3, "Pos": 1.3, "Mar": 1.2, "Pas": 1.1, "Wor": 1.1}
        elif pc & {"MC", "MR", "ML"}:
            weights = {"Pas": 1.2, "Vis": 1.2, "Tec": 1.1, "Wor": 1.1, "Sta": 1.1}
        elif pc & {"AMC", "AMR", "AML"}:
            weights = {"Dri": 1.2, "Tec": 1.2, "Pas": 1.1, "Fla": 1.1, "OtB": 1.1}
        elif pc & {"STC"}:
            weights = {"Fin": 1.5, "OtB": 1.3, "Cmp": 1.2, "Pac": 1.1}
        else:
            weights = {"Pac": 1.0, "Dri": 1.0, "Fin": 1.0}

    spec = {
        "position_codes": position_codes,
        "max_age": max_age, "min_age": min_age,
        "max_value": max_value, "min_value": min_value,
        "weights": weights,
        "top_n": 10,
    }
    return json.dumps(spec)


def _mock_report(prompt: str) -> str:
    m = re.search(r"\[.*\]", prompt, re.DOTALL)
    if not m:
        return "No candidates to report."
    try:
        players = json.loads(m.group(0))
    except json.JSONDecodeError:
        return "Could not parse shortlist."
    lines = ["Here are the top matches for your request:\n"]
    for i, p in enumerate(players[:5], 1):
        val = f"€{p['value_eur']:,}" if p.get("value_eur") else "value n/a"
        attrs = ", ".join(f"{k} {v}" for k, v in list(p.get("key_attributes", {}).items())[:3])
        lines.append(
            f"{i}. {p['name']} ({p['age']}, {p.get('club','?')}) — "
            f"suitability {p['suitability']}/100, {val}. Standout: {attrs}."
        )
    return "\n".join(lines)


def normalise_player_name(fragment: str) -> str | None:
    """Use the LLM to turn a Greek/greeklish/misspelled name into its Latin form.

    e.g. "βινισιους" -> "Vinicius". Returns None if no key or on failure.
    """
    if not _has_gemini():
        return None
    prompt = (f'A user referred to a footballer as "{fragment}". Reply with ONLY '
              f'the most likely full player name in Latin letters (no accents, no '
              f'extra words). If unsure, reply with your single best guess.')
    try:
        out = _call_gemini(prompt, system="", json_mode=False)
        return out.strip().split("\n")[0][:60]
    except Exception:
        return None
