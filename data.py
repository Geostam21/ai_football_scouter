"""
data.py — dataset loading and the attribute glossary.

The FM26 dataset uses short codes for attributes (Pac = Pace, Fin = Finishing).
The glossary below maps human words -> codes so the Requirements Agent can turn
a natural-language request into concrete column weights, and so the Reporting
Agent can turn codes back into readable text.
"""
import re
import numpy as np
import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).parent / "fm26_final.csv"
DATA_PATH_GZ = Path(__file__).parent / "fm26_final.csv.gz"

# ---- attribute code -> human-readable name ----
ATTRIBUTES = {
    # Technical
    "Cor": "Corners", "Cro": "Crossing", "Dri": "Dribbling", "Fin": "Finishing",
    "Fir": "First Touch", "Fre": "Free Kicks", "Hea": "Heading", "Lon": "Long Shots",
    "L Th": "Long Throws", "Mar": "Marking", "Pas": "Passing", "Pen": "Penalties",
    "Tck": "Tackling", "Tec": "Technique",
    # Mental
    "Agg": "Aggression", "Ant": "Anticipation", "Bra": "Bravery", "Cmp": "Composure",
    "Cnt": "Concentration", "Dec": "Decisions", "Det": "Determination",
    "Fla": "Flair", "Ldr": "Leadership", "OtB": "Off the Ball", "Pos": "Positioning",
    "Tea": "Teamwork", "Vis": "Vision", "Wor": "Work Rate",
    # Physical
    "Acc": "Acceleration", "Agi": "Agility", "Bal": "Balance", "Jum": "Jumping Reach",
    "Pac": "Pace", "Sta": "Stamina", "Str": "Strength",
    # Goalkeeping
    "Aer": "Aerial Reach", "Cmd": "Command of Area", "Com": "Communication",
    "Ecc": "Eccentricity", "Han": "Handling", "Kic": "Kicking", "1v1": "One on Ones",
    "Pun": "Punching", "Ref": "Reflexes", "TRO": "Rushing Out", "Thr": "Throwing",
}

# reverse: lowercase human word -> code (for fuzzy matching in the Requirements Agent)
NAME_TO_CODE = {v.lower(): k for k, v in ATTRIBUTES.items()}

# common synonyms people actually type -> code
SYNONYMS = {
    "speed": "Pac", "quick": "Pac", "quickness": "Pac", "fast": "Pac",
    "shooting": "Fin", "goalscoring": "Fin", "finishing ability": "Fin",
    "passing ability": "Pas", "vision": "Vis", "creativity": "Vis",
    "dribbling ability": "Dri", "ball control": "Fir", "technique": "Tec",
    "tackling": "Tck", "defending": "Tck", "marking": "Mar",
    "strength": "Str", "physical": "Str", "power": "Str",
    "stamina": "Sta", "fitness": "Sta", "endurance": "Sta",
    "heading": "Hea", "aerial": "Hea", "jumping": "Jum",
    "leadership": "Ldr", "workrate": "Wor", "work rate": "Wor",
    "composure": "Cmp", "decisions": "Dec", "positioning": "Pos",
    "acceleration": "Acc", "agility": "Agi", "balance": "Bal",
    "crossing": "Cro", "long shots": "Lon", "flair": "Fla",
}

# ---- position groups: human label -> the role codes that satisfy it ----
POSITION_GROUPS = {
    "goalkeeper": ["GK"],
    "right back": ["DR", "WBR"], "left back": ["DL", "WBL"],
    "centre back": ["DC"], "center back": ["DC"], "defender": ["DC", "DR", "DL"],
    "defensive midfielder": ["DM"], "central midfielder": ["MC"],
    "midfielder": ["MC", "DM", "AMC"],
    "right midfielder": ["MR"], "left midfielder": ["ML"],
    "right winger": ["AMR"], "left winger": ["AML"], "winger": ["AMR", "AML"],
    "right attacking midfielder": ["AMR"], "left attacking midfielder": ["AML"],
    "attacking midfielder": ["AMC"], "playmaker": ["AMC", "MC"],
    "striker": ["STC"], "forward": ["STC", "AMC"], "centre forward": ["STC"],
}

ALL_ATTR_CODES = list(ATTRIBUTES.keys())

# named league groups -> substrings that identify member leagues in 'Based'.
# NOTE: the FM26 export labels leagues by the in-game "Division" name, which is
# clean for most top flights (Premier League, Serie A, Bundesliga, Ligue 1). The
# Spanish top flight had no distinct label (it fell back to a generic "First
# Division" shared with many other countries), so the 20 La Liga 2025-26 clubs
# were re-tagged as "La Liga" by club name during dataset prep.
LEAGUE_GROUPS = {
    "top 5": ["Premier League", "La Liga", "Serie A", "Bundesliga", "Ligue 1"],
    "top5": ["Premier League", "La Liga", "Serie A", "Bundesliga", "Ligue 1"],
    "premier league": ["Premier League"],
    "la liga": ["La Liga"],
    "laliga": ["La Liga"],
    "serie a": ["Serie A"],
    "bundesliga": ["Bundesliga"],
    "ligue 1": ["Ligue 1"],
    "eredivisie": ["Eredivisie"],
    "championship": ["Sky Bet Championship"],
}

# European nationality codes (FM 3-letter), for "European players" requests
EUROPEAN_NATIONS = {
    "ENG", "ESP", "ITA", "GER", "FRA", "NED", "POR", "BEL", "SCO", "WAL", "NIR",
    "IRL", "SUI", "AUT", "SWE", "NOR", "DEN", "FIN", "ISL", "POL", "CZE", "SVK",
    "HUN", "ROU", "BUL", "GRE", "CRO", "SRB", "SVN", "BIH", "MKD", "MNE", "ALB",
    "UKR", "RUS", "BLR", "LTU", "LVA", "EST", "TUR", "CYP", "MLT", "LUX", "KOS",
    "GEO", "ARM", "AZE", "MDA", "AND", "SMR", "LIE", "FRO",
}

# EU-27 member states (FM 3-letter codes) — "A' zone" / EU nationals
EU27_NATIONS = {
    "AUT", "BEL", "BUL", "CRO", "CYP", "CZE", "DEN", "EST", "FIN", "FRA", "GER",
    "GRE", "HUN", "IRL", "ITA", "LVA", "LTU", "LUX", "MLT", "NED", "POL", "POR",
    "ROU", "SVK", "SVN", "ESP", "SWE",
}

# EPO "community" players = A' zone (EU-27) + B' zone (association agreements).
# Source: official EPS document, Κοινοτικοί Β' Ζώνης 2026-2027.
COMMUNITY_B_ZONE = {
    "SMR",  # San Marino
    "AZE",  # Azerbaijan
    "ALB",  # Albania
    "ALG",  # Algeria
    "AND",  # Andorra
    "VAT",  # Vatican
    "MKD",  # North Macedonia
    "BIH",  # Bosnia & Herzegovina
    "GEO",  # Georgia
    "GIB",  # Gibraltar
    "SUI",  # Switzerland
    "ENG", "WAL", "SCO", "NIR",  # United Kingdom
    "ISL",  # Iceland
    "KOS",  # Kosovo
    "LIE",  # Liechtenstein
    "MAR",  # Morocco
    "MNE",  # Montenegro
    "MDA",  # Moldova
    "MON",  # Monaco
    "NOR",  # Norway
    "UKR",  # Ukraine
    "RUS",  # Russia
    "SRB",  # Serbia
    "TUR",  # Turkey
    "TUN",  # Tunisia
}
COMMUNITY_NATIONS = EU27_NATIONS | COMMUNITY_B_ZONE

# country name (English/Greek/greeklish) -> FM 3-letter code, for nationality filters
COUNTRY_TO_CODE = {
    "greek": "GRE", "greece": "GRE", "ελληνας": "GRE", "ελληνα": "GRE", "ελλαδα": "GRE", "ellinas": "GRE",
    "brazilian": "BRA", "brazil": "BRA", "βραζιλιανος": "BRA", "βραζιλια": "BRA", "vrazilianos": "BRA",
    "argentine": "ARG", "argentinian": "ARG", "argentina": "ARG", "αργεντινος": "ARG", "αργεντινη": "ARG",
    "french": "FRA", "france": "FRA", "γαλλος": "FRA", "γαλλια": "FRA", "gallos": "FRA",
    "english": "ENG", "england": "ENG", "αγγλος": "ENG", "αγγλια": "ENG", "agglos": "ENG",
    "spanish": "ESP", "spain": "ESP", "ισπανος": "ESP", "ισπανια": "ESP", "ispanos": "ESP",
    "german": "GER", "germany": "GER", "γερμανος": "GER", "γερμανια": "GER", "germanos": "GER",
    "italian": "ITA", "italy": "ITA", "ιταλος": "ITA", "ιταλια": "ITA", "italos": "ITA",
    "portuguese": "POR", "portugal": "POR", "πορτογαλος": "POR", "πορτογαλια": "POR",
    "dutch": "NED", "netherlands": "NED", "holland": "NED", "ολλανδος": "NED", "ολλανδια": "NED",
    "belgian": "BEL", "belgium": "BEL", "βελγος": "BEL", "βελγιο": "BEL",
    "croatian": "CRO", "croatia": "CRO", "κροατης": "CRO", "κροατια": "CRO",
    "serbian": "SRB", "serbia": "SRB", "σερβος": "SRB", "σερβια": "SRB",
    "turkish": "TUR", "turkey": "TUR", "τουρκος": "TUR", "τουρκια": "TUR",
    "american": "USA", "usa": "USA", "united states": "USA", "αμερικανος": "USA",
    "mexican": "MEX", "mexico": "MEX", "μεξικανος": "MEX", "μεξικο": "MEX",
    "colombian": "COL", "colombia": "COL", "κολομβιανος": "COL", "κολομβια": "COL",
    "uruguayan": "URU", "uruguay": "URU", "ουρουγουανος": "URU", "ουρουγουαη": "URU",
    "nigerian": "NGA", "nigeria": "NGA", "νιγηριανος": "NGA",
    "japanese": "JPN", "japan": "JPN", "ιαπωνας": "JPN", "ιαπωνια": "JPN",
    "moroccan": "MAR", "morocco": "MAR", "μαροκινος": "MAR", "μαροκο": "MAR",
    "senegalese": "SEN", "senegal": "SEN", "σενεγαλεζος": "SEN",
    # --- Greek country names ---
    "αλβανια": "ALB", "αλβανος": "ALB", "κοσοβο": "KOS",
    "βοσνια": "BIH", "σερβια": "SRB", "σερβος": "SRB",
    "κροατια": "CRO", "κροατης": "CRO", "σλοβενια": "SVN",
    "σλοβακια": "SVK", "τσεχια": "CZE", "πολωνια": "POL",
    "ουγγαρια": "HUN", "ρουμανια": "ROU", "βουλγαρια": "BUL",
    "ρωσια": "RUS", "ουκρανια": "UKR", "τουρκια": "TUR",
    "κυπρος": "CYP", "ελβετια": "SUI", "αυστρια": "AUT",
    "σκωτια": "SCO", "ουαλια": "WAL", "ιρλανδια": "IRL",
    "νορβηγια": "NOR", "σουηδια": "SWE", "δανια": "DEN",
    "φινλανδια": "FIN", "ισλανδια": "ISL", "σκανδιναβια": "SWE",
    "νιγηρια": "NGA", "γκανα": "GHA", "καμερουν": "CMR",
    "σενεγαλη": "SEN", "ακτη ελεφαντοστου": "CIV", "αιγυπτος": "EGY",
    "μαροκο": "MAR", "αλγερια": "ALG", "τυνησια": "TUN",
    "νοτια αφρικη": "RSA", "βραζιλια": "BRA", "αργεντινη": "ARG",
    "ουρουγουαη": "URU", "χιλη": "CHI", "κολομβια": "COL",
    "περου": "PER", "εκουαδορ": "ECU", "παραγουαη": "PAR",
    "βενεζουελα": "VEN", "βολιβια": "BOL", "μεξικο": "MEX",
    "ιαπωνια": "JPN", "νοτια κορεα": "KOR", "κορεα": "KOR",
    "κινα": "CHN", "αυστραλια": "AUS", "ιραν": "IRN",
    "σαουδικη αραβια": "KSA", "καναδας": "CAN", "αμερικη": "USA",
    "ηπα": "USA", "γεωργια": "GEO", "αρμενια": "ARM",
    "ισραηλ": "ISR", "αλβανοι": "ALB",
    # --- full country coverage (all 217 dataset codes, English names) ---

    "afghanistan": "AFG", "albania": "ALB", "algeria": "ALG", "america": "USA",
    "american samoa": "ASA", "andorra": "AND", "angola": "ANG", "anguilla": "AIA",
    "antigua": "ATG", "antigua and barbuda": "ATG", "argentina": "ARG", "armenia": "ARM",
    "aruba": "ARU", "australia": "AUS", "austria": "AUT", "azerbaijan": "AZE",
    "bahrain": "BHR", "bangladesh": "BAN", "barbados": "BRB", "belarus": "BLR",
    "belgium": "BEL", "belize": "BLZ", "benin": "BEN", "bermuda": "BER",
    "bolivia": "BOL", "bonaire": "BOE", "bosnia": "BIH", "bosnia and herzegovina": "BIH",
    "botswana": "BOT", "brazil": "BRA", "british virgin islands": "VGB", "brunei": "BRU",
    "bulgaria": "BUL", "burkina faso": "BFA", "burundi": "BDI", "cambodia": "CAM",
    "cameroon": "CMR", "canada": "CAN", "cape verde": "CPV", "cayman islands": "CAY",
    "central african republic": "CTA", "chad": "CHA", "chile": "CHI", "china": "CHN",
    "chinese taipei": "TPE", "colombia": "COL", "comoros": "COM", "congo": "CGO",
    "cook islands": "COK", "costa rica": "CRC", "cote d'ivoire": "CIV", "croatia": "CRO",
    "cuba": "CUB", "curacao": "CUW", "cyprus": "CYP", "czech republic": "CZE",
    "czechia": "CZE", "democratic republic of congo": "COD", "denmark": "DEN", "djibouti": "DJI",
    "dominica": "DMA", "dominican republic": "DOM", "dr congo": "COD", "east timor": "TLS",
    "ecuador": "ECU", "egypt": "EGY", "el salvador": "SLV", "england": "ENG",
    "equatorial guinea": "EQG", "eritrea": "ERI", "estonia": "EST", "eswatini": "SWZ",
    "ethiopia": "ETH", "faroe islands": "FRO", "fiji": "FIJ", "finland": "FIN",
    "france": "FRA", "french guiana": "GUF", "gabon": "GAB", "gambia": "GAM",
    "georgia": "GEO", "germany": "GER", "ghana": "GHA", "gibraltar": "GIB",
    "greece": "GRE", "grenada": "GRN", "guadeloupe": "GLP", "guam": "GUM",
    "guatemala": "GUA", "guinea": "GUI", "guinea-bissau": "GNB", "guyana": "GUY",
    "haiti": "HAI", "holland": "NED", "honduras": "HON", "hong kong": "HKG",
    "hungary": "HUN", "iceland": "ISL", "india": "IND", "indonesia": "IDN",
    "iran": "IRN", "iraq": "IRQ", "ireland": "IRL", "israel": "ISR",
    "italy": "ITA", "ivory coast": "CIV", "jamaica": "JAM", "japan": "JPN",
    "jordan": "JOR", "kazakhstan": "KAZ", "kenya": "KEN", "korea": "KOR",
    "kosovo": "KOS", "kyrgyzstan": "KGZ", "laos": "LAO", "latvia": "LVA",
    "lebanon": "LBN", "lesotho": "LES", "liberia": "LBR", "libya": "LBY",
    "liechtenstein": "LIE", "lithuania": "LTU", "luxembourg": "LUX", "macau": "MAC",
    "macedonia": "MKD", "madagascar": "MAD", "malawi": "MWI", "malaysia": "MAS",
    "mali": "MLI", "malta": "MLT", "martinique": "MTQ", "mauritania": "MTN",
    "mauritius": "MRI", "mayotte": "MAY", "mexico": "MEX", "micronesia": "FSM",
    "moldova": "MDA", "mongolia": "MNG", "montenegro": "MNE", "montserrat": "MSR",
    "morocco": "MAR", "mozambique": "MOZ", "myanmar": "MYA", "namibia": "NAM",
    "nepal": "NEP", "netherlands": "NED", "new caledonia": "NCL", "new zealand": "NZL",
    "nicaragua": "NCA", "niger": "NIG", "nigeria": "NGA", "north korea": "PRK",
    "north macedonia": "MKD", "northern ireland": "NIR", "norway": "NOR", "oman": "OMA",
    "pakistan": "PAK", "palestine": "PLE", "panama": "PAN", "papua new guinea": "PNG",
    "paraguay": "PAR", "peru": "PER", "philippines": "PHI", "poland": "POL",
    "portugal": "POR", "puerto rico": "PUR", "qatar": "QAT", "reunion": "REU",
    "romania": "ROU", "russia": "RUS", "rwanda": "RWA", "saint kitts": "SKN",
    "saint kitts and nevis": "SKN", "saint lucia": "LCA", "saint martin": "SMA", "saint vincent": "VIN",
    "samoa": "SAM", "san marino": "SMR", "sao tome": "STP", "saudi arabia": "KSA",
    "scotland": "SCO", "senegal": "SEN", "serbia": "SRB", "seychelles": "SEY",
    "sierra leone": "SLE", "singapore": "SGP", "slovakia": "SVK", "slovenia": "SVN",
    "solomon islands": "SOL", "somalia": "SOM", "south africa": "RSA", "south korea": "KOR",
    "south sudan": "SSD", "spain": "ESP", "sri lanka": "SRI", "sudan": "SDN",
    "suriname": "SUR", "swaziland": "SWZ", "sweden": "SWE", "switzerland": "SUI",
    "syria": "SYR", "tahiti": "TAH", "taiwan": "TPE", "tajikistan": "TJK",
    "tanzania": "TAN", "thailand": "THA", "timor-leste": "TLS", "togo": "TOG",
    "tonga": "TGA", "trinidad": "TRI", "trinidad and tobago": "TRI", "tunisia": "TUN",
    "turkey": "TUR", "turkiye": "TUR", "turkmenistan": "TKM", "turks and caicos": "TCA",
    "tuvalu": "TUV", "uae": "UAE", "uganda": "UGA", "ukraine": "UKR",
    "united arab emirates": "UAE", "united states": "USA", "uruguay": "URU", "us virgin islands": "VIR",
    "usa": "USA", "uzbekistan": "UZB", "vanuatu": "VAN", "venezuela": "VEN",
    "vietnam": "VIE", "wales": "WAL", "yemen": "YEM", "zambia": "ZAM",
    "zimbabwe": "ZIM",
}

# The FM export writes 'Nat' as a 3-letter code (ITA) but '2nd Nat' as a full
# country name (Italy), so a naive set check misses dual nationals entirely.
# Mapping the names back to codes recovers ~37k community-eligible players
# (Julián Álvarez, Lautaro, Valverde, Vinícius...) who hold an EU passport
# through their second nationality — decisive for non-EU roster limits.
NATION_NAME_TO_CODE = {
    "italy": "ITA", "france": "FRA", "spain": "ESP", "england": "ENG",
    "portugal": "POR", "germany": "GER", "netherlands": "NED", "belgium": "BEL",
    "scotland": "SCO", "wales": "WAL", "northern ireland": "NIR",
    "republic of ireland": "IRL", "ireland": "IRL", "poland": "POL",
    "croatia": "CRO", "serbia": "SRB", "greece": "GRE", "romania": "ROU",
    "bulgaria": "BUL", "sweden": "SWE", "denmark": "DEN", "norway": "NOR",
    "switzerland": "SUI", "austria": "AUT", "czech republic": "CZE",
    "slovakia": "SVK", "slovenia": "SVN", "hungary": "HUN", "finland": "FIN",
    "turkey": "TUR", "ukraine": "UKR", "russia": "RUS", "morocco": "MAR",
    "tunisia": "TUN", "albania": "ALB", "bosnia & herzegovina": "BIH",
    "north macedonia": "MKD", "montenegro": "MNE", "kosovo": "KOS",
    "georgia": "GEO", "iceland": "ISL", "luxembourg": "LUX", "malta": "MLT",
    "cyprus": "CYP", "lithuania": "LTU", "latvia": "LVA", "estonia": "EST",
    "moldova": "MDA", "algeria": "ALG", "andorra": "AND", "san marino": "SMR",
    "monaco": "MON", "liechtenstein": "LIE", "gibraltar": "GIB",
    "azerbaijan": "AZE", "armenia": "ARM", "belarus": "BLR",
}


def _nat2_code(name):
    """Map a full country name from '2nd Nat' to its FM 3-letter code."""
    if not isinstance(name, str) or not name.strip():
        return None
    return NATION_NAME_TO_CODE.get(name.strip().lower())


# preferred foot normalisation
FOOT_KEYWORDS = {
    "left-footed": "left", "left foot": "left", "αριστεροποδαρος": "left", "αριστερο ποδι": "left",
    "aristeropodaros": "left", "left footed": "left",
    "right-footed": "right", "right foot": "right", "δεξιοποδαρος": "right", "δεξι ποδι": "right",
    "deksiopodaros": "right", "right footed": "right",
    "two-footed": "either", "both feet": "either", "διποδος": "either", "and with both feet": "either",
}

# Characters that some fonts render as empty boxes (□). We transliterate only
# these specific ones — NOT common accents like é/ö/ü/ñ, which render fine — so
# names like "Emir Yazıcı" show as "Emir Yazici" instead of "Emir Yaz□c□".
_DISPLAY_TRANSLIT = {
    "ı": "i", "İ": "I", "ş": "s", "Ş": "S", "ğ": "g", "Ğ": "G",   # Turkish
    "ł": "l", "Ł": "L", "ż": "z", "Ż": "Z", "ź": "z", "Ź": "Z",   # Polish
    "ą": "a", "Ą": "A", "ę": "e", "Ę": "E", "ń": "n", "Ń": "N",   # Polish
    "đ": "d", "Đ": "D",                                            # Croatian/Serbian
    "ħ": "h", "Ħ": "H",                                            # Maltese
}


def _clean_display(s):
    if not isinstance(s, str):
        return s
    return "".join(_DISPLAY_TRANSLIT.get(ch, ch) for ch in s)


# Reference "today" for contract maths. The FM26 export was taken in the
# 2025-26 season, so many deals dated 30/6/2026 have already run out — those
# players are effectively free agents, which is exactly what scouts want to see.
TODAY = "2026-08-18"


def _contract_status(months_left):
    """Bucket months-remaining into a readable contract status."""
    if months_left is None or pd.isna(months_left):
        return "unknown"
    if months_left <= 0:
        return "expired"          # out of contract -> free signing
    if months_left <= 6:
        return "expiring"         # can negotiate a pre-contract / cheap deal
    if months_left <= 12:
        return "final year"
    return "under contract"


def _parse_fm_position(s):
    """Parse a raw FM position string into our role codes.

    FM writes best/secondary positions like 'ST (C)', 'D (L), M (C)' or
    'D/WB (RL)'. This expands them to our internal codes, e.g.
    'D/WB (RL)' -> ['DR','DL','WBR','WBL'], 'ST (C)' -> ['STC'], 'GK' -> ['GK'].
    """
    if not isinstance(s, str) or not s.strip():
        return []
    out = []
    for chunk in s.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        m = re.match(r"([A-Za-z/]+)\s*(?:\(([RLC]+)\))?", chunk)
        if not m:
            continue
        roles = m.group(1).split("/")     # e.g. ['D','WB']
        sides = m.group(2)                # e.g. 'RL' or None
        for role in roles:
            role = role.strip().upper()
            if not role:
                continue
            if sides:
                for side in sides:
                    out.append(f"{role}{side}")
            else:
                out.append(role)          # e.g. 'DM','ST','GK'
    seen, res = set(), []
    for p in out:
        if p not in seen:
            seen.add(p)
            res.append(p)
    return res


def load_players(path: str | Path | None = None) -> pd.DataFrame:
    """Load the cleaned FM26 dataset (works across pandas versions).

    Prefers the gzip-compressed file (fm26_final.csv.gz) if present, otherwise
    the plain CSV. pandas reads .gz transparently, so no data is lost — the
    gzip is a byte-for-byte lossless copy of the full 91-column dataset.

    The CSV stores `positions` as a pipe-joined string ('DC|DL'); rebuild it
    into a real list here so the agents can work with it directly.
    """
    if path is None:
        path = DATA_PATH_GZ if DATA_PATH_GZ.exists() else DATA_PATH
    df = pd.read_csv(path)
    # transliterate box-rendering characters in names/clubs for clean display
    for col in ("Name", "Club"):
        if col in df.columns:
            df[col] = df[col].apply(_clean_display)
    # rebuild positions column into lists (CSV stores it as 'DC|DL')
    def _to_list(s):
        if isinstance(s, list):
            return s
        if isinstance(s, str) and s.strip():
            return s.split("|")
        return []
    df["positions"] = df["positions"].apply(_to_list)
    # parse the FM "Best Pos" string into our role codes (for best-position-only
    # searches). Falls back to the full positions list if Best Pos is missing.
    if "Best Pos" in df.columns:
        df["best_pos_codes"] = df["Best Pos"].apply(_parse_fm_position)
        df["best_pos_codes"] = [
            bp if bp else pos
            for bp, pos in zip(df["best_pos_codes"], df["positions"])
        ]
    else:
        df["best_pos_codes"] = df["positions"]
    # ---- contract expiry -> real dates + months remaining ----
    # FM writes 'Expires' as d/m/YYYY. Knowing how long a deal has left is a
    # major scouting signal: an expired or nearly-expired contract means the
    # player can be signed free (Bosman) or for far less than his listed value.
    if "Expires" in df.columns:
        df["contract_expires"] = pd.to_datetime(
            df["Expires"], format="%d/%m/%Y", errors="coerce")
        today = pd.Timestamp(TODAY)
        df["contract_months_left"] = (
            (df["contract_expires"] - today).dt.days / 30.44).round(1)
        df["contract_status"] = df["contract_months_left"].apply(_contract_status)
    else:
        df["contract_expires"] = pd.NaT
        df["contract_months_left"] = float("nan")
        df["contract_status"] = "unknown"
    # ensure attribute columns are numeric
    for code in ALL_ATTR_CODES:
        if code in df.columns:
            df[code] = pd.to_numeric(df[code], errors="coerce")
    # second nationality -> code, so passport-based eligibility filters work
    if "2nd Nat" in df.columns:
        df["nat2_code"] = df["2nd Nat"].apply(_nat2_code)
    else:
        df["nat2_code"] = None
    # ---- true first-year cost: the fee alone understates a signing badly.
    # A cheap player on huge wages can cost more in year one than a pricier one
    # on modest terms, so the two are added into a single comparable figure.
    if "salary_eur" in df.columns and "value_mid" in df.columns:
        df["first_year_cost"] = (df["value_mid"].fillna(0)
                                 + df["salary_eur"].fillna(0))
        df.loc[df["value_mid"].isna() & df["salary_eur"].isna(),
               "first_year_cost"] = np.nan
    else:
        df["first_year_cost"] = np.nan
    # ---- overall ability: mean of the attributes that matter for the player's
    # own position type. Used to break suitability ties on merit rather than on
    # price — a tie broken by market value would keep burying cheap players from
    # smaller leagues under expensive ones, which is exactly the bargain a scout
    # is looking for.
    _gk_codes = ["Aer", "Cmd", "Com", "Ecc", "Han", "Kic", "1v1", "Pun", "Ref",
                 "TRO", "Thr"]
    gk_attrs = [c for c in _gk_codes if c in df.columns]
    out_attrs = [c for c in ALL_ATTR_CODES if c in df.columns and c not in _gk_codes]
    is_gk = df["positions"].apply(lambda L: "GK" in L)
    df["overall_ability"] = np.where(
        is_gk,
        df[gk_attrs].mean(axis=1) if gk_attrs else np.nan,
        df[out_attrs].mean(axis=1) if out_attrs else np.nan,
    ).round(2)
    return df


def resolve_attribute(word: str) -> str | None:
    """Map a human word ('pace', 'finishing') to an attribute code ('Pac')."""
    w = word.strip().lower()
    if w in SYNONYMS:
        return SYNONYMS[w]
    if w in NAME_TO_CODE:
        return NAME_TO_CODE[w]
    # try partial match against full names
    for name, code in NAME_TO_CODE.items():
        if w in name or name in w:
            return code
    return None


def resolve_position(label: str) -> list[str]:
    """Map a human position ('left winger') to role codes (['AML'])."""
    l = label.strip().lower()
    if l in POSITION_GROUPS:
        return POSITION_GROUPS[l]
    for key, codes in POSITION_GROUPS.items():
        if l in key or key in l:
            return codes
    # maybe they typed a raw code already
    up = label.strip().upper()
    return [up]


if __name__ == "__main__":
    df = load_players()
    print(f"Loaded {len(df)} players, {df.shape[1]} columns")
    print("resolve 'pace' ->", resolve_attribute("pace"))
    print("resolve 'left winger' ->", resolve_position("left winger"))
