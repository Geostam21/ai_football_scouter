"""
data.py — dataset loading and the attribute glossary.

The FM23 dataset uses short codes for attributes (Pac = Pace, Fin = Finishing).
The glossary below maps human words -> codes so the Requirements Agent can turn
a natural-language request into concrete column weights, and so the Reporting
Agent can turn codes back into readable text.
"""
import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).parent / "fm23_final.csv"
DATA_PATH_GZ = Path(__file__).parent / "fm23_final.csv.gz"

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
    "attacking midfielder": ["AMC"], "playmaker": ["AMC", "MC"],
    "striker": ["STC"], "forward": ["STC", "AMC"], "centre forward": ["STC"],
}

ALL_ATTR_CODES = list(ATTRIBUTES.keys())

# named league groups -> substrings that identify member leagues in 'Based'
LEAGUE_GROUPS = {
    "top 5": ["England (Premier Division)", "Spain (Primera", "Spain (LaLiga",
              "Italy (Serie A)", "Germany (Bundesliga)", "France (Ligue 1"],
    "top5": ["England (Premier Division)", "Spain (Primera", "Spain (LaLiga",
             "Italy (Serie A)", "Germany (Bundesliga)", "France (Ligue 1"],
    "premier league": ["England (Premier Division)"],
    "la liga": ["Spain (Primera", "Spain (LaLiga"],
    "serie a": ["Italy (Serie A)"],
    "bundesliga": ["Germany (Bundesliga)"],
    "ligue 1": ["France (Ligue 1"],
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
}

# preferred foot normalisation
FOOT_KEYWORDS = {
    "left-footed": "left", "left foot": "left", "αριστεροποδαρος": "left", "αριστερο ποδι": "left",
    "aristeropodaros": "left", "left footed": "left",
    "right-footed": "right", "right foot": "right", "δεξιοποδαρος": "right", "δεξι ποδι": "right",
    "deksiopodaros": "right", "right footed": "right",
    "two-footed": "either", "both feet": "either", "διποδος": "either", "and with both feet": "either",
}


def load_players(path: str | Path | None = None) -> pd.DataFrame:
    """Load the cleaned FM23 dataset (works across pandas versions).

    Prefers the gzip-compressed file (fm23_final.csv.gz) if present, otherwise
    the plain CSV. pandas reads .gz transparently, so no data is lost — the
    gzip is a byte-for-byte lossless copy of the full 91-column dataset.

    The CSV stores `positions` as a pipe-joined string ('DC|DL'); rebuild it
    into a real list here so the agents can work with it directly.
    """
    if path is None:
        path = DATA_PATH_GZ if DATA_PATH_GZ.exists() else DATA_PATH
    df = pd.read_csv(path)
    # rebuild positions column into lists (CSV stores it as 'DC|DL')
    def _to_list(s):
        if isinstance(s, list):
            return s
        if isinstance(s, str) and s.strip():
            return s.split("|")
        return []
    df["positions"] = df["positions"].apply(_to_list)
    # ensure attribute columns are numeric
    for code in ALL_ATTR_CODES:
        if code in df.columns:
            df[code] = pd.to_numeric(df[code], errors="coerce")
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
