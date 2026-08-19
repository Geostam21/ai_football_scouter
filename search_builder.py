"""
search_builder.py — an optional guided panel that helps a user *construct* a
scouting prompt without needing to know the phrasing. It sits above the free
chat box: beginners fill a few fields, experienced users ignore it and type
directly. The panel only ever produces a normal text prompt, which then goes
through exactly the same pipeline as a typed one — so there is no second code
path to keep in sync.
"""
from __future__ import annotations

# position label -> (code used in prompt, list of style archetypes)
POSITIONS = {
    "Striker": ("striker", ["Poacher", "Target Man", "Complete Forward",
                            "Pressing Forward"]),
    "Right winger": ("right winger", ["Inside Forward", "Classic Winger",
                                      "Wide Playmaker", "Work-rate Wideman"]),
    "Left winger": ("left winger", ["Inside Forward", "Classic Winger",
                                    "Wide Playmaker", "Work-rate Wideman"]),
    "Attacking midfielder (C)": ("attacking midfielder",
                                 ["Advanced Playmaker", "Shadow Striker",
                                  "Trequartista"]),
    "Attacking midfielder (R)": ("right attacking midfielder",
                                 ["Advanced Playmaker", "Shadow Striker",
                                  "Trequartista"]),
    "Attacking midfielder (L)": ("left attacking midfielder",
                                 ["Advanced Playmaker", "Shadow Striker",
                                  "Trequartista"]),
    "Central midfielder": ("central midfielder",
                           ["Deep-lying Playmaker", "Box-to-Box", "Ball-winner"]),
    "Defensive midfielder": ("defensive midfielder",
                             ["Regista", "Anchor", "Ball-winning DM"]),
    "Centre-back": ("centre back", ["Ball-playing Defender", "No-nonsense Stopper",
                                   "Pace CB"]),
    "Right-back": ("right back", ["Attacking Full-back", "Wing-back",
                                 "Defensive Full-back"]),
    "Left-back": ("left back", ["Attacking Full-back", "Wing-back",
                               "Defensive Full-back"]),
    "Goalkeeper": ("goalkeeper", []),
}

# common qualities offered as quick-pick chips -> phrase inserted in the prompt
QUALITIES = {
    "Finishing": "good finishing",
    "Pace": "quick",
    "Dribbling": "good dribbling",
    "Passing": "good passing",
    "Vision": "great vision",
    "Tackling": "good tackling",
    "Heading": "strong in the air",
    "Strength": "physically strong",
    "Crossing": "good crossing",
    "Work rate": "high work rate",
    "Composure": "composed",
    "Leadership": "a leader",
}

# UI text in both languages (labels only; the generated prompt stays English so
# the parser — which is strongest in English — gets the cleanest input, while
# still accepting Greek if the user types it themselves).
UI_TEXT = {
    "en": {
        "title": "Build your search",
        "intro": "Fill in what you can — or just type in the box below.",
        "position": "Position",
        "style": "Playing style (optional)",
        "qualities": "Key qualities (pick any)",
        "age": "Max age",
        "budget": "Budget (€M, transfer value)",
        "similar": "Similar to a player (optional)",
        "club": "Scout for a club (optional)",
        "contract": "Contract status",
        "foot": "Preferred foot",
        "nationality": "Nationality (optional)",
        "min_height": "Min height (cm)",
        "height_on": "Set a minimum height",
        "any": "Any",
        "search": "Search",
        "preview": "The player you're after",
        "contract_opts": ["Any", "Free agents / expired", "Expiring soon"],
        "foot_opts": ["Any", "Right", "Left", "Either"],
        "similar_note": "Heads-up: when you pick a player to be similar to, the "
                        "search finds his closest matches by playing profile — "
                        "the other filters (age, budget, club) are only applied "
                        "on top where they can be combined.",
        "chat_placeholder": "Free-text search — type anything",
    },
    "el": {
        "title": "Δόμησε την αναζήτησή σου",
        "intro": "Συμπλήρωσε ό,τι θες — ή γράψε ελεύθερα πιο κάτω.",
        "position": "Θέση",
        "style": "Στυλ παιχνιδιού (προαιρετικό)",
        "qualities": "Χαρακτηριστικά (διάλεξε όσα θες)",
        "age": "Μέγιστη ηλικία",
        "budget": "Προϋπολογισμός (€εκατ., αξία)",
        "similar": "Να μοιάζει με παίκτη (προαιρετικό)",
        "club": "Για ποιον σύλλογο (προαιρετικό)",
        "contract": "Κατάσταση συμβολαίου",
        "foot": "Προτιμώμενο πόδι",
        "nationality": "Εθνικότητα (προαιρετικό)",
        "min_height": "Ελάχιστο ύψος (εκ.)",
        "height_on": "Όρισε ελάχιστο ύψος",
        "any": "Οποιαδήποτε",
        "search": "Αναζήτηση",
        "preview": "Ο παίκτης που ψάχνεις",
        "contract_opts": ["Οποιαδήποτε", "Ελεύθεροι / ληγμένο", "Λήγει σύντομα"],
        "foot_opts": ["Οποιοδήποτε", "Δεξί", "Αριστερό", "Και τα δύο"],
        "similar_note": "Σημείωση: όταν διαλέγεις παίκτη για να του μοιάζει, η "
                        "αναζήτηση βρίσκει τους πιο κοντινούς σε προφίλ παιχνιδιού "
                        "— τα υπόλοιπα φίλτρα (ηλικία, budget, σύλλογος) "
                        "εφαρμόζονται μόνο επιπρόσθετα όπου συνδυάζονται.",
        "chat_placeholder": "Ελεύθερη αναζήτηση — γράψε ό,τι θες",
    },
}


def build_prompt(position=None, style=None, qualities=None, max_age=None,
                 budget=None, similar_to=None, club=None, contract_idx=0,
                 foot=None, nationality=None, min_height=None) -> str:
    """Assemble a natural-language prompt from the guided fields.

    Kept intentionally simple and English-only: the point is to feed the
    existing parser a clean, well-formed sentence, not to invent new syntax.
    """
    # "similar to X" routes to the similarity engine, which finds the closest
    # players by profile. Fields the engine CAN also honour as hard filters
    # (age, budget, height, foot, nationality, contract, club) are appended so
    # the guided panel isn't silently dropping them; style/position/qualities
    # don't apply because the reference player already defines the profile.
    if similar_to:
        base = f"players like {similar_to.strip()}"
        if max_age:
            base += f" under {int(max_age)}"
        if budget:
            base += f" budget {int(budget)}M"
        if min_height:
            base += f" over {int(min_height)}cm"
        if foot and foot.lower() in ("right", "left"):
            base += f" {foot.lower()}-footed"
        elif foot and foot.lower() == "either":
            base += " two-footed"
        if nationality:
            low = nationality.strip().lower()
            if low in ("community", "eu-eligible", "κοινοτικος", "κοινοτικός"):
                base += " community player"
            elif low in ("eu", "eu national", "european"):
                base += " EU national"
            else:
                base += f" from {nationality.strip()}"
        if contract_idx == 1:
            base += " out of contract"
        elif contract_idx == 2:
            base += " with expiring contract"
        if club:
            base += f" for {club.strip()}"
        return base

    parts = []
    pos = POSITIONS.get(position, (None, []))[0] if position else None
    if style and pos:
        # Emit BOTH the style and the position so each parser picks up its own
        # signal: the style filter matches the archetype, and the position
        # (incl. its left/right side) still narrows the role. Concatenating them
        # into one phrase ("left-sided defensive full-back") parses as neither,
        # so we keep the style word followed by the position.
        side = next((s for s in ("left", "right") if pos.startswith(s)), None)
        if side and side not in style.lower():
            parts.append(f"{style.lower()} {pos}")   # "defensive full-back left back"
        else:
            parts.append(style.lower())
    elif style:
        parts.append(style.lower())
    elif pos:
        parts.append(pos)
    else:
        parts.append("player")

    quals = qualities or []
    qual_phrases = [QUALITIES[q] for q in quals if q in QUALITIES]

    sentence = " ".join(parts) if parts else "player"
    if foot and foot.lower() in ("right", "left"):
        sentence = f"{foot.lower()}-footed " + sentence
    elif foot and foot.lower() == "either":
        sentence = "two-footed " + sentence
    if qual_phrases:
        sentence += " with " + ", ".join(qual_phrases)
    if min_height:
        sentence += f" over {int(min_height)}cm"
    if max_age:
        sentence += f" under {int(max_age)}"
    if budget:
        sentence += f" budget {int(budget)}M"
    if nationality:
        n = nationality.strip()
        low = n.lower()
        if low in ("community", "eu-eligible", "κοινοτικος", "κοινοτικός"):
            sentence += " community player"
        elif low in ("eu", "eu national", "european", "ε.ε.", "ευρωπαιος"):
            sentence += " EU national"
        else:
            sentence += f" from {n}"
    if contract_idx == 1:
        sentence += " out of contract"
    elif contract_idx == 2:
        sentence += " with expiring contract"
    if club:
        sentence += f" for {club.strip()}"
    return sentence
