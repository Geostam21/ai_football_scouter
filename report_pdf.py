"""
report_pdf.py — generate a full scouting report PDF from a search result.

Includes: the request summary, the ranked shortlist table, and a profile block
per top player (key data, strengths, weaknesses, value vs predicted). Themed to
match the dark/gold UI on a light page (readable when printed).
"""
from __future__ import annotations
import io
import os
import unicodedata
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, PageBreak)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

GOLD = colors.HexColor("#b8973f")
DARK = colors.HexColor("#242429")
MUTED = colors.HexColor("#666666")

# Register a Unicode font (DejaVu Sans) so the PDF can render Greek, Turkish,
# and Central/Eastern European names correctly instead of showing boxes.
_FONT = "Helvetica"          # fallback
_FONT_BOLD = "Helvetica-Bold"
_here = os.path.dirname(__file__)
try:
    _reg = os.path.join(_here, "DejaVuSans.ttf")
    _bold = os.path.join(_here, "DejaVuSans-Bold.ttf")
    if os.path.exists(_reg):
        pdfmetrics.registerFont(TTFont("DejaVu", _reg))
        _FONT = "DejaVu"
        if os.path.exists(_bold):
            pdfmetrics.registerFont(TTFont("DejaVu-Bold", _bold))
            _FONT_BOLD = "DejaVu-Bold"
        else:
            _FONT_BOLD = "DejaVu"
except Exception:
    pass


def _pdf_safe(s):
    """Normalise text so the PDF font renders it correctly.

    Converts to NFC (precomposed) form so that a letter+combining-accent pair
    like "η"+◌́ becomes a single "ή" glyph the font can draw — otherwise the
    stray combining accent shows as a box. With the Unicode font registered the
    text otherwise passes through; if the font failed to load we drop non
    Latin-1 chars so the PDF still builds.
    """
    if not isinstance(s, str):
        return s
    s = unicodedata.normalize("NFC", s)
    if _FONT != "Helvetica":
        return s  # Unicode font handles everything
    return s.encode("latin-1", "ignore").decode("latin-1")


def _styles():
    ss = getSampleStyleSheet()
    ss.add(ParagraphStyle("H1x", parent=ss["Title"], textColor=GOLD, fontSize=20,
                          fontName=_FONT_BOLD))
    ss.add(ParagraphStyle("H2x", parent=ss["Heading2"], textColor=DARK, fontSize=13,
                          fontName=_FONT_BOLD))
    ss.add(ParagraphStyle("Small", parent=ss["Normal"], textColor=MUTED, fontSize=8,
                          fontName=_FONT))
    ss.add(ParagraphStyle("Body", parent=ss["Normal"], fontSize=9.5, leading=13,
                          fontName=_FONT))
    return ss


def _profile_text(d: dict, format_value) -> str:
    """Build a fuller multi-sentence profile from deterministic dashboard data
    (no LLM, so it stays fast for all 10 players)."""
    name = d["name"]
    age = d.get("age", "")
    style = d.get("tagline") or "versatile player"
    pos = "/".join(d.get("positions", [])) or "several roles"
    foot = d.get("foot")
    foot_desc = ""
    if isinstance(foot, str) and foot.strip():
        fl = foot.lower()
        if "left" in fl and "right" not in fl:
            foot_desc = "left-footed "
        elif "right" in fl and "left" not in fl:
            foot_desc = "right-footed "
        elif "either" in fl or "both" in fl:
            foot_desc = "two-footed "
    tops = d.get("top_attributes", [])
    strengths = [a for a, _ in tops[:3]]
    more = [a for a, _ in tops[3:5]]
    weaks = [a for a, _, _ in d.get("weaknesses", [])[:2]]

    val = d.get("value_eur")
    pred = d.get("predicted_value")
    parts = []
    parts.append(f"{name} is a {age}-year-old {foot_desc}{str(style).lower()} operating in {pos}.")
    if strengths:
        parts.append("The standout qualities are "
                     + ", ".join(strengths[:-1]) + (f" and {strengths[-1]}" if len(strengths) > 1 else strengths[0])
                     + ", which define how this player influences a game.")
    if more:
        parts.append("There is further quality in " + " and ".join(more) + ".")
    if weaks:
        parts.append("Relative to positional peers, the profile is weaker in "
                     + " and ".join(weaks) + ", which should be weighed against the intended role.")
    # value framing
    if val is not None and pred is not None:
        try:
            if pred >= val * 1.3 and val > 0:
                parts.append(f"On value, the model rates the player around {format_value(pred)} "
                             f"versus a listed {format_value(val)} — a potential bargain.")
            elif val > 0 and pred <= val * 0.7:
                parts.append(f"On value, the listed {format_value(val)} sits above the model's "
                             f"estimate of {format_value(pred)}, so the price looks full.")
            else:
                parts.append(f"The listed value ({format_value(val)}) is broadly in line with the "
                             f"model estimate ({format_value(pred)}).")
        except Exception:
            pass
    parts.append("Overall, a profile worth a closer look for teams prioritising these attributes.")
    return " ".join(parts)


def build_report(result: dict, request: str, readable: str,
                 format_value, dashboards: list[dict],
                 team_note: str | None = None) -> bytes:
    """Return PDF bytes for a scouting report."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            topMargin=18 * mm, bottomMargin=16 * mm,
                            leftMargin=16 * mm, rightMargin=16 * mm)
    ss = _styles()
    story = []

    # header
    story.append(Paragraph("AI Football Scouter — Scouting Report", ss["H1x"]))
    story.append(Paragraph(f'Request: "{_pdf_safe(request)}"', ss["Body"]))
    story.append(Paragraph(f"Interpreted as: {_pdf_safe(readable)}", ss["Small"]))
    if team_note:
        story.append(Paragraph(_pdf_safe(team_note), ss["Small"]))
    story.append(Paragraph(f"Searched {result['pool_size']:,} matching players.", ss["Small"]))
    story.append(Spacer(1, 8))

    # shortlist table
    story.append(Paragraph("Shortlist", ss["H2x"]))
    header = ["#", "Player", "Age", "Club", "Nat", "Value", "Predicted", "Fit"]
    rows = [header]
    for i, (p, d) in enumerate(zip(result["shortlist"], dashboards), 1):
        club = p.get("club")
        club = "Free agent" if not isinstance(club, str) or not club.strip() else club[:18]
        rows.append([
            str(i), _pdf_safe(p["name"]), str(p["age"]), _pdf_safe(club),
            d.get("nationality", "") or "",
            format_value(p["value_eur"]),
            format_value(d.get("predicted_value")),
            f"{p['suitability']:.0f}",
        ])
    t = Table(rows, repeatRows=1, colWidths=[8*mm, 40*mm, 10*mm, 34*mm, 12*mm, 24*mm, 24*mm, 12*mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), DARK),
        ("TEXTCOLOR", (0, 0), (-1, 0), GOLD),
        ("FONTNAME", (0, 0), (-1, 0), _FONT_BOLD),
        ("FONTNAME", (0, 1), (-1, -1), _FONT),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f2ec")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(t)
    story.append(Spacer(1, 10))

    # per-player profiles (top players)
    story.append(Paragraph("Player Profiles", ss["H2x"]))
    for d in dashboards:
        story.append(Spacer(1, 4))
        name = _pdf_safe(d["name"])
        club = d.get("club")
        club = "free agent" if not isinstance(club, str) or not club.strip() else _pdf_safe(club)
        foot = d.get("foot")
        foot_str = f" · {foot} foot" if isinstance(foot, str) and foot.strip() else ""
        hc = d.get("height_cm"); wk = d.get("weight_kg")
        phys = ""
        if hc:
            phys = f" · {hc}cm"
            if wk:
                phys += f"/{wk}kg"
        meta = f"{d.get('age','')} · {'/'.join(d.get('positions', []))} · {club} · {d.get('nationality','')}{foot_str}{phys}"
        story.append(Paragraph(f"<b>{name}</b>  <font size=8 color='#666666'>{_pdf_safe(meta)}</font>", ss["Body"]))

        # build a fuller narrative from the deterministic data
        story.append(Paragraph(_pdf_safe(_profile_text(d, format_value)), ss["Body"]))
        strengths = ", ".join(f"{a} {v}" for a, v in d.get("top_attributes", [])[:5])
        story.append(Paragraph(f"<b>Strengths:</b> {strengths}", ss["Small"]))
        if d.get("weaknesses"):
            weak = ", ".join(f"{a} {v}" for a, v, _ in d["weaknesses"][:4])
            story.append(Paragraph(f"<b>Weaknesses:</b> {weak}", ss["Small"]))
        story.append(Spacer(1, 8))

    story.append(Spacer(1, 10))
    story.append(Paragraph("Generated by AI Football Scouter — data: FM23 dataset. "
                           "Predicted value is a model estimate (league-adjusted).",
                           ss["Small"]))

    doc.build(story)
    buf.seek(0)
    return buf.read()
