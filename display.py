"""
display.py — pretty printing helpers for use in a notebook.

Turns the pipeline's output into readable text/HTML for Jupyter, so you can see
the shortlist summary first, then a full dashboard for each of the 10 players.
"""
from orchestrator import format_value


def print_shortlist(result: dict):
    """Compact summary table of the shortlist (text)."""
    print(f"Searched {result['pool_size']:,} matching players. Top {len(result['shortlist'])}:\n")
    print(f"{'#':>2}  {'Player':22} {'Age':>3}  {'Value':>9}  {'Fit':>5}  Club")
    print("-" * 70)
    for i, p in enumerate(result["shortlist"], 1):
        print(f"{i:>2}. {p['name'][:22]:22} {p['age']:>3}  "
              f"{format_value(p['value_eur']):>9}  {p['suitability']:>5}  {p.get('club','')}")


def print_dashboard(profile: dict):
    """Full profile for one player (text)."""
    print("=" * 70)
    pos = "/".join(profile["positions"])
    print(f"{profile['name']}  ({profile['age']}y · {pos} · {profile['club']})")
    tagline = profile.get("tagline")
    if tagline:
        print(f'"{tagline}"')
    phys = []
    if profile.get("height_cm"): phys.append(f"{profile['height_cm']}cm")
    if profile.get("weight_kg"): phys.append(f"{profile['weight_kg']}kg")
    if profile.get("foot"): phys.append(f"{profile['foot']} foot")
    if profile.get("nationality"): phys.append(profile["nationality"])
    print("  ".join(phys) + f"   |   {format_value(profile['value_eur'])}")
    print()
    print(profile["summary"])
    print()
    print("STRENGTHS:")
    for a, v in profile["top_attributes"][:5]:
        print(f"   + {a}: {v}")
    print("WEAKNESSES:")
    if profile["weaknesses"]:
        for a, v, p in profile["weaknesses"]:
            print(f"   - {a}: {v}  (bottom {p}% vs peers)")
    else:
        print("   (none notable — well-rounded)")
    print()


def show_all(pipeline, result):
    """Summary first, then every dashboard — the full scouting output."""
    print_shortlist(result)
    print("\n" + "#" * 70)
    print("FULL PROFILES")
    print("#" * 70 + "\n")
    for prof in pipeline.build_dashboards(result):
        print_dashboard(prof)


def dashboard_html(profile: dict) -> str:
    """Rich HTML card for one player (use with IPython.display.HTML)."""
    pos = "/".join(profile["positions"])
    strengths = "".join(
        f"<li>{a}: <b>{v}</b></li>" for a, v in profile["top_attributes"][:5]
    )
    if profile["weaknesses"]:
        weaks = "".join(
            f"<li>{a}: <b>{v}</b> <span style='color:#999'>(bottom {p}%)</span></li>"
            for a, v, p in profile["weaknesses"]
        )
    else:
        weaks = "<li style='color:#2a9d6f'>None notable — well-rounded</li>"

    bars = ""
    for cat, attrs in profile["attribute_groups"].items():
        bars += f"<div style='margin-top:8px;font-weight:600;font-size:13px'>{cat}</div>"
        for a in attrs[:5]:
            pct = a["percentile"] or 0
            color = "#2a9d6f" if pct >= 80 else ("#e0952b" if pct >= 50 else "#c0563a")
            bars += (
                f"<div style='display:flex;align-items:center;gap:8px;margin:3px 0'>"
                f"<span style='width:110px;font-size:12px'>{a['attr']}</span>"
                f"<div style='flex:1;height:7px;background:#eee;border-radius:4px'>"
                f"<div style='width:{pct}%;height:100%;background:{color};border-radius:4px'></div></div>"
                f"<span style='width:70px;font-size:11px;color:#666;text-align:right'>{a['value']} · top {100-pct}%</span>"
                f"</div>"
            )

    return f"""
    <div style='border:1px solid #ddd;border-radius:12px;padding:16px 20px;margin:12px 0;font-family:sans-serif'>
      <div style='display:flex;justify-content:space-between;align-items:baseline'>
        <div><span style='font-size:18px;font-weight:600'>{profile['name']}</span>
        <span style='color:#666;font-size:13px'> · {profile['age']}y · {pos} · {profile['club']}</span></div>
        <div style='font-size:18px;font-weight:600'>{format_value(profile['value_eur'])}</div>
      </div>
      <div style='color:#666;font-size:13px;margin:6px 0 10px'>{profile.get('tagline','')}</div>
      <div style='font-size:13px;line-height:1.5;margin-bottom:10px'>{profile['summary']}</div>
      <div style='display:flex;gap:24px;font-size:13px'>
        <div><div style='font-weight:600'>Strengths</div><ul style='margin:4px 0;padding-left:18px'>{strengths}</ul></div>
        <div><div style='font-weight:600'>Weaknesses</div><ul style='margin:4px 0;padding-left:18px'>{weaks}</ul></div>
      </div>
      <div style='margin-top:8px'>{bars}</div>
    </div>
    """
