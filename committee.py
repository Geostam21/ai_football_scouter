"""
committee.py — a collaborative multi-agent "Scouting Committee".

Sits ON TOP of the deterministic pipeline and models how a real scouting
department reaches a verdict: several specialists look at the same candidate
from different angles, argue their case, and a head scout reconciles their
(often conflicting) views into a final recommendation.

Design principle: every NUMBER an agent argues over comes from the deterministic
tools — the ML value model, the team-fit analyser, the suitability engine. The
LLM is used for each agent's REASONING and the head scout's synthesis, never to
invent a figure. So the quantitative evidence is reproducible; only the
natural-language argumentation varies. A genuine multi-agent system (each agent
reasons in its own voice) with a trustworthy statistical core.

Agents:
    Coordinator        -> parses the request, builds the candidate pool
    Technical Scout    -> on-pitch quality, style, profile          (LLM + tools)
    Financial Analyst  -> value for money, budget realism           (LLM + tools)
    Tactical Fit       -> fit to the specific club/system           (LLM + tools)
    Head Scout         -> reconciles the three -> PURSUE/CONSIDER/PASS (LLM)
"""
from __future__ import annotations
import json
import pandas as pd

from llm import call_llm
from data import ATTRIBUTES


class _Specialist:
    """A committee member: computes evidence from tools, then argues via LLM."""
    name = "Specialist"
    system = "You are a football specialist."

    def _evidence(self, row, spec) -> dict:
        raise NotImplementedError

    def assess(self, row, spec) -> dict:
        ev = self._evidence(row, spec)
        from llm import _has_gemini
        if _has_gemini():
            prompt = (f"Player: {row.get('Name')}\n"
                      f"Evidence (from data tools, treat as fact):\n"
                      f"{json.dumps(ev['facts'], ensure_ascii=False)}\n\n"
                      f"Give your assessment in 1-2 sentences, from your role's "
                      f"angle only. Be direct about strengths and concerns.")
            try:
                argument = call_llm(prompt, system=self.system,
                                    json_mode=False).strip()
            except Exception:
                argument = ev["fallback"]
        else:
            # no live LLM -> use the deterministic, tool-derived sentence
            argument = ev["fallback"]
        return {"agent": self.name, "score": ev["score"],
                "verdict": ev["verdict"], "argument": argument,
                **ev.get("extra", {})}


class TechnicalScout(_Specialist):
    name = "Technical Scout"
    system = ("You are a Technical Scout. You care ONLY about on-pitch quality: "
              "attributes, playing style, and how well the player fits the "
              "requested profile. You ignore price and club politics. Be honest "
              "about technical ceiling and weaknesses. Plain text, no markdown.")

    def __init__(self, roles):
        self.roles = roles

    def _evidence(self, row, spec) -> dict:
        import math
        suit = row.get("_suitability", 0)
        try:
            suit = float(suit)
            if math.isnan(suit):
                suit = 0.0
        except (TypeError, ValueError):
            suit = 0.0
        style = row.get("style") if isinstance(row.get("style"), str) else None
        top = row.get("_top_attrs") or []
        weak = row.get("_weak_attrs") or []
        verdict = ("elite" if suit >= 85 else "strong" if suit >= 72
                   else "decent" if suit >= 60 else "limited")
        return {
            "score": round(suit), "verdict": verdict,
            "facts": {"suitability_0_100": round(suit), "style": style,
                      "standout_attributes": top[:4], "weak_attributes": weak[:3]},
            "fallback": f"Technically {verdict} for the brief"
                        + (f", a {style}." if style else "."),
            "extra": {"style": style},
        }


class FinancialAnalyst(_Specialist):
    name = "Financial Analyst"
    system = ("You are a Financial Analyst for a football club. You care about "
              "value for money and budget realism: is the fee justified by the "
              "model's valuation, is he a bargain or overpriced, does he fit the "
              "budget. You ignore tactics. Plain text, no markdown.")

    def __init__(self, value_model):
        self.vm = value_model

    def _evidence(self, row, spec) -> dict:
        import math

        def _num(v):
            try:
                v = float(v)
                return 0.0 if math.isnan(v) else v
            except (TypeError, ValueError):
                return 0.0

        listed = _num(row.get("value_mid"))
        pred = _num(row.get("_predicted"))
        ratio = (pred / listed) if listed > 0 else 0
        if ratio >= self.vm.bargain_hi:
            verdict, score = "bargain", 90
        elif ratio <= self.vm.bargain_lo and listed > 0:
            verdict, score = "overpriced", 30
        else:
            verdict, score = "fair value", 60
        budget = spec.get("max_value")
        affordable = (budget is None) or (listed <= budget)
        if not affordable:
            score = min(score, 25)
        return {
            "score": score, "verdict": verdict,
            "facts": {"listed_value_eur": int(listed),
                      "model_valuation_eur": int(pred),
                      "assessment": verdict, "within_budget": affordable,
                      "budget_eur": int(budget) if budget else None},
            "fallback": (f"Model values him at EUR{pred/1e6:.1f}M vs "
                         f"EUR{listed/1e6:.1f}M listed - {verdict}"
                         + ("" if affordable else ", and over budget.") + "."),
            "extra": {"affordable": affordable},
        }


class TacticalFitAnalyst(_Specialist):
    name = "Tactical Fit"
    system = ("You are a Tactical Fit analyst. You care ONLY about whether the "
              "player fits the target club's specific needs and system: does the "
              "squad need this position, is he an upgrade on current options, "
              "does he cover their weaknesses. You ignore price. Plain text.")

    def __init__(self, team_fit):
        self.tf = team_fit

    def _evidence(self, row, spec) -> dict:
        club = spec.get("club_name")
        if not club:
            return {"score": None, "verdict": "n/a",
                    "facts": {"note": "no target club specified"},
                    "fallback": "No target club specified, so I can't judge fit."}
        club_resolved = self.tf.find_club(club)
        if not club_resolved:
            return {"score": None, "verdict": "n/a",
                    "facts": {"note": f"club '{club}' not found"},
                    "fallback": f"I couldn't find the club '{club}'."}
        fit = self.tf.player_fit(row, club_resolved)
        s = fit.get("sporting_fit", 0)
        verdict = ("great" if s >= 75 else "useful" if s >= 55 else "marginal")
        return {
            "score": round(s), "verdict": verdict,
            "facts": {"club": club_resolved, "sporting_fit_0_100": round(s),
                      "positional_need": fit.get("need"),
                      "upgrade_on_squad": fit.get("upgrade"),
                      "fills_weak_spots": fit.get("gap_fit"),
                      "weak_areas_addressed": fit.get("weak_attr_names", [])[:3]},
            "fallback": (f"{verdict.title()} fit at {club_resolved}: need "
                         f"{fit.get('need')}, upgrade {fit.get('upgrade')}."),
            "extra": {"detail": fit},
        }


_HEAD_SYSTEM = """You are the Head Scout chairing a recruitment meeting. Three
specialists have each argued their case about a player: a Technical Scout
(on-pitch quality), a Financial Analyst (value for money), and a Tactical Fit
analyst (fit to the target club). Their 0-100 scores come from data - treat them
as given, never invent numbers.

Weigh the three arguments and reconcile them. EXPLICITLY name any disagreement
(e.g. technically excellent but overpriced, or a bargain who doesn't fit
tactically) and say how you resolve it. Finish with a clear call: PURSUE,
CONSIDER, or PASS. 2-4 sentences, plain text, no markdown."""


class HeadScout:
    def decide(self, player_name, assessments) -> dict:
        tech = next((a for a in assessments if a["agent"] == "Technical Scout"), {})
        fin = next((a for a in assessments if a["agent"] == "Financial Analyst"), {})
        tac = next((a for a in assessments if a["agent"] == "Tactical Fit"), {})
        parts, weights = [], []
        if tech.get("score") is not None:
            parts.append(tech["score"]); weights.append(0.45)
        if fin.get("score") is not None:
            parts.append(fin["score"]); weights.append(0.25)
        if tac.get("score") is not None:
            parts.append(tac["score"]); weights.append(0.30)
        agg = round(sum(p * w for p, w in zip(parts, weights)) / sum(weights)) \
            if parts else 0
        rec = "PURSUE" if agg >= 72 else "CONSIDER" if agg >= 55 else "PASS"

        debate = [{"agent": a["agent"], "score": a["score"],
                   "argument": a.get("argument", "")} for a in assessments]
        from llm import _has_gemini
        if _has_gemini():
            prompt = (f'Player: {player_name}\n'
                      f'Specialist arguments (JSON):\n'
                      f'{json.dumps(debate, ensure_ascii=False)}\n\n'
                      f'Aggregate score: {agg}/100. Suggested call: {rec}.\n'
                      f'Write your verdict, resolving any disagreement.')
            try:
                summary = call_llm(prompt, system=_HEAD_SYSTEM,
                                   json_mode=False).strip()
            except Exception:
                summary = _fallback_summary(player_name, tech, fin, tac, rec)
        else:
            summary = _fallback_summary(player_name, tech, fin, tac, rec)
        return {"player": player_name, "aggregate": agg,
                "recommendation": rec, "verdict": summary,
                "assessments": assessments}


def _fallback_summary(name, tech, fin, tac, rec) -> str:
    bits = [f"{name}:"]
    if tech:
        bits.append(f"technically {tech.get('verdict','?')}.")
    if fin:
        bits.append(f"Financially {fin.get('verdict','?')}.")
    if tac and tac.get("score") is not None:
        bits.append(f"{tac.get('verdict','?').title()} tactical fit.")
    bits.append(f"Recommendation: {rec}.")
    return " ".join(bits)


class ScoutingCommittee:
    """Coordinates the specialists over a shortlist and returns verdicts."""

    def __init__(self, pipeline):
        self.p = pipeline
        from ml import ValueModel
        self._vm = ValueModel(pipeline.players)
        self.technical = TechnicalScout(getattr(pipeline, "roles", None))
        self.financial = FinancialAnalyst(self._vm)
        self.tactical = TacticalFitAnalyst(pipeline.team_fit)
        self.dashboard = getattr(pipeline, "dashboard", None)

    def review(self, request: str, top_n: int = 5,
               club_name: str | None = None) -> dict:
        # 1. Coordinator: parse + build the candidate pool via the normal pipeline
        out = self.p.run(request)
        spec = dict(out["spec"])
        spec["club_name"] = (club_name or spec.get("team_fit_club")
                             or spec.get("club_name"))

        ranked = out["ranked_df"].head(top_n).copy()
        preds = self._vm.predict(ranked)

        verdicts = []
        for idx, row in ranked.iterrows():
            r = row.copy()
            r["_suitability"] = row.get("suitability", 0)
            r["_predicted"] = preds.loc[idx] if idx in preds.index else 0
            top_attrs, weak_attrs = [], []
            if self.dashboard is not None:
                try:
                    d = self.dashboard.build(row, with_summary=False)
                    top_attrs = d.get("top_attributes", [])
                    weak_attrs = d.get("weaknesses", [])
                    r["_top_attrs"] = [a for a, _ in top_attrs]
                    r["_weak_attrs"] = [a for a, *_ in weak_attrs]
                except Exception:
                    pass
            # 2. Specialists each argue their case (tool evidence + LLM reasoning)
            assessments = [
                self.technical.assess(r, spec),
                self.financial.assess(r, spec),
                self.tactical.assess(r, spec),
            ]
            # 3. Head Scout reconciles the debate
            v = HeadScout().decide(row["Name"], assessments)
            # carry display data so the UI can render a profile card + radar
            v["player_index"] = idx
            v["top_attributes"] = top_attrs
            v["weaknesses"] = weak_attrs
            v["positions"] = row.get("positions") if isinstance(
                row.get("positions"), list) else []
            v["club"] = row.get("Club")
            v["age"] = int(row["Age"]) if pd.notna(row.get("Age")) else None
            # acquisition data — what a director needs to actually sign him
            v["value_eur"] = (None if pd.isna(row.get("value_mid"))
                              else int(row["value_mid"]))
            v["salary_eur"] = (None if pd.isna(row.get("salary_eur"))
                               else int(row["salary_eur"]))
            v["predicted_eur"] = int(preds.loc[idx]) if idx in preds.index else None
            v["contract_status"] = row.get("contract_status")
            v["contract_expires"] = row.get("Expires")
            v["nat"] = row.get("Nat")
            v["nat2"] = row.get("nat2_code") if isinstance(
                row.get("nat2_code"), str) else None
            verdicts.append(v)

        verdicts.sort(key=lambda v: v["aggregate"], reverse=True)
        return {"request": request, "club": spec.get("club_name"),
                "verdicts": verdicts, "spec": spec}
