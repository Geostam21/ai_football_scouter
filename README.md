# AI Football Scouter

A multi-agent football scouting system: turns a natural-language request into a
ranked, explained shortlist from an FM26 dataset (176,477 players), with radar
dashboards, a value-prediction bargain detector, team-fit analysis, and player
similarity search. Dark/gold Streamlit UI.

## Run locally

```bash
pip install -r requirements.txt
set GEMINI_API_KEY=your_key      # Windows (optional; app also works without a key)
streamlit run app.py
```

Without a key the app still runs using a built-in rule-based fallback; with a
Gemini key it understands richer, conceptual requests.

## Agents & ML

- Requirements (LLM): request -> structured spec
- Candidate (code): hard-constraint filtering
- Scoring (code): weighted, normalised ranking
- Reporting (LLM): shortlist -> briefing
- Dashboard (code+LLM): per-player profile, percentiles, strengths/weaknesses
- Similarity (ML): "players like X" via nearest neighbours
- Value model (ML): league-adjusted value prediction -> bargain flag
- Team-fit: squad-gap analysis that tilts weights toward a club's needs

## Deploy free on Streamlit Community Cloud

See DEPLOY.md.
