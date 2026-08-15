# Deploying AI Football Scouter to Streamlit Community Cloud (free)

## 1. Put the code on GitHub
1. Create a free account at https://github.com if you don't have one.
2. Create a new **public** repository, e.g. `ai-football-scouter`.
3. Upload ALL files from this folder EXCEPT:
   - `.streamlit/secrets.toml` (never upload your key)
   - `__pycache__/`, `*.pkl`
   The included `.gitignore` already excludes these.
   Make sure you DO upload: `app.py`, all other `.py` files, `fm23_final.csv`,
   `requirements.txt`, `.streamlit/config.toml`.

   (Easiest: on the repo page, "Add file" -> "Upload files" -> drag everything in.)

## 2. Deploy
1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click "Create app" -> "Deploy a public app from GitHub".
3. Pick your repo, branch `main`, main file `app.py`. Click "Deploy".
4. First build takes a few minutes (installs packages, loads the dataset).

## 3. Add your Gemini key as a secret
1. In the app's page on Streamlit Cloud, open "Settings" -> "Secrets".
2. Paste this (with your real key):
   ```
   GEMINI_API_KEY = "your-gemini-api-key-here"
   ```
3. Save. The app restarts and now uses Gemini.

## 4. Share
Copy the app URL (looks like `https://your-app.streamlit.app`) and send it to
anyone — it opens in their browser, no install needed.

## Notes
- Free tier is fine for a handful of testers. Heavy simultaneous use can hit the
  Gemini free-tier rate limit; the app falls back to its rule-based mode if so.
- To update the app, just push new files to GitHub — it redeploys automatically.
