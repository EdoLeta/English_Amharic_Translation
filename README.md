# English ↔ Amharic Translator

A web app for translating between English and Amharic in both directions,
powered by the **Google Cloud Translation API** for high-quality results.

## Setup

### 1. Get a Google Cloud Translation API key

1. Go to <https://console.cloud.google.com/> and create (or pick) a project.
2. Enable the **Cloud Translation API**
   (APIs & Services → Library → search "Cloud Translation API" → Enable).
3. Create an API key
   (APIs & Services → Credentials → Create credentials → API key).

The free tier covers **500,000 characters/month**; beyond that it is billed
per character.

### 2. Install and run

Requires Python 3.8+.

```bash
git clone https://github.com/EdoLeta/English_Amharic_Translation.git
cd English_Amharic_Translation

pip3 install -r requirements.txt

# provide your API key (macOS / Linux):
export GOOGLE_TRANSLATE_API_KEY="your-key-here"
# Windows PowerShell:
#   $env:GOOGLE_TRANSLATE_API_KEY="your-key-here"

python3 app.py
```

Open <http://localhost:5000>. If the key isn't set, the app shows a setup
reminder instead of translating.

## Project layout

```
app.py              Flask app (routes + direction handling)
translator.py       Google Translate API client (bidirectional)
templates/          main.html (input) + result.html (output)
static/css/         Styling
requirements.txt    Dependencies
Procfile            Gunicorn entry point (e.g. for Heroku)
```

When deploying (e.g. Heroku), set `GOOGLE_TRANSLATE_API_KEY` as a config var
rather than hard-coding it.

## History

The original project scraped Bible verses from JW.org to build a parallel
corpus and translate from it. That gave poor, Bible-biased results, so it was
replaced — first with an open neural model, and now with the Google Cloud
Translation API, which gives the best Amharic quality.
