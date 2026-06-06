"""
Bidirectional English ↔ Amharic translation via the Google Cloud
Translation API (v2 REST endpoint, API-key auth).

Setup
-----
1. Create a project at https://console.cloud.google.com/
2. Enable "Cloud Translation API".
3. Create an API key (APIs & Services → Credentials → Create credentials → API key).
4. Expose it to the app:

       export GOOGLE_TRANSLATE_API_KEY="your-key-here"

   (On Windows PowerShell:  $env:GOOGLE_TRANSLATE_API_KEY="your-key-here")

The free tier covers 500,000 characters/month; beyond that it is billed
per character.
"""

from __future__ import annotations
import os

import requests

_ENDPOINT = "https://translation.googleapis.com/language/translate/v2"
_API_KEY_ENV = "GOOGLE_TRANSLATE_API_KEY"

# direction -> (source code, target code, source label, target label)
DIRECTIONS = {
    "en-am": ("en", "am", "English", "Amharic"),
    "am-en": ("am", "en", "Amharic", "English"),
}

# kept for backwards compatibility with app.py / templates
DIRECTION_LABELS = {k: (v[2], v[3]) for k, v in DIRECTIONS.items()}


class TranslationError(Exception):
    pass


class TranslationEngine:
    def __init__(self, timeout: int = 15):
        self._timeout = timeout

    @staticmethod
    def _api_key() -> str:
        key = os.environ.get(_API_KEY_ENV, "").strip()
        if not key:
            raise TranslationError(
                f"No API key found. Set the {_API_KEY_ENV} environment variable "
                f"to your Google Cloud Translation API key. See README.md for steps."
            )
        return key

    def translate(self, text: str, direction: str = "en-am") -> dict:
        """
        direction: "en-am" (English→Amharic) or "am-en" (Amharic→English).
        Returns: {direction, source_lang, target_lang, original, translation}.
        """
        if direction not in DIRECTIONS:
            direction = "en-am"
        src_code, tgt_code, src_lang, tgt_lang = DIRECTIONS[direction]

        text = (text or "").strip()
        if not text:
            return {"direction": direction, "source_lang": src_lang,
                    "target_lang": tgt_lang, "original": text, "translation": ""}

        params = {"key": self._api_key()}
        payload = {
            "q": text,
            "source": src_code,
            "target": tgt_code,
            "format": "text",
        }

        try:
            resp = requests.post(_ENDPOINT, params=params, data=payload,
                                 timeout=self._timeout)
        except requests.RequestException as exc:
            raise TranslationError(f"Network error contacting Google Translate: {exc}")

        if resp.status_code != 200:
            # surface Google's error message if present
            try:
                msg = resp.json()["error"]["message"]
            except Exception:
                msg = resp.text[:200]
            raise TranslationError(f"Google Translate API error ({resp.status_code}): {msg}")

        try:
            translation = resp.json()["data"]["translations"][0]["translatedText"]
        except (KeyError, IndexError, ValueError) as exc:
            raise TranslationError(f"Unexpected API response: {exc}")

        return {"direction": direction, "source_lang": src_lang,
                "target_lang": tgt_lang, "original": text, "translation": translation}


engine = TranslationEngine()
