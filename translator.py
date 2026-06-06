"""
Bidirectional English ↔ Amharic neural translation engine.

Uses Helsinki-NLP MarianMT models (trained on diverse multilingual data,
not Bible text):
    en → am :  Helsinki-NLP/opus-mt-en-am
    am → en :  Helsinki-NLP/opus-mt-am-en

Models are loaded lazily (only when first used) and cached in-process.
First use downloads the weights (~300 MB each) and caches them locally.

Install:
    pip3 install transformers sentencepiece torch
"""

from __future__ import annotations

MODELS = {
    "en-am": "Helsinki-NLP/opus-mt-en-am",
    "am-en": "Helsinki-NLP/opus-mt-am-en",
}

DIRECTION_LABELS = {
    "en-am": ("English", "Amharic"),
    "am-en": ("Amharic", "English"),
}


class TranslationEngine:
    def __init__(self):
        # one (tokenizer, model) pair per direction, loaded on demand
        self._loaded: dict[str, tuple] = {}

    def _get(self, direction: str):
        if direction not in MODELS:
            raise ValueError(f"Unknown direction: {direction!r}")
        if direction not in self._loaded:
            from transformers import MarianMTModel, MarianTokenizer
            name = MODELS[direction]
            print(f"[translator] Loading {name} (first run downloads ~300 MB) …")
            tok = MarianTokenizer.from_pretrained(name)
            mdl = MarianMTModel.from_pretrained(name)
            self._loaded[direction] = (tok, mdl)
            print(f"[translator] {direction} model ready.")
        return self._loaded[direction]

    def translate(self, text: str, direction: str = "en-am") -> dict:
        """
        direction: "en-am" (English→Amharic) or "am-en" (Amharic→English).
        Returns: {direction, source_lang, target_lang, original, translation}.
        """
        src_lang, tgt_lang = DIRECTION_LABELS.get(direction, ("", ""))
        text = (text or "").strip()
        if not text:
            return {"direction": direction, "source_lang": src_lang,
                    "target_lang": tgt_lang, "original": text, "translation": ""}

        import torch
        tok, mdl = self._get(direction)
        inputs = tok([text], return_tensors="pt", padding=True,
                     truncation=True, max_length=512)
        with torch.no_grad():
            out_ids = mdl.generate(**inputs, num_beams=4, max_length=256)
        translation = tok.decode(out_ids[0], skip_special_tokens=True)

        return {"direction": direction, "source_lang": src_lang,
                "target_lang": tgt_lang, "original": text, "translation": translation}


engine = TranslationEngine()
