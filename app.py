import os
from flask import Flask, render_template, request
from translator import engine

app = Flask(__name__)


# Pre-load the translation index at startup so the first request isn't slow.
# This runs in the background once the Flask dev server starts.
def _preload():
    try:
        engine.load()
    except Exception as exc:
        print(f"[app] Warning: could not pre-load translation engine: {exc}")


@app.before_request
def ensure_engine():
    if not engine._ready:
        engine.load()


@app.route("/", methods=["GET"])
def home():
    return render_template("main.html")


@app.route("/translate", methods=["GET", "POST"])
def get_translation():
    if request.method == "POST":
        eng_sentence = request.form.get("input_text", "").strip()
        if not eng_sentence:
            return render_template("main.html", error="Please enter a sentence to translate.")
        try:
            result = engine.translate(eng_sentence)
        except Exception as exc:
            return render_template("main.html", error=f"Translation error: {exc}")

        return render_template(
            "result.html",
            original=eng_sentence,
            trans=result["translation"],
            method=result["method"],
            score=result["score"],
            matched_en=result.get("matched_en", ""),
        )

    return render_template("main.html")


if __name__ == "__main__":
    _preload()
    app.run(debug=True)
