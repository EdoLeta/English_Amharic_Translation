import os
from flask import Flask, render_template, request
from translator import engine, DIRECTION_LABELS

app = Flask(__name__)

VALID_DIRECTIONS = set(DIRECTION_LABELS.keys())


def _api_key_set() -> bool:
    return bool(os.environ.get("GOOGLE_TRANSLATE_API_KEY", "").strip())


@app.route("/", methods=["GET"])
def home():
    return render_template("main.html", direction="en-am",
                           api_key_missing=not _api_key_set())


@app.route("/translate", methods=["GET", "POST"])
def get_translation():
    if request.method != "POST":
        return render_template("main.html", direction="en-am",
                               api_key_missing=not _api_key_set())

    direction = request.form.get("direction", "en-am")
    if direction not in VALID_DIRECTIONS:
        direction = "en-am"

    text = request.form.get("input_text", "").strip()
    if not text:
        return render_template("main.html", direction=direction,
                               api_key_missing=not _api_key_set(),
                               error="Please enter some text to translate.")

    try:
        result = engine.translate(text, direction=direction)
    except Exception as exc:
        return render_template("main.html", direction=direction,
                               api_key_missing=not _api_key_set(),
                               error=str(exc))

    return render_template("result.html", **result)


if __name__ == "__main__":
    app.run(debug=True)
