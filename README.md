# English ↔ Amharic Translator

A web app for neural machine translation between English and Amharic, in both
directions.

It uses the pre-trained [Helsinki-NLP MarianMT](https://huggingface.co/Helsinki-NLP)
models, which are trained on diverse multilingual data:

| Direction | Model |
|-----------|-------|
| English → Amharic | `Helsinki-NLP/opus-mt-en-am` |
| Amharic → English | `Helsinki-NLP/opus-mt-am-en` |

## Run locally

Requires Python 3.8+ and an internet connection (to download the models on
first use).

```bash
git clone https://github.com/EdoLeta/English_Amharic_Translation.git
cd English_Amharic_Translation

# macOS / Linux use python3 / pip3
pip3 install -r requirements.txt
python3 app.py
```

Then open <http://localhost:5000>.

On the **first translation** in each direction, the corresponding model
(~300 MB) is downloaded and cached locally by the `transformers` library, so
later runs are fast.

## Project layout

```
app.py              Flask app (routes + direction handling)
translator.py       Bidirectional neural translation engine
templates/          main.html (input) + result.html (output)
static/css/         Styling
requirements.txt    Dependencies
Procfile            Gunicorn entry point (e.g. for Heroku)
```

## Notes

The original version of this project scraped Bible verses from JW.org to build
a parallel corpus. That Bible data and the corpus-retrieval approach have been
removed in favour of the pre-trained neural models above, which give far better
general-purpose translation quality.
