"""
Train a custom English→Amharic sequence-to-sequence model using the scraped corpus.

Usage:
    python train.py [--epochs 20] [--units 256] [--max-pairs 30000]

Saved artifacts (loaded automatically by app.py when present):
    model/          Keras SavedModel directory
    model/eng_tokenizer.pkl
    model/amh_tokenizer.pkl
    model/config.json
"""

import argparse
import json
import os
import pickle
import re
import unicodedata

import numpy as np
import pandas as pd


# ── Text cleaning ──────────────────────────────────────────────────────────

_VERSE_NUM = re.compile(r"^\d+\s+")          # strip leading verse numbers
_REFS      = re.compile(r"[\+\*]")            # strip reference markers
_SPACES    = re.compile(r"\s+")


def clean_en(text: str) -> str:
    text = _VERSE_NUM.sub("", text.strip())
    text = _REFS.sub("", text)
    text = text.lower()
    text = _SPACES.sub(" ", text)
    return text.strip()


def clean_am(text: str) -> str:
    text = _VERSE_NUM.sub("", text.strip())
    text = _REFS.sub("", text)
    text = unicodedata.normalize("NFC", text)
    text = _SPACES.sub(" ", text)
    return text.strip()


def add_tokens(text: str) -> str:
    return "<start> " + text + " <end>"


# ── Data loading ───────────────────────────────────────────────────────────

def load_pairs(csv_path: str, max_pairs: int) -> tuple[list, list]:
    df = pd.read_csv(csv_path).dropna()
    df = df.iloc[:max_pairs]
    en_sentences = [clean_en(str(s)) for s in df["English"]]
    am_sentences = [add_tokens(clean_am(str(s))) for s in df["Amharic"]]
    return en_sentences, am_sentences


# ── Model definition ───────────────────────────────────────────────────────

def build_model(src_vocab: int, tgt_vocab: int, units: int, embedding_dim: int):
    import tensorflow as tf
    from tensorflow import keras

    # Encoder
    enc_input = keras.Input(shape=(None,), name="enc_input")
    enc_emb   = keras.layers.Embedding(src_vocab, embedding_dim, mask_zero=True)(enc_input)
    enc_out, enc_h, enc_c = keras.layers.LSTM(
        units, return_sequences=True, return_state=True, name="encoder"
    )(enc_emb)

    # Decoder
    dec_input = keras.Input(shape=(None,), name="dec_input")
    dec_emb   = keras.layers.Embedding(tgt_vocab, embedding_dim, mask_zero=True)(dec_input)
    dec_lstm  = keras.layers.LSTM(units, return_sequences=True, return_state=True, name="decoder")
    dec_out, _, _ = dec_lstm(dec_emb, initial_state=[enc_h, enc_c])

    # Bahdanau-style attention
    attn_scores = keras.layers.Dot(axes=[2, 2])([dec_out, enc_out])
    attn_weights = keras.layers.Activation("softmax")(attn_scores)
    context = keras.layers.Dot(axes=[2, 1])([attn_weights, enc_out])
    concat  = keras.layers.Concatenate()([context, dec_out])

    output = keras.layers.Dense(tgt_vocab, activation="softmax", name="output")(concat)

    model = keras.Model([enc_input, dec_input], output)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ── Training ───────────────────────────────────────────────────────────────

def train(args):
    import tensorflow as tf
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    corpus_path = os.path.join(os.path.dirname(__file__), "data", "en_am.csv")
    print(f"Loading corpus from {corpus_path} …")
    en_sents, am_sents = load_pairs(corpus_path, args.max_pairs)

    # Tokenise
    eng_tok = Tokenizer(filters="", oov_token="<unk>")
    eng_tok.fit_on_texts(en_sents)

    amh_tok = Tokenizer(filters="", oov_token="<unk>")
    amh_tok.fit_on_texts(am_sents)

    eng_vocab = len(eng_tok.word_index) + 1
    amh_vocab = len(amh_tok.word_index) + 1
    print(f"English vocab: {eng_vocab}  |  Amharic vocab: {amh_vocab}")

    en_seq = pad_sequences(eng_tok.texts_to_sequences(en_sents), padding="post")
    am_seq = pad_sequences(amh_tok.texts_to_sequences(am_sents), padding="post")

    dec_input  = am_seq[:, :-1]
    dec_target = am_seq[:, 1:]

    model = build_model(eng_vocab, amh_vocab, args.units, args.embedding_dim)
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint("model/best.keras", save_best_only=True),
    ]

    model.fit(
        [en_seq, dec_input],
        dec_target[..., np.newaxis],
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=0.1,
        callbacks=callbacks,
    )

    # Save artifacts
    os.makedirs("model", exist_ok=True)
    model.save("model/seq2seq.keras")
    with open("model/eng_tokenizer.pkl", "wb") as f:
        pickle.dump(eng_tok, f)
    with open("model/amh_tokenizer.pkl", "wb") as f:
        pickle.dump(amh_tok, f)

    config = {
        "eng_stdlen": int(en_seq.shape[1]),
        "amh_stdlen": int(am_seq.shape[1]),
        "eng_vocab": eng_vocab,
        "amh_vocab": amh_vocab,
        "units": args.units,
        "start_token": amh_tok.word_index.get("<start>"),
        "end_token":   amh_tok.word_index.get("<end>"),
    }
    with open("model/config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("Training complete. Artifacts saved to model/")


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train English→Amharic seq2seq model")
    parser.add_argument("--epochs",        type=int, default=20)
    parser.add_argument("--units",         type=int, default=256)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--batch-size",    type=int, default=64)
    parser.add_argument("--max-pairs",     type=int, default=30000)
    train(parser.parse_args())
