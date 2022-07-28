import streamlit as st
from flair.data import Sentence
from flair.models import SequenceTagger
from flair.visual.ner_html import render_ner_html
import flair
from pathlib import Path
import torch

from onnxruntime import InferenceSession, SessionOptions
from transformers import AutoTokenizer
import numpy as np
from helper import (
    get_config,
    group_entities,
    render_ner_html_custom,
)
from urllib import request
from quant_flair_model import QuantizableLanguageModel
from transformers.models.bert.tokenization_bert import BertTokenizer

flair.models.LanguageModel = QuantizableLanguageModel
flair.models.language_model.LanguageModel = QuantizableLanguageModel
flair.device = torch.device("cpu")

IGNORE_LABELS = set(["O"])
config = get_config()
colors = {
    "PRS": "#F7FF53",  # YELLOW
    "PER": "#F7FF53",  # YELLOW
    "ORG": "#E8902E",  # ORANGE (darker)
    "LOC": "#FF40A3",  # PINK
    "MISC": "#4647EB",  # PURPLE
    "EVN": "#06b300",  # GREEN
    "MSR": "#FFEDD5",  # ORANGE (lighter)
    "TME": "#ff7398",  # PINK (pig)
    "WRK": "#c5ff73",  # YELLOW (REFLEX)
    "OBJ": "#4ed4b0",  # TURQUOISE
    "O": "#ddd",  # GRAY
}


# load tagger for POS and
@st.experimental_memo
def load_flair_model():
    tagger = SequenceTagger.load("londogard/flair-swe-ner")
    q_tagger = torch.quantization.quantize_dynamic(
        tagger, {torch.nn.LSTM, torch.nn.Linear}, dtype=torch.qint8
    )
    del tagger
    return q_tagger


@st.experimental_memo
def predict_flair(_model, text):
    manual_sentence = Sentence(manual_user_input)
    _model.predict(manual_sentence)
    return render_ner_html(manual_sentence, colors=colors, wrap_page=False)


# load tagger for POS and
@st.experimental_singleton
def load_model():
    if not Path("kb-bert-cased-ner-optimized-quantized.onnx").is_file():
        request.urlretrieve(
            "https://www.dropbox.com/s/bjr14jw6n2o3dmu/kb-bert-cased-ner-optimized-quantized.onnx?dl=1",
            "kb-bert-cased-ner-optimized-quantized.onnx",
        )
    onnx_options = SessionOptions()
    session = InferenceSession(
        "kb-bert-cased-ner-optimized-quantized.onnx",
        onnx_options,
        providers=["CPUExecutionProvider"],
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "KB/bert-base-swedish-cased-ner", use_fast=False
    )
    return session, tokenizer


@st.experimental_memo
def predict(_session: InferenceSession, _tokenizer: BertTokenizer, text):
    tokens = _tokenizer(text, return_attention_mask=True, return_tensors="pt")
    inputs_onnx = {k: np.atleast_2d(v) for k, v in tokens.items()}
    entities = _session.run(None, inputs_onnx)[0].squeeze(0)

    input_ids = tokens["input_ids"][0]
    score = np.exp(entities) / np.exp(entities).sum(-1, keepdims=True)
    labels_idx = score.argmax(axis=-1)

    entities = []
    # Filter to labels not in `self.ignore_labels`
    filtered_labels_idx = [
        (idx, label_idx)
        for idx, label_idx in enumerate(labels_idx)
        if config["id2label"][str(label_idx)] not in IGNORE_LABELS
    ]

    for idx, label_idx in filtered_labels_idx:
        entity = {
            "word": _tokenizer.convert_ids_to_tokens(int(input_ids[idx])),
            "score": score[idx][label_idx].item(),
            "entity": config["id2label"][str(label_idx)],
            "index": idx,
        }

        entities += [entity]
    answers = []

    answers += [group_entities(entities, _tokenizer)]

    answers = answers[0] if len(answers) == 1 else answers
    return render_ner_html_custom(text, answers, colors=colors)


session, tokenizer = load_model()
flair_model = load_flair_model()

st.title("Swedish Named Entity Recognition (NER) tagger")
st.subheader("Created with ❤️ by [Londogard](https://londogard.com) (Hampus Londögård)")
model = st.radio(
    "Select which model to use",
    ("Flair (F1: 85.6 - faster & 80MB)", "Bert (F1: 92.0 - 120MB)"),
)
st.title("Please type something in the box below")
manual_user_input = st.text_area("", "Hampus bor i Skåne!")

if len(manual_user_input) > 0:
    if model.startswith("Flair"):
        sentence = predict_flair(flair_model, manual_user_input)
    else:
        sentence = predict(session, tokenizer, manual_user_input)
    st.success("Below is your tagged string.")
    st.write(sentence, unsafe_allow_html=True)
