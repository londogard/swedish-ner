import streamlit as st
from flair.data import Sentence
from flair.models import SequenceTagger
from flair.visual.ner_html import render_ner_html
from streamlit import caching
import streamlit.components.v1 as components


# Inspiration: https://github.com/sgoede/NER_POS_TAG/blob/main/NER_POS_TAG.py

# load tagger for POS and
# session_state = SessionState.get(tagger = None)
@st.cache(allow_output_mutation=True) # allow_output_mutation=True
def load_model():
    tagger = SequenceTagger.load('best-model-large-data.pt')
    return tagger

@st.cache(allow_output_mutation=True)
def predict(model, text):
    manual_sentence = Sentence(manual_user_input)
    model.predict(manual_sentence)
    return render_ner_html(manual_sentence)

st.title("Swedish Named Entity Recognition (NER) tagger")
st.subheader("Created by: Londogard AB (Hampus Londögård)")

#use_type = st.sidebar.radio("Please select the type of input you want to use",
#                             ("Manually type a sentence","Automatically tag an Excel file"))

# if use_type == "Manually type a sentence":
tagger = load_model()
st.title("Please type something in the box below")
manual_user_input = st.text_area("")
if len(manual_user_input) > 0:
    sentence = predict(tagger, manual_user_input)
    st.success("Below is your tagged string.")
    st.write(sentence, unsafe_allow_html=True)