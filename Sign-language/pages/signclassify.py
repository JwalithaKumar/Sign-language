import streamlit as st
import imagerec
import pandas as pd
import random
import streamlit.components.v1 as components
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="SignLanguageDetection",
    initial_sidebar_state="expanded",
)


st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)

uploaded_file = None

st.title("Sign Language Detection")

st.write('<style>div.row-widget.stMarkdown { font-size: 24px; }</style>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose a File", type=['jpg','png','jpeg'])


if uploaded_file!=None:
    st.image(uploaded_file)
x = st.button("Predict")
if x:
    with st.spinner("Thinking..."):
        model = load_model('./model/SignL.h5')

