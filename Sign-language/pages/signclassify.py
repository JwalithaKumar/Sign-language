import streamlit as st
import imagerec
import pandas as pd
import random
import os
import streamlit.components.v1 as components
from tensorflow.keras.models import load_model 
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input
#import subprocess
#if not os.path.isfile('Sign-language/models/SignL.h5'):
    #subprocess.run(['curl --output SignL.h5 "https://media.githubusercontent.com/media/JwalithaKumar/Sign-language/main/sep_5.h5"'], shell=True)


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
        model = load_model('Sign-language/models/SignL.h5', compile = False)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        predictions = model.predict(img_array)
        


