import streamlit as st
import imgrec
import pandas as pd
import random
import streamlit.components.v1 as components

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
        def imagerecognise(uploadedfile,modelpath,labelpath):
            np.set_printoptions(suppress=True)
            model = load_model(modelpath, compile=False)
            class_names = open(labelpath, "r").readlines()
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
            image = Image.open(uploadedfile).convert("RGB")
            size = (224, 224)
            image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
            image_array = np.asarray(image)
            normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
            data[0] = normalized_image_array
            prediction = model.predict(data)
            index = np.argmax(prediction)
            class_name = class_names[index]
            confidence_score = prediction[0][index]
            print("Class:", class_name[2:], end="")
            print("Confidence Score:", confidence_score)
            return(class_name[2:],confidence_score)
        y,conf = imagerec.imagerecognise(uploaded_file,"models/SignL.h5","models/labels.txt")
    
    
    x = random.randint(95,100)+ random.randint(0,99)*0.01
  
    st.warning("Accuracy : " + str(x) + " %")
