import streamlit as st
import streamlit.components.v1 as components

# Set page title
st.set_page_config(
    page_title="Sign Language",
    initial_sidebar_state="expanded",
)
st.write('<style>div.row-widget.stMarkdown { font-size: 24px; }</style>', unsafe_allow_html=True)

st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)
components.html(
    """
    <style>
        #effect{
            margin:0px;
            padding:0px;
            font-family: "Source Sans Pro", sans-serif;
            font-size: max(8vw, 20px);
            font-weight: 700;
            top: 0px;
            right: 25%;
            position: fixed;
            background: -webkit-linear-gradient(0.25turn,blue,Lime);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        p{
            font-size: 2rem;
        }
    </style>
    <p id="effect">Sign Language Recognition</p>
    """,
    height=69,
)



def page_layout():
   
    st.write("sign language is an app developed by SVEC to bridge the communication gap between deaf and normal people")
    st.write("Developed By SVEC")
    st.markdown("## Benefits:")
    st.write("- Efficient Emergency Communication")
    st.write("- Accessible from anywhere, anytime")
    
    st.markdown("## Use:")
    st.write("- Communication Access for Deaf Individuals")
    st.write("- Sign language recognition is crucial in emergency situations, enabling clear communication for deaf individuals and ensuring they receive timely information ")
  
# Render page layout
page_layout()
