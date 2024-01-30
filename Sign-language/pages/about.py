import streamlit as st
import streamlit.components.v1 as components


# Set page title
st.set_page_config(
    page_title="Sign Langusge is ....",
    initial_sidebar_state="expanded",
)
st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
img{border-radius: 10px;}
</style> """, unsafe_allow_html=True)
st.write('<style>div.row-widget.stMarkdown { font-size: 24px; }</style>', unsafe_allow_html=True)



st.markdown("## SignSense ")
st.write("Introducing SignSense, a revolutionary mobile app designed to bridge communication gaps for the deaf and hard-of-hearing community. SignSense utilizes cutting-edge technology to recognize and interpret sign language gestures in real-time, providing an instant and accurate translation into written or spoken language. With an intuitive user interface, this app empowers users to effortlessly communicate with those who may not be familiar with sign language. Whether in educational settings, social interactions, or everyday conversations, SignSense opens up a world of possibilities for inclusive and accessible communication. Break down barriers and embrace seamless communication with SignSense – connecting people through the universal language of signs.")
