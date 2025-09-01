import streamlit as st, sys, platform
st.title("Smoke Test")
st.write("Python:", sys.version)
st.write("Platform:", platform.platform())
st.success("If you see this page, the server is healthy.")
