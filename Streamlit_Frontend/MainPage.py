import streamlit as st

st.set_page_config(
    page_title="Main",
)

st.write("# Welcome to Food Recommendation System! 👋")

st.sidebar.success("Select a recommendation app.")

st.markdown(
    """
    A food recommendation web application developed with FastAPI and Streamlit.
    """
)
