import asyncio

import streamlit as st


@st.cache_resource
def get_event_loop():
    """Crée et retourne un event loop asyncio qui persiste entre les rechargements de page"""
    return asyncio.new_event_loop()
