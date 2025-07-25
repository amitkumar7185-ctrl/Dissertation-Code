import streamlit as st

def create_output_container():
    """Create a container that can be cleared when switching views"""
    if 'output_container' not in st.session_state:
        st.session_state.output_container = st.empty()
    return st.session_state.output_container

def clear_output():
    """Clear the output container"""
    if 'output_container' in st.session_state:
        st.session_state.output_container.empty()

def show_loading_spinner(message="Processing..."):
    """Show a loading spinner with message"""
    return st.spinner(message)

def clear_cache_and_rerun():
    """Clear Streamlit cache and rerun the app"""
    st.cache_data.clear()
    st.experimental_rerun()
