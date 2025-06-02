import time
import streamlit as st

def timer(func):
    """Decorator to time function execution."""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        st.sidebar.write(f"⏱️ {func.__name__} took {(end - start):.2f} seconds")
        return result
    return wrapper
