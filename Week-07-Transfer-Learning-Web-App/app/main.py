import streamlit as st
from PIL import Image
import os

# Import the custom modules
import model_utils
import ui_components

# 1. Setup Page Configuration
st.set_page_config(
    page_title="Flower Classifier | Week 7",
    page_icon="🌸",
    layout="wide"
)

def main():
    # 2. Render the static UI elements
    ui_components.render_sidebar()


if __name__ == "__main__":
    main()