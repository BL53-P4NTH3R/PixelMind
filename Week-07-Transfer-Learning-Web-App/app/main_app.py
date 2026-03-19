import streamlit as st
from PIL import Image
import os

# Import the custom modules
import model_utils
import ui_components

def main_app_page():
    """Renders the main app contents"""

    # 1. Custom CSS
    st.markdown("""
        <style>
        /* Style the file uploader to be green */
        [data-testid="stFileUploader"] {
            background-color: #e8f4ea;
            border-radius: 10px;
            padding: 10px;
        }
        /* Style the second column's container */
        .result-container {
            background-color: #e8f4ea;
            padding: 20px;
            border-radius: 10px;
            height: 100%;
        }
        /* Style the bottom "About" container */
        .about-container {
            background-color: #e8f4ea;
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
        }
        </style>
    """, unsafe_allow_html=True)

    ui_components.render_header()

    # 2. Define paths to the model artifacts
    app_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(app_dir)
    models_dir = os.path.join(project_root, "models")

    WEIGHTS_PATH = os.path.join(models_dir, "resnet50_flowers102.pth")
    CLASSES_PATH = os.path.join(models_dir, "class_names.json")
    NUM_CLASSES = 102

    # 3. Load the model and class names (cached to prevent reloading)
    if not os.path.exists(WEIGHTS_PATH) or not os.path.exists(CLASSES_PATH):
        st.warning("⚠️ Model weights or class names JSON not found. Please ensure they are in the 'models' folder.")
        return
    
    model = model_utils.load_model(WEIGHTS_PATH, NUM_CLASSES)
    class_names = model_utils.load_class_names(CLASSES_PATH)

    if model is None:
        st.error("Failed to load the model. Please check your weights file.")
        return

    # 4. File Uploader Section
    with st.container():
        uploaded_file = st.file_uploader(
            "Upload an image of a flower...", 
            type=['jpg', 'jpeg', 'png'],
            help="Upload your image to see the classificaiton.")

    if uploaded_file is not None:
        # Read the image using PIL
        image = Image.open(uploaded_file).convert('RGB')

        # Add a spinner to give the user visual feedback wile the model runs
        with st.spinner("Analyzing petals and leaves..."):
            # Preprocess the image
            image_tensor = model_utils.preprocess_image(image)

            # Run inference
            predictions = model_utils.predict(model, image_tensor, class_names, top_k=3)

        # 5. Display the results using the UI component
        ui_components.display_results(uploaded_file, predictions)