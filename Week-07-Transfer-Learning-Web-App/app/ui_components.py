import streamlit as st
from PIL import Image
import os

import main_app



def about_tl_page():
    """Renders about transfer learning informational content"""
    st.title("About Transfer Learning")
    st.markdown(
        """
        Transfer learning reuses a model that was already trained on a large dataset and adapts it to a new task.

        For this project, a pretrained ResNet50 backbone was fine-tuned to classify flower species more efficiently than training from scratch.
        """
    )

def model_details_page():
    """Displays details of the model."""
    st.title("Model Details")
    st.markdown(
        """
        - Architecture: ResNet50
        - Task: 102-class flower classification
        - Framework: PyTorch
        - Interface: Streamlit
        """
    )


def render_sidebar():
    """Renders the sidebar with project information and links."""

    # 1. Navigation Menu
    pg = st.navigation({"Project Info":  [
            st.Page(main_app.main_app_page, title="App", icon=":material/grid_view:"),
            st.Page(about_tl_page, title="About Transfer Learning", icon=":material/info:"),
            st.Page(model_details_page, title="Model Details", icon=":material/deployed_code:")
    ]})

    # Add the static sidebar elements
    with st.sidebar:
        # 1. Top Image
        try:
            app_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(app_dir)
            assets_dir = os.path.join(project_root, "assets")

            image_path = os.path.join(assets_dir, "flowers_sidebar.jpg")
            if not os.path.exists(image_path):
                image_path = os.path.join(assets_dir, "flowers_sidebar.jpeg")

            sidebar_img = Image.open(image_path)
            st.image(sidebar_img, use_container_width=True)
        except FileNotFoundError:
            # Fallback just in case the image isn't in the folder yet
            st.info("Sidebar image not found. Place 'flowers_sidebar.jpg' or 'flowers_sidebar.jpeg' in the assets folder.")
        st.divider()
        st.markdown("**Week 7 Project**:  \nFlower Classifier")
        st.write("") 
    
        st.markdown("**Base Model**:  \nFine-tuned ResNet50")
        st.write("")
    
        st.markdown("**Framework**:  \nPyTorch")
        st.write("")
    
        st.markdown("**Built with**:  \nStreamlit")
        
    pg.run()
    


def render_header():
    """Renders the main title and upload instructions."""
    st.title("🌸 Flower Classifier - Week 7 Project")
    # st.markdown(
    #     """
    #     Upload an image of a flower, and the fine-tuned ResNet model will predict it species.
    #     """
    # )

def display_results(uploaded_file, predictions: list):
    """
    Displays the uploaded image and the top predictions side-by-side.
    Expects predictions to be a list of dicts: [{"class": name, "confidence": score}]
    """
    if uploaded_file is None:
        st.info("Upload an image to see predictions.")
        return

    # Read the image using PIL
    image = Image.open(uploaded_file).convert("RGB")
    resized_image = image.resize((250, 250))

    # Create two columns for a clean, dashboard-like layout
    col1, col2 = st.columns(2, gap="medium")

    with col1:
        with st.container(border=True):
            st.image(resized_image, use_container_width=True)
            st.caption(f"Dimensions: {image.width} x {image.height}")
            st.caption(f"Filename: {uploaded_file.name}")

    with col2:
        with st.container():
            st.markdown('<div class="result-container result-box">', unsafe_allow_html=True)
            st.subheader("Classification Results:")

            if not predictions:
                st.warning("No predictions returned.")
                return
            
            # Top prediction gets highlighted
            top_pred = predictions[0]
            
            st.write(f"**Predicted Flower:** {top_pred['class']}")
            st.write(f"**Confidence:** {top_pred['confidence']:.1f}%")
            
            # Green progress bar
            st.progress(top_pred['confidence'] / 100)
            
            st.divider()
            st.write("**Top-3 predictions:**")
            for i, pred in enumerate(predictions[:3]):
                st.write(f"{i+1}. {pred['class']}: {pred['confidence']:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)

    # 2. "About This Flower" section below the columns
    st.markdown(f"""
        <div class="about-container">
            <h3>About This Flower</h3>
            <p>The {top_pred['class']} is a beautiful species... [Add your model's description here].</p>
            <br>
        </div>
    """, unsafe_allow_html=True)
    
    # Button to reset/upload another
    if st.button("Classify Another Image"):
        st.rerun()

