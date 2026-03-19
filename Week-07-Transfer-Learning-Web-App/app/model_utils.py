import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import json
import streamlit as st

# Setup device-agnostic code
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

@st.cache_resource
def load_model(weight_path, num_classes):
    """
    Loads the fine-tuned ResNet50 model and weights.
    @st.cache_resource ensures the model is only loaded once per session,
    preventing Streamlit from reloading it on every UI interaction.
    """
    try:
        # 1. Initialize the base model
        model = models.resnet50(weights=None)

        # 2. Modify the final connected layer to match the flower classes
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

        # 3. Load the fine-tuned weights
        model.load_state_dict(torch.load(weight_path, map_location=device))

        # 4. Move to appropriate device and set to evaluation model
        model = model.to(device)
        model.eval()

        return model
    except Exception as e:
        st.error(F"Error loading model: {e}")
        return None


@st.cache_data
def load_class_names(json_path):
    """Loads the mapping of class indices to flower names."""
    with open(json_path, 'r') as f:
        class_names = json.load(f)
    return class_names

def preprocess_image(image: Image.Image):
    """
    Applies standard ImageNet transformations to the incoming PIL Image.
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])
    # Add a batch dimension (B, C, H, W)
    return transform(image).unsqueeze(0).to(device)

def predict(model, image_tensor, class_names, top_k=3):
    """
    Runs inference and returns the top K predictions with confidence scores.
    """
    with torch.no_grad():
        outputs = model(image_tensor)

        # Apply softmax to get probabilities
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

        # Get the top K probabilities and their corresponding indices
        top_prob, top_indicies = torch.topk(probabilities, top_k)

        results = []
        for i in range(top_k):
            idx = top_indicies[i].item()
            prob = top_prob[i].item() * 100

            # Retrieve class name if available, otherwise use index
            if isinstance(class_names, list):
                if 0 <= idx < len(class_names):
                    class_name = class_names[idx]
                else:
                    class_name = f"Class {idx}"
            elif isinstance(class_names, dict):
                class_name = (
                    class_names.get(str(idx))
                    or class_names.get(idx)
                    or class_names.get(str(idx + 1))
                    or class_names.get(idx + 1)
                    or f"Class {idx}"
                )
            else:
                class_name = f"Class {idx}"

            results.append({"class": class_name, "confidence": prob})

        return results