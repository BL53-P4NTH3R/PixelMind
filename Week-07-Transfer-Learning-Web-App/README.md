# Week 07: Transfer Learning Web App

## Overview
This project is a Streamlit-based flower image classification app built with transfer learning.
It uses a fine-tuned ResNet50 model trained for 102 flower classes and provides a simple web UI for:

- uploading a flower image,
- running model inference,
- displaying top predictions with confidence scores,
- browsing project pages from the sidebar (App, About Transfer Learning, Model Details).

## Features
- Transfer learning inference with ResNet50
- Streamlit multipage sidebar navigation
- Cached model and class-name loading for faster interaction
- Responsive prediction layout with image preview and confidence bar
- Graceful fallback if model files or sidebar image are missing

## Folder Structure
```
Week-07-Transfer-Learning-Web-App/
|-- app/
|   |-- main.py
|   |-- main_app.py
|   |-- model_utils.py
|   `-- ui_components.py
|-- assets/
|   |-- flowers_sidebar.jpeg
|   `-- style.css
|-- models/
|   |-- class_names.json
|   |-- resnet50_flowers102.pth
|   `-- resnet50_full_model.pth
|-- output_images/
|   |-- img1.png
|   |-- pred_img.png
|   `-- training_loss.png
|-- week07-flowers-classification-model-notebook.ipynb
|-- requirements.txt
`-- README.md
```

## Key Files
- `app/main.py`: Streamlit entry point and page config.
- `app/main_app.py`: Main inference page workflow (upload, preprocess, predict, display).
- `app/model_utils.py`: Model loading, preprocessing transforms, and prediction helpers.
- `app/ui_components.py`: Sidebar navigation/pages, header, and result UI components.
- `models/resnet50_flowers102.pth`: Primary fine-tuned model weights.
- `models/class_names.json`: Class index to flower-name mapping.
- `week07-flowers-classification-model-notebook.ipynb`: Training/experimentation notebook.

## Requirements
Dependencies are listed in `requirements.txt`:

- opencv-python
- numpy
- matplotlib
- torch
- torchvision
- streamlit

## Setup and Run
From the `Week-07-Transfer-Learning-Web-App` folder:

1. Create and activate a Python environment (recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the app:

```bash
streamlit run app/main.py
```

4. Open the local Streamlit URL shown in the terminal.

## Model Artifacts
The app expects these files in `models/`:

- `resnet50_flowers102.pth`
- `class_names.json`

If either file is missing, the app shows a warning and stops inference.

## Notes
- Sidebar image is loaded from `assets/flowers_sidebar.jpg` with fallback to `assets/flowers_sidebar.jpeg`.
- The model is loaded with `@st.cache_resource` to avoid reloading on each interaction.
- Class names are loaded with `@st.cache_data`.

## Output Samples
The `output_images/` folder contains sample artifacts:

- `img1.png`
- `pred_img.png`
- `training_loss.png`
