# Week 6: CNN Image Classification (MNIST)

## 📌 Project Overview
This notebook builds and trains a simple Convolutional Neural Network (CNN) using PyTorch to classify handwritten digits from the MNIST dataset (28x28 grayscale images). The workflow demonstrates dataset loading with `torchvision`, model definition, training loop, evaluation, and visualization of loss and predictions.

## 🎯 Objectives
- Understand convolutional neural networks and their components
- Load datasets using `torchvision` and `DataLoader`
- Define a CNN architecture in PyTorch
- Implement a full training loop and optimizer
- Evaluate model performance on the MNIST test set
- Visualize training loss and example predictions
- Target: achieve >95% accuracy on MNIST (notebook notes >99% with this simple model)

## 🛠️ Key Tools & Libraries
- Python 3.x
- PyTorch (`torch`) and `torchvision`
- NumPy
- Matplotlib

GPU support is optional — the notebook uses `torch.device("cuda" if available else "cpu")`.

## 📂 Files
- `notebook.ipynb`: Main experiment (dataset load, model, training, evaluation)
- `requirements.txt`: Python dependencies for the notebook
- `data/`: (created by the notebook) downloaded MNIST dataset

Note: there is no separate `main.py` in this folder — the notebook contains the runnable training code.

## 📷 Results & Screenshots
- Test accuracy reported in the notebook: >99% (simple CNN on MNIST)
- The notebook includes: sample training images, training loss curve, and example predictions. Add saved PNGs/GIFs to this folder and reference them here for visual results.

## ▶️ How to run
1. Create / activate your virtual environment.
2. Install dependencies:

```
pip install -r requirements.txt
```

3. Launch Jupyter and open `notebook.ipynb`:

```
jupyter notebook Week-06-CNN-Image-Classification/notebook.ipynb
```

Or run the cells in your preferred notebook environment. Training uses a small model and runs quickly on CPU, faster on GPU.

## 🔗 References
- PyTorch tutorials: https://pytorch.org/tutorials/
- MNIST dataset: available via `torchvision.datasets.MNIST`

---
If you'd like, I can: add a `train.py` script, save model checkpoints, or include example result images in this README.
