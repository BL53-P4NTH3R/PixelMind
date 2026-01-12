# Week 1: Basic Image Processing

## 📌 Project Overview
This project is part of **PixelMind**, my 52-week journey to build practical computer vision skills from the ground up.

In Week 1, I focused on basic image processing techniques, specifically **image blurring** and **edge detection**. The goal was not to build a complex system, but to understand how different preprocessing techniques affect images and how they influence edge detection results.

All experiments were done using simple test images, with a strong emphasis on **visual comparison and parameter exploration**.

---

## 🎯 Objectives
- [x] Load and visualize images using OpenCV  
- [x] Apply different image blurring techniques  
- [x] Understand the effect of kernel size and parameters  
- [x] Implement multiple edge detection methods  
- [x] Compare outputs across different techniques

---

## 🛠️ Key Tools & Libraries
- **Python**
- **OpenCV**
- **NumPy**
- **Matplotlib**

*(No deep learning libraries used in this week)*

---


## 📂 Files
- `notebook.ipynb`: Step-by-step experiments, visualizations, and comparisons  
- `main.py` *(optional)*: Script version of core operations  
- `requirements.txt`: Project dependencies

---


## 🧪 Implemented Techniques

### Image Blurring
- Gaussian Blurring  
- Median Blurring  
- Bilateral Blurring  

### Edge Detection
- Sobel Operator  
- Laplacian Operator  
- Canny Edge Detection  

Each technique was tested with different parameter values to observe trade-offs between noise reduction, edge preservation, and computational cost.

---

## 📷 Results & Visual Comparisons
Results are presented as side-by-side visualizations in the notebook, showing:
- Original vs blurred images  
- Effects of different kernel sizes  
- Edge detection outputs for each method 

---

## 🔍 Key Learnings
- Blurring plays a critical role in edge detection quality  
- Different blurring techniques preserve image details differently  
- Edge detectors respond very differently to noise  
- Canny performs best when preprocessing is done correctly  

---

## 🔗 References
- [OpenCV Documentation: Image Filtering](https://opencv.org/blog/image-filtering-using-convolution-in-opencv/)
- [OpenCV Documentation: Edge Detection](https://opencv.org/blog/edge-detection-using-opencv/)
- *Computer Vision: Algorithms and Applications* — Richard Szeliski