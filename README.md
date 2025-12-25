# PixelMind - 52 Weeks of Computer Vision 👁️

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![YOLO](https://img.shields.io/badge/YOLO-Object%20Detection-yellow)
![Status](https://img.shields.io/badge/Status-In%20Progress-orange)

## 📖 About This Repository
This repository documents a year-long journey to master Computer Vision (CV). Following a structured **52-week roadmap**, this project covers a progressive series of standalone projects that build in complexity—from basic filtering to 3D reconstruction, SLAM, and autonomous systems.

The goal is to build a comprehensive portfolio demonstrating expertise in traditional CV, deep learning, and real-world application domains like healthcare, robotics, and AR/VR.

## ✍️ Medium Series
I am documenting the entire 52-week process, technical challenges, and UI/UX design decisions in a dedicated Medium series.

[**📖 Read the Full Series Here**](YOUR_MEDIUM_SERIES_LINK_HERE)

* **Why follow?** I break down the code, explain the math, and share the real-world constraints faced during implementation.

## 🛠️ Tech Stack & Tools
The projects leverage a wide range of industry-standard tools and libraries:
* **Languages:** Python, C++ (optional for SLAM/Robotics).
* **Core CV:** OpenCV, NumPy, PIL.
* **Deep Learning:** PyTorch, torchvision, Ultralytics YOLO (v8), TensorFlow Lite.
* **Specialized Libs:** MediaPipe, Open3D, Tesseract OCR, Detectron2, Hugging Face Transformers.
* **Deployment/App:** FastAPI, Streamlit, Docker, ARCore/ARKit, Raspberry Pi/Jetson.

## 📂 Repository Structure
Each week is organized into its own directory containing the source code, notebooks, and requirements.

```text
52-Weeks-Computer-Vision/
├── Week-01-Basic-Image-Processing/
├── Week-02-Feature-Detection-Matching/
...
├── Week-52-Capstone-Project/
├── data/              # Shared datasets (gitignored if large)
├── utils/             # Helper scripts used across multiple weeks
└── README.md
```

## 🗺️ The Roadmap

### Phase 1: Foundations (Weeks 1-8)
| Week | Project Title | Key Tools | Status |
| :--- | :--- | :--- | :--- |
| 01 | **Basic Image Processing** (Filters & Edges) | OpenCV, NumPy | ⬜ |
| 02 | **Feature Detection & Matching** (SIFT/ORB) | OpenCV (features2d) | ⬜ |
| 03 | **Contour Detection & Shape Recognition** | OpenCV (findContours) | ⬜ |
| 04 | **Color Segmentation & Tracking** | OpenCV (inRange) | ⬜ |
| 05 | **Face Detection with Haar Cascades** | OpenCV (CascadeClassifier) | ⬜ |
| 06 | **CNN Image Classification** (MNIST/CIFAR) | PyTorch | ⬜ |
| 07 | **Transfer Learning Web App** (Plant Disease) | PyTorch, Flask/Streamlit | ⬜ |
| 08 | **Real-Time Object Detection** (YOLO) | Ultralytics YOLO | ⬜ |

### Phase 2: Segmentation & Tracking (Weeks 9-13)
| Week | Project Title | Key Tools | Status |
| :--- | :--- | :--- | :--- |
| 09 | **Semantic Segmentation** (UNet/FCN) | PyTorch, Albumentations | ⬜ |
| 10 | **Instance Segmentation** (Mask R-CNN) | Detectron2 / Torchvision | ⬜ |
| 11 | **Multi-Object Tracking in Video** | YOLO, SORT/DeepSORT | ⬜ |
| 12 | **Human Pose Estimation** | OpenPose / MediaPipe | ⬜ |
| 13 | **Hand Gesture Recognition** | MediaPipe Hands | ⬜ |

### Phase 3: Applied CV Applications (Weeks 14-20)
| Week | Project Title | Key Tools | Status |
| :--- | :--- | :--- | :--- |
| 14 | **Autonomous Driving: Lane & Sign Detection** | OpenCV, HoughLines | ⬜ |
| 15 | **Medical Imaging Diagnosis** (X-ray/MRI) | PyTorch, Grad-CAM | ⬜ |
| 16 | **Augmented Reality** (Marker-based) | OpenCV (ArUco) | ⬜ |
| 17 | **OCR App** (Document Scanning) | Tesseract / PyTorch | ⬜ |
| 18 | **Face Recognition & Deepfake Demo** | face_recognition, dlib | ⬜ |
| 19 | **Retail Analytics Dashboard** (People Counting) | YOLO, Plotly/Streamlit | ⬜ |
| 20 | **Advanced Face Recognition System** | DeepFace / FaceNet | ⬜ |

### Phase 4: 3D Vision & Robotics (Weeks 21-24)
| Week | Project Title | Key Tools | Status |
| :--- | :--- | :--- | :--- |
| 21 | **3D Reconstruction from Multiple Views** | Open3D, OpenCV | ⬜ |
| 22 | **SLAM** (Simultaneous Localization & Mapping) | ORB-SLAM, RTAB-Map | ⬜ |
| 23 | **Robotics: Vision-Based Navigation** | ROS, Gazebo/PyBullet | ⬜ |
| 24 | **Reinforcement Learning with Visual Input** | OpenAI Gym, DQN | ⬜ |

### Phase 5: Generative AI & Mobile (Weeks 25-30)
| Week | Project Title | Key Tools | Status |
| :--- | :--- | :--- | :--- |
| 25 | **Neural Style Transfer** (Artistic Filters) | PyTorch (VGG) | ⬜ |
| 26 | **GANs: Image Generation** | DCGAN / StyleGAN | ⬜ |
| 27 | **Image/Video Captioning** (CV + NLP) | PyTorch (CNN+LSTM) | ⬜ |
| 28 | **AR Navigation Mobile App** | ARCore/ARKit, Unity | ⬜ |
| 29 | **Panoramic Image Stitching** | OpenCV Stitcher | ⬜ |
| 30 | **Image Super-Resolution** | SRGAN / EDSR | ⬜ |

### Phase 6: Advanced Deployment & Analysis (Weeks 31-40)
| Week | Project Title | Key Tools | Status |
| :--- | :--- | :--- | :--- |
| 31 | **Advanced Segmentation** (Cityscapes) | DeepLabv3+ | ⬜ |
| 32 | **Domain Adaptation in Vision** | CycleGAN / CORAL | ⬜ |
| 33 | **Edge Deployment** (Raspberry Pi/Jetson) | TFLite / TensorRT | ⬜ |
| 34 | **Mobile App with On-Device CV** | Android/iOS, TFLite | ⬜ |
| 35 | **Vision as a Web Service** (REST API) | FastAPI, Docker | ⬜ |
| 36 | **Video Anomaly Detection** | Autoencoders | ⬜ |
| 37 | **Human Action Recognition** | 3D CNN / Transformers | ⬜ |
| 38 | **Face Animation / Motion Transfer** | First-Order Motion | ⬜ |
| 39 | **3D Object Detection** (Lidar + Camera) | PointPillars | ⬜ |
| 40 | **Satellite Image Analysis** | GDAL, PyTorch | ⬜ |

### Phase 7: Specialization & Capstone (Weeks 41-52)
| Week | Project Title | Key Tools | Status |
| :--- | :--- | :--- | :--- |
| 41 | **Medical Image Segmentation** (3D MRI/CT) | MONAI, 3D U-Net | ⬜ |
| 42 | **Visual Question Answering (VQA)** | Transformers, VQA v2 | ⬜ |
| 43 | **Explainable AI in Vision** | Captum, LIME | ⬜ |
| 44 | **Vision for E-Commerce** (Virtual Try-On) | OpenCV, Siamese Nets | ⬜ |
| 45 | **Robot Pick-and-Place Vision** | PyBullet / ROS | ⬜ |
| 46 | **AR Filter** (Mixed Reality) | MediaPipe, ARKit | ⬜ |
| 47 | **Sports Video Analytics** | Tracking, Pose Est. | ⬜ |
| 48 | **Wildlife / Environmental Monitoring** | Object Detection | ⬜ |
| 49 | **Multi-Camera Integration** | OpenCV, Multithreading | ⬜ |
| 50 | **Real-Time Distributed Vision Pipeline** | GStreamer, ZeroMQ | ⬜ |
| 51 | **Large-Scale CV & Cloud Deployment** | AWS/GCP, Spark | ⬜ |
| 52 | **Capstone Project: Integrated CV System** | Full Stack CV | ⬜ |


## 📚 References & Resources
This roadmap was inspired by industry trends and expert collections. The following resources provide additional context and guidance for the projects in this repository:

* **Ultralytics YOLO Documentation** - [Official Docs](https://docs.ultralytics.com/)
    * *Primary resource for real-time object detection tasks (Weeks 8, 11, 19).*
* **DataCamp** - [19 Computer Vision Projects From Beginner to Advanced](https://www.datacamp.com/blog/computer-vision-projects)
    * *Source for project difficulty scaling and foundational concepts.*
* **DigitalOcean** - [11 Computer Vision Projects to Master Real-World Applications](https://www.digitalocean.com/resources/articles/computer-vision-projects)
    * *Guide for applying CV to practical domains like healthcare and security.*
* **Analytics Vidhya** - [30 Computer Vision Projects for 2025](https://www.analyticsvidhya.com/blog/2025/01/computer-vision-projects/)
    * *Insights on future trends and advanced project ideas.*

## 📄 License
Distributed under the MIT License. See `LICENSE` for more information.

This project is for educational purposes. Please attribute original authors when using specific datasets or pre-trained models.

## 📫 Connect with Me
| Platform | Handle |
| :--- | :--- |
| <a href="https://medium.com/@blue-panther"><img src="https://img.shields.io/badge/Medium-12100E?style=for-the-badge&logo=medium&logoColor=white" height="40"/></a> | **[@blue-panther](https://medium.com/@blue-panther)** |
| <a href="https://www.linkedin.com/in/olabisi-abdusshakur-3986b3219"><img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" height="40"/></a> | **[Abdusshakur Olabisi](https://www.linkedin.com/in/olabisi-abdusshakur-3986b3219)** |
| <a href="https://x.com/blue_panther__"><img src="https://img.shields.io/badge/X-000000?style=for-the-badge&logo=x&logoColor=white" height="40"/></a> | **[@blue_panther__](https://x.com/blue_panther__)** |
| <a href="https://zindi.africa/users/alphaTechie_DSN"><img src="https://img.shields.io/badge/Zindi-4B0082?style=for-the-badge&logo=google-earth&logoColor=white" height="40"/></a> | **[@alphaTechie_DSN](https://zindi.africa/users/alphaTechie_DSN)** |
