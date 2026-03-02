# Week 5: Face Detection with Haar Cascades

## 📌 Project Overview
This notebook demonstrates classical face detection using the Viola–Jones Haar Cascade approach implemented in OpenCV. It covers loading pretrained cascade classifiers, detecting faces and eyes in static images, and running real-time detection on webcam/video input.

## 🎯 Objectives
- Understand the Viola–Jones / Haar Cascade face detection pipeline
- Load and use OpenCV's pretrained Haar cascade XML classifiers
- Detect faces in static images and draw bounding boxes
- Detect eyes within detected face regions (hierarchical detection)
- Run a real-time face detection loop using a webcam

## 🛠️ Key Tools & Libraries
- Python 3.x
- OpenCV (`cv2`) — cascade classifiers and video capture
- NumPy
- Matplotlib (for image display in the notebook)

Important OpenCV components used:
- `cv.CascadeClassifier` (pretrained XMLs: `haarcascade_frontalface_default.xml`, `haarcascade_eye.xml`)
- `detectMultiScale` (tunable `scaleFactor`, `minNeighbors`, `minSize`)
- Video capture utilities (`cv.VideoCapture` / `cv.imshow`)

## 📂 Files
- `notebook.ipynb`: Main experiment (loading cascades, image detection, webcam demo)
- `requirements.txt`: Python dependencies
- `data/pictures/`: example images used by the notebook (e.g., `../data/pictures/pics/img_n_34.jpg`)

Note: there is no `main.py` in this folder — the notebook contains the runnable pipeline and demonstration code.

## 📷 Results & Screenshots
- The notebook shows detected face bounding boxes and eye boxes on sample images.
- It includes a live webcam demo that draws green boxes around detected faces in real time.
- For reproducible examples, run the notebook and save output images or screen-record the webcam session, then add them to this folder and reference them here.

## ▶️ How to run
1. Create and activate your Python environment.
2. Install dependencies:

```
pip install -r requirements.txt
```

3. Open the notebook and run cells to step through the pipeline:

```
jupyter notebook Week-05-Face-Detection-Haar-Cascades/notebook.ipynb
```

4. To run the webcam demo, execute the webcam cell and press `q` to quit the live window. Ensure a camera is available and accessible.

## ▶️ Notes & Tips
- Haar cascades work best on grayscale images — the notebook uses `cv.cvtColor(..., cv.COLOR_BGR2GRAY)` and `cv.equalizeHist` to improve contrast.
- Tuning `scaleFactor`, `minNeighbors`, and `minSize` in `detectMultiScale` affects detection rate and false positives.
- Cascade XMLs are loaded from `cv.data.haarcascades` (no manual download required when using OpenCV).

## 🔗 References
- OpenCV Haar cascades: https://docs.opencv.org/
- Viola–Jones original paper and summaries available online

---
Would you like me to extract the webcam demo into a `detect.py` script or add sample output images to this README?
