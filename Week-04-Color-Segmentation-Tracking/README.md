# Week 4: Color Segmentation & Object Tracking

## 📌 Project Overview
This notebook demonstrates color-based segmentation and real-time object tracking using OpenCV. The pipeline converts frames to HSV, thresholds by color, cleans masks with morphological operations, finds the largest contour, computes centroids, and visualizes an object's trajectory over time.

## 🎯 Objectives
- Segment objects based on color using HSV thresholds
- Understand why HSV is preferred for color segmentation
- Create and clean binary masks with `cv2.inRange`, erosion, and dilation
- Find contours, compute centroids, and select the largest object
- Track object position across video frames and visualize trajectory

## 🛠️ Key Tools & Libraries
- Python 3.x
- OpenCV (`cv2`) for image/video I/O, color conversion, morphology, contours, and display
- NumPy for numerical operations
- `collections.deque` used to store recent centroid positions for trajectory drawing

## 📂 Files
- `notebook.ipynb`: Main experiment (video reading, mask creation, tracking loop)
- `requirements.txt`: Python dependencies

Example media input used by the notebook: webcam (device `0`) or a video file supplied to `cv2.VideoCapture`.

## 📷 Results & Screenshots
- The notebook displays: original frame, HSV conversion, binary mask, clean mask, and live tracking with drawn centroid and trajectory lines.
- For reproducible examples, run the notebook and save screenshots or a short recording of the webcam demo, then add them to this folder and reference them here.

## ▶️ How to run
1. Create and activate your Python environment.
2. Install dependencies:

```
pip install -r requirements.txt
```

3. Open the notebook and run cells, or execute the tracking loop directly in a script.

```
jupyter notebook Week-04-Color-Segmentation-Tracking/notebook.ipynb
```

4. To run live tracking ensure a camera is available; press `q` in the display window to quit.

## ▶️ Notes & Tips
- Use HSV bounds tuned for your target color; example blue bounds are shown in the notebook.
- Use `cv2.equalizeHist` or adjust exposure if lighting is poor.
- Increase contour area threshold (e.g., >500 px) to ignore small noise blobs.
- `deque(maxlen=64)` is used to keep a short history for trajectory drawing.

## 🔗 References
- OpenCV documentation: https://docs.opencv.org/
- Tutorials on color spaces and contours available online

---
Would you like me to extract the tracking loop into a standalone `track.py` script or add example output images to this README?
