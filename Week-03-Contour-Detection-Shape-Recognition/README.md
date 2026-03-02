# Week 3: Contour Detection & Shape Recognition

## 📌 Project Overview
This notebook implements a classical computer vision pipeline to detect contours and recognize simple geometric shapes (triangles, quadrilaterals, circles) using OpenCV. It converts images to binary, extracts contours, approximates them as polygons, and uses vertex counts and geometric heuristics to label shapes.

## 🎯 Objectives
- Detect object contours in binary images using `cv.findContours`
- Approximate contours with `cv.approxPolyDP` and simplify shapes
- Classify simple geometric shapes based on polygon vertex counts
- Compute centroids and label detected shapes on the image
- Visualize detected contours, approximations, and labeled outputs

## 🛠️ Key Tools & Libraries
- Python 3.x
- OpenCV (`cv2`) — `findContours`, `approxPolyDP`, `drawContours`, `moments`
- NumPy
- Matplotlib (for inline visualization in the notebook)

## 📂 Files
- `notebook.ipynb`: Main experiment (image load, thresholding, contour extraction, shape classification)
- `requirements.txt`: Python dependencies
- `data/pictures/`: example images used by the notebook (e.g., `../data/pictures/pics/shapes_2.jpeg`)

Note: there is no `main.py` in this folder — the notebook contains the runnable pipeline and visualization code.

## 📷 Results & Screenshots
- The notebook displays: original image, binary image, detected contours, and final labeled output with shape names placed near each detected object's centroid.
- Sample output shows successful detection and labeling of triangles, quadrilaterals, and circular shapes for clean sample images.

## ▶️ How to run
1. Create and activate your Python environment.
2. Install dependencies:

```
pip install -r requirements.txt
```

3. Open and run the notebook:

```
jupyter notebook Week-03-Contour-Detection-Shape-Recognition/notebook.ipynb
```

## ▶️ Notes & Tips
- Thresholding quality is critical — tune the threshold or use adaptive thresholding for uneven lighting.
- Small contour areas can be ignored (e.g., area < 500) to reduce noise.
- For more robust classification, add aspect-ratio checks to distinguish squares vs rectangles and circularity measures for circles.

## 🔗 References
- OpenCV contours documentation: https://docs.opencv.org/
- Tutorials on shape detection and `approxPolyDP` available online

---
Would you like me to extract the main processing loop into a `shapes.py` script or add example output images to this README?
