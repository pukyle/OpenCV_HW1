# OpenCVDl Homework 1 – Image Processing with PyQt5 GUI  

---

## Overview
This project implements a **graphical image processing application** using **Python (OpenCV + PyQt5)**.  
It provides a user-friendly GUI that performs five fundamental image operations — color processing, image smoothing, edge detection, affine transformation, and adaptive thresholding.  

Each function can be executed interactively through buttons and trackbars, allowing users to visualize image processing results in real-time.

---

## Features
### 1. **Color Processing**
- **1.1 Color Separation**  
  Splits an RGB image into three single-channel images (R, G, and B).
- **1.2 Color Transformation**  
  Converts RGB to grayscale using both OpenCV’s weighted formula and manual averaging.
- **1.3 Color Extraction**  
  Removes yellow and green regions from the image using HSV thresholding.

---

### 2. **Image Smoothing**
- **2.1 Gaussian Blur**  
  Applies Gaussian filtering with adjustable kernel radius via a trackbar.  
- **2.2 Bilateral Filter**  
  Smooths images while preserving edges; supports kernel size adjustment.  
- **2.3 Median Filter**  
  Reduces noise by replacing each pixel with the median of its neighborhood.

---

### 3. **Edge Detection**
- **3.1 Sobel X / 3.2 Sobel Y**  
  Implements Sobel operator manually to detect vertical and horizontal edges.  
- **3.3 Combination and Threshold**  
  Combines Sobel X and Y results, normalizes the gradient, and applies binary thresholds.  
- **3.4 Gradient Angle**  
  Calculates gradient direction and visualizes specific angle ranges (e.g., 170–190°, 260–280°).

---

### 4. **Transforms**
- Applies **2D affine transformations** (rotation, scaling, translation) based on user input.  
- Uses transformation matrices:  
 M' = M_translate × M_rotate/scale
- Default center: `(240, 200)`; output resolution: `1920x1080`.

---

### 5. **Adaptive Threshold**
- **5.1 Global Threshold**  
  Applies fixed thresholding (T=80) to separate foreground and background.  
- **5.2 Local (Adaptive) Threshold**  
  Uses adaptive mean thresholding for uneven illumination conditions.

---

## GUI Interface
The GUI is built using **PyQt5**, providing an intuitive layout with labeled blocks:  
- Image Loading (Load Image 1 / 2)  
- Five function blocks with individual buttons  
- Optional watermark display (bottom-right corner)

---

## Project Structure
```text
OpenCV_HW1/
├── main.py               # Main GUI and image processing logic
├── makefile              # For automated build/run (optional)
├── poetry.lock           # Poetry dependency lock file
├── pyproject.toml        # Project and dependency configuration
├── images/
│   ├── Q1/               # Images for color processing
│   ├── Q2/               # Images for smoothing
│   ├── Q3/               # Images for edge detection
│   ├── Q4/               # Images for transform
│   ├── Q5/               # Images for thresholding
│   └── ilovehomework.png # Watermark background
└── README.md
```
---

## Requirements
- **Python 3.10+**
- **OpenCV (cv2)**
- **NumPy**
- **SciPy**
- **PyQt5**

Install all dependencies:
```bash
pip install opencv-python numpy scipy PyQt5
```

## Run the Program
```bash
python main.py
```
The GUI window will appear.
Load images using “Load Image 1” or “Load Image 2”, then click on any processing button to execute the function.

## Sample Screenshots
<img width="790" height="826" alt="image" src="https://github.com/user-attachments/assets/86715973-a5e3-48ff-afe7-2616a5cd1edd" />
