import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QSizePolicy
)
from scipy import signal

file_name = None
image1 = None
image1_gray = None
image2 = None


def update_radius(value):
    """
    Callback function for OpenCV trackbar.
    Updates the global variable 'window_radius' with the current slider value.
    """
    global window_radius
    window_radius = value


def _GaussianBlur(img):
    """
    Apply a fixed 3x3 Gaussian Blur to the input image using a custom kernel.

    Parameters:
        img (numpy.ndarray): Grayscale or single-channel image.
    
    Returns:
        numpy.ndarray: Blurred image (uint8 type).
    """
    # Define a 3x3 Gaussian kernel (normalized to approximate standard Gaussian weights)
    kernel = np.array([
        [0.045, 0.122, 0.045],
        [0.122, 0.332, 0.122],
        [0.045, 0.122, 0.045],
    ])

    # Perform 2D convolution (with symmetric boundary handling)
    blurred_img = signal.convolve2d(img, kernel, mode="same", boundary="symm")

    return blurred_img.astype(np.uint8)


def Sobel(img, op):
    """
    Apply the Sobel edge detection operator manually on the given image.

    Parameters:
        img (numpy.ndarray): Grayscale image.
        op (numpy.ndarray): 3x3 Sobel operator (e.g., Sobel X or Sobel Y).

    Returns:
        numpy.ndarray: Image after applying the Sobel operator.
    """
    # Step 1: Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(img, (3, 3), 0)

    # Step 2: Create a padded image (zero padding on all sides)
    padding = np.zeros((blur.shape[0] + 2, blur.shape[1] + 2))
    padding[1:-1, 1:-1] = blur

    # Step 3: Convolve Sobel operator manually (pixel by pixel)
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            region = padding[x : x + 3, y : y + 3]  # Extract local 3x3 region
            blur[x, y] = abs(np.sum(region * op))   # Apply operator and take absolute value

    return blur



# ----------------------------------- #
# Define functions for button actions
# ----------------------------------- #

# === General utility ===
def get_path():
    """
    Open a file dialog and update the global variable 'file_name' 
    with the selected file path.
    """
    global file_name
    file_name = QFileDialog.getOpenFileName(None, "Open Image File", ".")[0]


# === Load Image 1 (with color and grayscale versions) ===
def load_img1_btn_clicked():
    """
    Load the first image (both in color and grayscale).
    This image is used for most processing blocks (1, 2, 3, 4, and 5).
    """
    global image1, image1_gray
    get_path()  # Get file path from user dialog

    # Read the color image (BGR)
    image1 = cv2.imread(file_name)
    # Read the same image in grayscale
    image1_gray = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)

    # Check if the image was successfully loaded
    if image1 is None:
        print("[ERROR]: Failed to load Image 1.")
    else:
        print(f"Loaded Image 1: {file_name}")


# === Load Image 2 (used in Block 2: Smoothing) ===
def load_img2_btn_clicked():
    """
    Load the second image (only in color).
    This image is mainly used for Image Smoothing tasks (Block 2).
    """
    global image2
    get_path()  # Get file path from user dialog

    # Read the color image (BGR)
    image2 = cv2.imread(file_name)

    # Check if the image was successfully loaded
    if image2 is None:
        print("[ERROR]: Failed to load Image 2.")
    else:
        print(f"Loaded Image 2: {file_name}")



def Block1_btn_1_1_clicked():
    print("Color Separation button clicked")

    if image1 is None:
        print("[ERROR]: Please load image first")
        return

    # Split BGR image into individual channels
    b, g, r = cv2.split(image1)  # b: Blue channel, g: Green channel, r: Red channel

    # Create a zero matrix (black image) for channel merging
    zeros = np.zeros_like(b)

    # Merge each channel back to BGR images individually
    b_image = cv2.merge((b, zeros, zeros))  # Only Blue channel visible
    g_image = cv2.merge((zeros, g, zeros))  # Only Green channel visible
    r_image = cv2.merge((zeros, zeros, r))  # Only Red channel visible

    # Display each image as Figure
    cv2.imshow("b_image (Blue Channel)", b_image)
    cv2.imshow("g_image (Green Channel)", g_image)
    cv2.imshow("r_image (Red Channel)", r_image)

    # Wait for user input to close windows (press any key)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def Block1_btn_1_2_clicked():
    """
    Perform Color Transformation (Convert RGB → Grayscale)
    Q1: Use OpenCV cvtColor() function for weighted grayscale conversion.
    Q2: Compute average grayscale manually by averaging B, G, R channels.
    """
    print("Color Transformation button clicked")

    # Ensure image is loaded before processing
    if image1 is None:
        print("[ERROR]: Please load image first")
        return

    # Step 1: Split image into B, G, R channels
    b, g, r = cv2.split(image1)

    # Step 2: Q1 - Convert to grayscale using OpenCV built-in function (perceptual weighting)
    cv_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

    # Step 3️: Q2 - Compute grayscale by averaging 3 channels (equal weighting)
    # Formula: avg_gray = (b/3 + g/3 + r/3)
    avg_gray = ((b / 3) + (g / 3) + (r / 3)).astype(np.uint8)

    # Step 4️: Display both results
    cv2.imshow("cv_gray (OpenCV weighted grayscale)", cv_gray)
    cv2.imshow("avg_gray (Average weighted grayscale)", avg_gray)

    # Wait for any key press to close all windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def Block1_btn_1_3_clicked():
    """
    Perform Color Extraction (Remove Yellow and Green regions)
    Steps:
        1 Convert BGR image to HSV color space.
        2 Create a mask for yellow-green color range using cv2.inRange().
        3 Invert the mask to get regions excluding yellow-green.
        4 Apply the inverted mask to remove yellow-green colors from the original image.
    """
    print("Color Extraction button clicked")

    # Check if image is loaded
    if image1 is None:
        print("[ERROR]: Please load image first")
        return

    # Step 1️: Convert BGR → HSV
    hsv_image = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)

    # Step 2️: Define yellow-green HSV range and create mask
    lower_bound = np.array([15, 25, 25])     # Lower bound for yellow-green
    upper_bound = np.array([85, 255, 255])   # Upper bound for yellow-green
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)  # Yellow-green mask (I₁)

    # Step 3️: Invert mask to get non-yellow-green regions
    mask_inverse = cv2.bitwise_not(mask)  # Inversed mask (I₁_inverse)

    # Step 4️: Remove yellow-green colors using bitwise AND with inverted mask
    extracted_image = cv2.bitwise_and(image1, image1, mask=mask_inverse)  # Output (I₂)

    # Step 5️: Display results
    cv2.imshow("I1 mask (Yellow-Green Mask)", mask)
    cv2.imshow("Inversed Mask (I1_inverse)", mask_inverse)
    cv2.imshow("Extracted Image (Image without Yellow & Green)", extracted_image)

    # Wait for key press to close all windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()



import threading

def Block2_btn_2_1_clicked():
    """
    Run Gaussian Blur with interactive trackbar in a separate thread.
    This prevents PyQt5 GUI from freezing or closing when the OpenCV window is closed.
    """

    def gaussian_blur_thread():
        print("Gaussian Blur button clicked")

        if image1 is None:
            print("[ERROR]: Please load Image 1 first")
            return

        window_name = "Gaussian Blur"
        cv2.namedWindow(window_name)
        cv2.createTrackbar("m:", window_name, 1, 5, update_radius)

        print("Use the trackbar (m: 1~5) to adjust blur strength. Press 'q' or close the window to quit.")

        while True:
            # Check if the OpenCV window is still visible
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

            # Quit when pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            current_radius = cv2.getTrackbarPos("m:", window_name)
            if current_radius < 1:
                current_radius = 1

            kernel_size = (2 * current_radius + 1, 2 * current_radius + 1)
            blurred_img = cv2.GaussianBlur(image1, kernel_size, 0)

            cv2.imshow(window_name, blurred_img)

        cv2.destroyAllWindows()

    # Run OpenCV logic in a separate thread to isolate from PyQt GUI
    t = threading.Thread(target=gaussian_blur_thread, daemon=True)
    t.start()


import threading

def Block2_btn_2_2_clicked():
    """
    Apply Bilateral Filter with adjustable kernel radius using a trackbar.
    This version runs in a separate thread to prevent conflicts with the PyQt5 event loop.

    Steps:
        1. Load Image 1 (must be done before clicking the button).
        2. Create a popup window and a trackbar (m ∈ [1, 5]).
        3. Precompute bilateral filter results for different kernel sizes.
        4. Display the filtered image corresponding to the current trackbar value.
        5. Press 'q' or close the window to quit.

    Notes:
        - sigmaColor = 90, sigmaSpace = 90 (fixed)
        - Diameter of kernel = (2m + 1)
        - Running this in a separate thread ensures the PyQt GUI remains responsive.
    """

    def bilateral_filter_thread():
        print("Bilateral Filter button clicked")

        # Step 1: Verify that Image 1 is loaded
        if image1 is None:
            print("[ERROR]: Please load Image 1 first")
            return

        # Step 2: Create OpenCV window and trackbar
        window_name = "Bilateral Filter"
        cv2.namedWindow(window_name)
        cv2.createTrackbar("m:", window_name, 1, 5, update_radius)

        print("Use trackbar (m: 1~5) to adjust bilateral filter size. Press 'q' or close the window to quit.")

        # Step 3: Precompute all filtered images (so it's fast when switching)
        img_list = []
        for i in range(6):  # For m = 0~5
            d = 2 * i + 1
            print(f"Precomputing bilateral filter with d={d}")
            filtered = cv2.bilateralFilter(image1, d, 90, 90)
            img_list.append(filtered)

        # Step 4: Display updates based on trackbar position
        while True:
            # Automatically break if the OpenCV window is closed
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

            # Exit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Get trackbar value safely
            current_radius = cv2.getTrackbarPos("m:", window_name)
            current_radius = max(0, min(current_radius, 5))  # Clamp index

            cv2.imshow(window_name, img_list[current_radius])

        cv2.destroyAllWindows()

    # Launch in a separate thread to avoid blocking the PyQt5 GUI
    t = threading.Thread(target=bilateral_filter_thread, daemon=True)
    t.start()



import threading

def Block2_btn_2_3_clicked():
    """
    Apply Median Filter with adjustable kernel radius using a trackbar.
    This version runs in a separate thread to prevent the PyQt5 GUI from freezing or closing.

    Steps:
        1. Load Image 2 (must be done via the GUI button).
        2. Create a popup window named 'Median Filter' with a trackbar (m ∈ [1, 5]).
        3. Precompute the median-filtered images for all valid kernel sizes.
        4. Display results interactively based on the current trackbar position.
        5. Press 'q' or close the window to quit.

    Notes:
        - Median filter replaces each pixel with the median of its neighborhood.
        - Larger kernel size → stronger smoothing, less noise, blurrier edges.
        - Running inside a separate thread ensures smooth coexistence with PyQt5.
    """

    def median_filter_thread():
        print("Median Filter button clicked")

        # Step 1: Ensure image2 is loaded
        if image2 is None:
            print("[ERROR]: Please load Image 2 first")
            return

        # Step 2: Create the OpenCV window and trackbar
        window_name = "Median Filter"
        cv2.namedWindow(window_name)
        cv2.createTrackbar("m:", window_name, 1, 5, lambda x: None)

        print("Use trackbar (m: 1~5) to adjust median filter size. Press 'q' or close the window to quit.")

        # Step 3: Precompute all filtered images (for smoother switching)
        img_list = []
        for i in range(6):  # Generate filters for m = 0~5
            ksize = 2 * i + 1
            print(f"Precomputing Median Filter: kernel size = {ksize}x{ksize}")
            filtered_img = cv2.medianBlur(image2, ksize)
            img_list.append(filtered_img)

        # Step 4: Real-time interactive display
        while True:
            # Auto-exit if the OpenCV window is closed
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

            # Exit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Get trackbar position safely
            current_radius = cv2.getTrackbarPos("m:", window_name)
            current_radius = max(0, min(current_radius, 5))  # Clamp range

            # Display the precomputed image
            cv2.imshow(window_name, img_list[current_radius])

        # Step 5: Cleanup
        cv2.destroyAllWindows()

    # Run the filter logic in a separate thread to avoid PyQt event conflicts
    t = threading.Thread(target=median_filter_thread, daemon=True)
    t.start()




def Block3_btn_3_1_clicked():
    """
    Perform Sobel X edge detection manually.
    Steps:
        1 Use grayscale image as input.
        2 Apply Gaussian smoothing to reduce noise.
        3 Apply 3x3 Sobel X operator (detect vertical edges).
        4 Display result using cv2.imshow().

    Note:
        - cv2.Sobel() and cv2.filter2D() are NOT allowed.
        - The Sobel X operator detects vertical edges by 
          emphasizing horizontal intensity changes.
    """
    print("Sobel X button clicked")

    # Step 1️: Check if grayscale image is loaded
    if image1_gray is None:
        print("[ERROR]: Please load image first")
        return

    # Step 2️: Define 3×3 Sobel X operator
    # Detect vertical edges (emphasize left-right gradient)
    sobel_operator = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])

    # Step 3️: Apply Sobel filter manually (custom function Sobel())
    result = Sobel(image1_gray, sobel_operator)

    # Step 4️: Display result
    cv2.imshow("Sobel X (Vertical Edge)", result)

    # Wait for any key to close
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def Block3_btn_3_2_clicked():
    """
    Perform Sobel Y edge detection manually.
    Steps:
        1 Use grayscale image as input.
        2 Apply Gaussian smoothing to reduce noise.
        3 Apply 3x3 Sobel Y operator (detect horizontal edges).
        4 Display result using cv2.imshow().

    Note:
        - cv2.Sobel() and cv2.filter2D() are NOT allowed.
        - The Sobel Y operator detects horizontal edges by 
          emphasizing vertical intensity changes.
    """
    print("Sobel Y button clicked")

    # Step 1️: Check if grayscale image is loaded
    if image1_gray is None:
        print("[ERROR]: Please load image first")
        return

    # Step 2️: Define 3×3 Sobel Y operator
    # Detect horizontal edges (emphasize top-bottom gradient)
    sobel_operator = np.array([
        [1,  2,  1],
        [0,  0,  0],
        [-1, -2, -1]
    ])

    # Step 3️: Apply Sobel filter manually using the custom Sobel() function
    result = Sobel(image1_gray, sobel_operator)

    # Step 4️: Display result
    cv2.imshow("Sobel Y (Horizontal Edge)", result)

    # Wait for any key to close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def Block3_btn_3_3_clicked():
    """
    Combine Sobel X and Sobel Y results, then apply thresholding.
    Steps:
        1 Compute Sobel X and Sobel Y using custom Sobel() function.
        2 Combine them by sqrt(Sobel_x^2 + Sobel_y^2).
        3 Normalize result to range [0, 255] using cv2.normalize().
        4 Apply binary thresholding (two levels: 128 and 28).
        5 Display both combination image and threshold results.

    Note:
        - cv2.normalize() with cv2.NORM_MINMAX ensures proper scaling.
        - Threshold at 128 keeps strong edges; threshold at 28 shows weaker edges too.
    """
    print("Combination and Threshold button clicked")

    # Step 1️: Check image
    if image1_gray is None:
        print("[ERROR]: Please load image first")
        return

    # Step 2️: Compute Sobel X & Y using custom implementation
    sobel_x = Sobel(image1_gray, np.array([[-1, 0, 1],
                                           [-2, 0, 2],
                                           [-1, 0, 1]]))
    sobel_y = Sobel(image1_gray, np.array([[1, 2, 1],
                                           [0, 0, 0],
                                           [-1, -2, -1]]))

    # Step 3️: Combine results (edge magnitude)
    combination = np.sqrt(sobel_x.astype(np.float32)**2 + sobel_y.astype(np.float32)**2)

    # Step 4️: Normalize to [0, 255]
    normalized = cv2.normalize(combination, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Step 5️: Apply threshold (two values)
    _, thresh_128 = cv2.threshold(normalized, 128, 255, cv2.THRESH_BINARY)
    _, thresh_28  = cv2.threshold(normalized, 28,  255, cv2.THRESH_BINARY)

    # Step 6️: Display results
    cv2.imshow("Combination (Sobel X + Sobel Y)", normalized)
    cv2.imshow("Threshold = 128", thresh_128)
    cv2.imshow("Threshold = 28", thresh_28)

    cv2.waitKey(0)
    cv2.destroyAllWindows()



def Block3_btn_3_4_clicked():
    """
    Compute Gradient Angle θ using Sobel X and Sobel Y, and show specific angle ranges.
    Steps:
        1 Apply Gaussian smoothing to reduce noise.
        2 Compute Sobel X and Y manually (3×3 kernels).
        3 Compute gradient magnitude and angle θ = arctan2(Y, X).
        4 Generate two masks:
            - Mask 1: θ ∈ [170°, 190°]
            - Mask 2: θ ∈ [260°, 280°]
        5 Apply bitwise AND to show edge regions within those angle ranges.
        6 Display both results.

    Note:
        - Angle range follows the coordinate system in the figure:
          X-axis: Sobel X, Y-axis: Sobel Y.
        - np.arctan2 returns angle in radians → converted to degrees (0–360°).
    """
    print("Gradient Angle button clicked")

    # Step 1️: Check if image is loaded
    if image1_gray is None:
        print("[ERROR]: Please load image first")
        return

    # Step 2️: Gaussian smoothing
    blur = cv2.GaussianBlur(image1_gray, (3, 3), 0)

    # Step 3️: Apply Sobel X and Sobel Y manually
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    sobel_y = np.array([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]])

    padding = np.zeros((blur.shape[0] + 2, blur.shape[1] + 2))
    padding[1:-1, 1:-1] = blur

    grad_x = np.zeros_like(blur, dtype=np.float32)
    grad_y = np.zeros_like(blur, dtype=np.float32)

    # Step 4️: Convolution manually (pixel by pixel)
    for x in range(blur.shape[0]):
        for y in range(blur.shape[1]):
            grad_x[x, y] = np.sum(padding[x:x+3, y:y+3] * sobel_x)
            grad_y[x, y] = np.sum(padding[x:x+3, y:y+3] * sobel_y)

    # Step 5️: Compute gradient magnitude and angle
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    gradient_angle = (np.degrees(np.arctan2(grad_y, grad_x)) + 360) % 360

    # Step 6️: Create two masks based on angle ranges
    mask1 = ((gradient_angle >= 170) & (gradient_angle <= 190)).astype(np.uint8) * 255
    mask2 = ((gradient_angle >= 260) & (gradient_angle <= 280)).astype(np.uint8) * 255

    # Step 7️: Apply bitwise AND to extract regions in those angle ranges
    result1 = cv2.bitwise_and(magnitude.astype(np.uint8),
                              magnitude.astype(np.uint8),
                              mask=mask1)
    result2 = cv2.bitwise_and(magnitude.astype(np.uint8),
                              magnitude.astype(np.uint8),
                              mask=mask2)

    # Step 8️: Display results
    cv2.imshow("Angle Range [170°, 190°]", result1)
    cv2.imshow("Angle Range [260°, 280°]", result2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def Block4_btn_transforms_clicked():
    """
    Perform 2D affine transformation (rotation, scaling, translation)
    according to user input in GUI.

    Transformation formula:
        M' = M_translation × M_rotation/scale

    Default setup (based on slide example):
        - Rotation: +30° (counter-clockwise)
        - Scale: 0.9
        - Translation: Tx=535, Ty=335
        - Rotation center: (240, 200)
        - Output size: (1920, 1080)

    Result:
        The burger image is rotated, scaled, and translated 
        such that its center moves from (240,200) → (775,535).
    """

    print("Transforms button clicked")

    # Step 1️: Ensure image loaded
    if image1 is None:
        print("[ERROR]: Please load image first")
        return

    # Step 2️: Read user input values
    try:
        rotation = float(Block4_input_rotation.text()) or 0.0
        scale = float(Block4_input_scaling.text()) or 1.0
        tx = float(Block4_input_TX.text()) or 0.0
        ty = float(Block4_input_TY.text()) or 0.0
    except ValueError:
        print("[ERROR]: Invalid input format.")
        return

    print(f"Rotation = {rotation}° | Scale = {scale} | Tx = {tx} | Ty = {ty}")

    # Step 3️: Define image size
    h, w = image1.shape[:2]
    output_size = (1920, 1080)

    # Step 4️: Define rotation+scale matrix (center = (240,200))
    center = (240, 200)
    M_rotate_scale = cv2.getRotationMatrix2D(center, rotation, scale)

    # Step 5️: Extend M to 3×3 for matrix multiplication
    M_rotate_scale = np.vstack([M_rotate_scale, [0, 0, 1]])

    # Step 6️: Define translation matrix
    M_translate = np.float32([[1, 0, tx],
                              [0, 1, ty],
                              [0, 0, 1]])

    # Step 7️: Combine transforms
    M_final = np.matmul(M_translate, M_rotate_scale)

    # Step 8️: Apply affine transform (only first 2 rows used)
    transformed_img = cv2.warpAffine(image1, M_final[:2, :], output_size)

    # Step 9️: Display result
    cv2.imshow("Transformed Image", transformed_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# For Block5
def Block5_btn_load_img_clicked():
    print("Load Image button clicked")


def Block5_btn_5_1_clicked():
    """
    Apply Global Thresholding to a non-uniformly illuminated image.
    Steps:
        1 Convert input image to grayscale.
        2 Apply cv2.threshold() with a fixed threshold value (80).
        3 Display both original and thresholded images for comparison.
    
    Reference:
        threshold_image = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
    """
    print("Global Threshold button clicked")

    # Step 1️: Check if image is loaded
    if image1 is None:
        print("[ERROR]: Please load image first")
        return

    # Step 2️: Convert to grayscale
    gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

    # Step 3️: Apply global threshold (fixed threshold = 80)
    _, th_global = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)

    # Step 4️: Display original and thresholded image
    cv2.imshow("Original Image (QR.png)", gray)
    cv2.imshow("Global Threshold (T=80)", th_global)

    # Wait until key pressed
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def Block5_btn_5_2_clicked():
    """
    Apply Local (Adaptive) Thresholding to a non-uniformly illuminated image.

    Steps:
        1 Convert input image to grayscale.
        2 Apply adaptive threshold using local mean (cv2.ADAPTIVE_THRESH_MEAN_C).
        3 Display both original and local threshold results for comparison.

    Reference:
        threshold_image = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 19, -1
        )
    """
    print("Local (Adaptive) Threshold button clicked")

    # Step 1️: Check if image is loaded
    if image1 is None:
        print("[ERROR]: Please load image first")
        return

    # Step 2️: Convert to grayscale
    gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

    # Step 3️: Apply local adaptive threshold
    th_local = cv2.adaptiveThreshold(
        gray,                      # input grayscale image
        255,                       # max value for pixels above threshold
        cv2.ADAPTIVE_THRESH_MEAN_C, # threshold computed from local mean
        cv2.THRESH_BINARY,         # binary output (black/white)
        19,                        # block size (local neighborhood)
        -1                         # constant subtracted from mean
    )

    # Step 4️: Display results
    cv2.imshow("Original Image (QR.png)", gray)
    cv2.imshow("Local Threshold", th_local)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QPainter, QPixmap

    app = QApplication(sys.argv)
    window = QWidget()
    window.setWindowTitle("Hw1_AN4126018_部政佑_V1")
    window.resize(1000, 1000)

    def set_background(window):
        base_dir = "."
        bg_path = base_dir + "/images/ilovehomework.png"
        pixmap = QPixmap(bg_path)
        if pixmap.isNull():
            print(f"[WARNING] Background not found: {bg_path}")
            return

        old_paint_event = window.paintEvent

        def paintEvent(event):
            old_paint_event(event)
            painter = QPainter(window)
            painter.setRenderHint(QPainter.SmoothPixmapTransform)
            target_height = window.height() // 4
            scaled = pixmap.scaledToHeight(target_height, Qt.SmoothTransformation)
            margin = 20
            x = window.width() - scaled.width() - margin
            y = window.height() - scaled.height() - margin
            painter.setOpacity(0.5)
            painter.drawPixmap(x, y, scaled)
            painter.end()

        window.paintEvent = paintEvent

    set_background(window)

    # --- create main Layout ---
    main_layout = QVBoxLayout()
    top_row_layout = QHBoxLayout()
    bottom_row_layout = QHBoxLayout()
    load_img_layout = QHBoxLayout()

    # --- Global Load Buttons ---
    load_img1_btn = QPushButton("Load Image 1")
    load_img2_btn = QPushButton("Load Image 2")
    load_img_layout.addWidget(load_img1_btn)
    load_img_layout.addWidget(load_img2_btn)

    # --- Block 1 ---
    block1 = QWidget()
    Block1_layout = QVBoxLayout()
    Block1_layout.addWidget(QLabel("1. Color Processing"))
    Block1_btn_1_1 = QPushButton("1.1 Colors Separation")
    Block1_btn_1_2 = QPushButton("1.2 Color Transformation")
    Block1_btn_1_3 = QPushButton("1.3 Color Extraction")
    for btn in [Block1_btn_1_1, Block1_btn_1_2, Block1_btn_1_3]:
        Block1_layout.addWidget(btn)
    block1.setLayout(Block1_layout)

    # --- Block 2 ---
    block2 = QWidget()
    Block2_layout = QVBoxLayout()
    Block2_layout.addWidget(QLabel("2. Image Smoothing"))
    Block2_btn_2_1 = QPushButton("2.1 Gaussian Blur")
    Block2_btn_2_2 = QPushButton("2.2 Bilateral Filter")
    Block2_btn_2_3 = QPushButton("2.3 Median Filter")
    for btn in [Block2_btn_2_1, Block2_btn_2_2, Block2_btn_2_3]:
        Block2_layout.addWidget(btn)
    block2.setLayout(Block2_layout)

    # --- Block 3 ---
    block3 = QWidget()
    Block3_layout = QVBoxLayout()
    Block3_layout.addWidget(QLabel("3. Edge Detection"))
    Block3_btn_3_1 = QPushButton("3.1 Sobel X")
    Block3_btn_3_2 = QPushButton("3.2 Sobel Y")
    Block3_btn_3_3 = QPushButton("3.3 Combination and Threshold")
    Block3_btn_3_4 = QPushButton("3.4 Gradient Angle")
    for btn in [Block3_btn_3_1, Block3_btn_3_2, Block3_btn_3_3, Block3_btn_3_4]:
        Block3_layout.addWidget(btn)
    block3.setLayout(Block3_layout)

    # --- Block 4 ---
    global Block4_input_rotation, Block4_input_scaling, Block4_input_TX, Block4_input_TY
    block4 = QWidget()
    Block4_layout = QVBoxLayout()
    Block4_layout.addWidget(QLabel("4. Transforms"))
    Block4_input_rotation = QLineEdit()
    Block4_input_scaling = QLineEdit()
    Block4_input_TX = QLineEdit()
    Block4_input_TY = QLineEdit()
    for label_text, widget in [
        ("Rotation (deg):", Block4_input_rotation),
        ("Scaling:", Block4_input_scaling),
        ("Tx (pixel):", Block4_input_TX),
        ("Ty (pixel):", Block4_input_TY),
    ]:
        Block4_layout.addWidget(QLabel(label_text))
        Block4_layout.addWidget(widget)
    Block4_btn_transforms = QPushButton("4. Transforms")
    Block4_layout.addWidget(Block4_btn_transforms)
    block4.setLayout(Block4_layout)

    # --- Block 5 ---
    block5 = QWidget()
    Block5_layout = QVBoxLayout()
    Block5_layout.addWidget(QLabel("5. Adaptive Threshold"))
    Block5_btn_5_1 = QPushButton("5.1 Global Threshold")
    Block5_btn_5_2 = QPushButton("5.2 Local Threshold")
    Block5_layout.addWidget(Block5_btn_5_1)
    Block5_layout.addWidget(Block5_btn_5_2)
    block5.setLayout(Block5_layout)

    # --- Combine Layouts ---
    top_row_layout.addWidget(block1)
    top_row_layout.addWidget(block2)
    bottom_row_layout.addWidget(block3)
    bottom_row_layout.addWidget(block4)

    main_layout.addLayout(load_img_layout)
    main_layout.addLayout(top_row_layout)
    main_layout.addLayout(bottom_row_layout)
    main_layout.addWidget(block5)
    window.setLayout(main_layout)

    # --- Connect Buttons ---
    load_img1_btn.clicked.connect(load_img1_btn_clicked)
    load_img2_btn.clicked.connect(load_img2_btn_clicked)
    Block1_btn_1_1.clicked.connect(Block1_btn_1_1_clicked)
    Block1_btn_1_2.clicked.connect(Block1_btn_1_2_clicked)
    Block1_btn_1_3.clicked.connect(Block1_btn_1_3_clicked)
    Block2_btn_2_1.clicked.connect(Block2_btn_2_1_clicked)
    Block2_btn_2_2.clicked.connect(Block2_btn_2_2_clicked)
    Block2_btn_2_3.clicked.connect(Block2_btn_2_3_clicked)
    Block3_btn_3_1.clicked.connect(Block3_btn_3_1_clicked)
    Block3_btn_3_2.clicked.connect(Block3_btn_3_2_clicked)
    Block3_btn_3_3.clicked.connect(Block3_btn_3_3_clicked)
    Block3_btn_3_4.clicked.connect(Block3_btn_3_4_clicked)
    Block4_btn_transforms.clicked.connect(Block4_btn_transforms_clicked)
    Block5_btn_5_1.clicked.connect(Block5_btn_5_1_clicked)
    Block5_btn_5_2.clicked.connect(Block5_btn_5_2_clicked)

    # --- Show Window ---
    window.show()
    sys.exit(app.exec_())



if __name__ == "__main__":
    main()
