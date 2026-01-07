# Day 22 - Computer Vision Basics with OpenCV
 
 **Topics Covered:** Image Representation (pixels), Reading/Writing Images, Color Spaces (RGB, BGR, HSV), Resizing, Thresholding, Blurring, Edge Detection.
 
 ---
 
 ## Question 1: How Computers See Images
 
 **Topic:** Concept
 **Difficulty:** Basic
 
 ### Question
 How is a grayscale image represented differently from a color image in memory (using NumPy)?
 
 ### Answer
 
 - **Grayscale Image:** A 2D Matrix $(H, W)$. Each value is a pixel intensity from 0 (Black) to 255 (White).
 - **Color Image (RGB):** A 3D Tensor $(H, W, 3)$. The 3 channels correspond to Red, Green, and Blue intensities.
 - **Shape:**
    - Grayscale: `(Height, Width)`
    - Color: `(Height, Width, 3)`
 
 ---
 
 ## Question 2: Reading Images with OpenCV
 
 **Topic:** Implementation
 **Difficulty:** Basic
 
 ### Question
 Write code to read an image called "cat.jpg" using OpenCV and check its shape. What is the default color format in OpenCV?
 
 ### Answer
 
 ```python
 import cv2
 
 # Read image
 img = cv2.imread('cat.jpg')
 
 if img is not None:
     print("Shape:", img.shape)
 else:
     print("Image not found")
 ```
 
 **Default Format:** OpenCV reads images in **BGR** (Blue-Green-Red) format by default, not RGB. This is a historical quirk. matplotlib uses RGB.
 
 ---
 
 ## Question 3: Color Spaces (HSV vs RGB)
 
 **Topic:** Concept
 **Difficulty:** Intermediate
 
 ### Question
 Why do we often convert images to HSV (Hue, Saturation, Value) for object detection based on color?
 
 ### Answer
 
 **Problem with RGB:**
 - In RGB, "Red" is a mix of (255, 0, 0).
 - If a shadow falls on the red object, it becomes (100, 0, 0).
 - All three channels change significantly with lighting changes.
 
 **Advantage of HSV:**
 - **Hue:** Represents the *color* itself (0-179 in OpenCV).
 - **Saturation:** How "pure" the color is.
 - **Value:** How bright it is.
 - To detect a red ball, you only need to threshold the **Hue** channel ($H \approx 0$ or $179$). The detection is robust to shadows (Value changes) and lighting.
 
 ---
 
 ## Question 4: Resizing and Interpolation
 
 **Topic:** Implementation
 **Difficulty:** Basic
 
 ### Question
 What happens when you resize a $100 \times 100$ image to $200 \times 200$? What is interpolation?
 
 ### Answer
 
 **Resizing (Upscaling):** You are creating new pixels where none existed.
 **Interpolation:** The method to calculate the color of these new pixels based on neighbors.
 - `cv2.INTER_NEAREST`: Fast, blocky (pixelated).
 - `cv2.INTER_LINEAR`: Default, smooth. Good for upscaling.
 - `cv2.INTER_CUBIC`: Slow, very smooth. Best quality.
 
 ```python
 # Resize to half
 resized = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
 ```
 
 ---
 
 ## Question 5: Thresholding
 
 **Topic:** Implementation
 **Difficulty:** Intermediate
 
 ### Question
 What is Binary Thresholding? Write code to convert a grayscale image into a binary mask (Black/White) where pixels > 127 become white.
 
 ### Answer
 
 **Concept:** Converting a continuous grayscale image into a strict binary image (0 or 255) based on a cutoff value.
 
 ```python
 gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
 # Threshold: val > 127 ? 255 : 0
 _, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
 ```
 
 ---
 
 ## Question 6: Gaussian Blur vs Median Blur
 
 **Topic:** Concept
 **Difficulty:** Intermediate
 
 ### Question
 Both blur images. When would you use Median Blur over Gaussian Blur?
 
 ### Answer
 
 - **Gaussian Blur:** Averages pixels using a weighted bell curve. Good for removing general **Gaussian noise** (random smooth noise).
 - **Median Blur:** Replaces a pixel with the median of its neighbors.
    - **Use Case:** it is excellent for removing **"Salt and Pepper" noise** (random white/black dots).
    - It preserves edges better than Gaussian blur.
 
 ---
 
 ## Question 7: Canny Edge Detection
 
 **Topic:** Algorithm
 **Difficulty:** Advanced
 
 ### Question
 Explain the steps of the Canny Edge Detector algorithm.
 
 ### Answer
 
 1. **Noise Reduction:** Apply Gaussian blur to smooth the image.
 2. **Gradient Calculation:** Find intensity gradients (Sobel x and y) to detect where colors change primarily.
 3. **Non-Maximum Suppression:** Thin the edges. If a pixel isn't the local maximum in the gradient direction, set it to 0. (Makes lines 1-pixel wide).
 4. **Hysteresis Thresholding:** Use two thresholds (High, Low).
    - If gradient > High: Strong Edge (Keep).
    - If gradient < Low: Not Edge (Discard).
    - If Low < gradient < High: Weak Edge (Keep *only if* connected to a Strong Edge).
 
 ---
 
 ## Question 8: Finding Contours
 
 **Topic:** Implementation
 **Difficulty:** Intermediate
 
 ### Question
 What are Contours? How do they differ from Edges?
 
 ### Answer
 
 - **Edges:** Local changes in intensity (points). Example: Canny output is just a map of white dots.
 - **Contours:** Continuous curves joining all points along a boundary of the same color/intensity. OpenCV stores them as **Lists of (x, y) coordinates**.
 - **Usage:** Contours are "objects". You can calculate Area, Perimeter, Centroid, and Bounding Box of a contour.
 
 ```python
 contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
 cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
 ```
 
 ---
 
 ## Question 9: Perspective Transform
 
 **Topic:** Advanced Implementation
 **Difficulty:** Advanced
 
 ### Question
 What is a Perspective Transform? Give a real-world example.
 
 ### Answer
 
 **Concept:** Changing the perspective of an image (3D projection -> 2D).
 - **Example:** Document Scanner App.
    - You take a photo of a receipt from an angle. It looks like a trapezoid.
    - **Perspective Transform** warps it to look like a flat top-down rectangle (Bird's eye view).
 - Requires 4 points: Top-Left, Top-Right, Bottom-Right, Bottom-Left.
 
 ---
 
 ## Question 10: Video Capturing
 
 **Topic:** Implementation
 **Difficulty:** Basics
 
 ### Question
 Write a boilerplate script to read frames from a webcam and display them until 'q' is pressed.
 
 ### Answer
 
 ```python
 import cv2
 
 cap = cv2.VideoCapture(0) # 0 = default camera
 
 while True:
     ret, frame = cap.read() # Read one frame
     if not ret:
         break
         
     cv2.imshow('Webcam', frame)
     
     # Wait 1ms. If key is 'q', break
     if cv2.waitKey(1) & 0xFF == ord('q'):
         break
         
 cap.release()
 cv2.destroyAllWindows()
 ```
 
 ---
 
 ## Key Takeaways
 
 - **OpenCV** uses BGR format. Always convert to RGB for matplotlib/model input.
 - **Grayscale** simplifies many tasks (Edge detection, Thresholding).
 - **HSV** is superior for color-based segmentation.
 - **Blurring** (Gaussian/Median) is a critical pre-processing step to reduce noise before detection.
 - **Contours** allow shape analysis (Area, Perimeter).
 
 **Next:** [Day 23 - Modern CNN Architectures](../Day-23/README.md)
