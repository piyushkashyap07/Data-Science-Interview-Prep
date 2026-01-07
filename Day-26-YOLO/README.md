# Day 26 - YOLO (You Only Look Once)
 
 **Topics Covered:** Grid-based detection, Anchor Boxes, One-Stage Architecture, YOLO Loss Function, Evolution (v1 to v8)
 
 ---
 
 ## Question 1: The YOLO Philosophy
 
 **Topic:** Concept
 **Difficulty:** Basic
 
 ### Question
 How did YOLO (v1) fundamentally change object detection compared to R-CNN?
 
 ### Answer
 
 **R-CNN:** Treat detection as a **Classification** problem on region proposals. (Look 2000 times).
 **YOLO:** Treat detection as a **Regression** problem. (Look 1 time).
 - It sees the entire image at once.
 - It predicts Bounding Boxes and Class Probabilities directly from full images in a single forward pass.
 - **Result:** Real-time speed (45 FPS vs 7 FPS).
 
 ---
 
 ## Question 2: The Grid System
 
 **Topic:** Architecture
 **Difficulty:** Intermediate
 
 ### Question
 Explain the $S \times S$ grid in YOLOv1. Who is responsible for detecting an object?
 
 ### Answer
 
 - The input image is divided into an $S \times S$ grid (e.g., $7 \times 7$ = 49 cells).
 - **Rule:** If the **center** of an object falls into a grid cell, **that specific cell** is responsible for detecting it.
 - Each cell predicts:
    1. $B$ Bounding Boxes $(x, y, w, h)$.
    2. Confidence Score (Is there an object?).
    3. Classification Probabilities (Cat vs Dog).
 
 ---
 
 ## Question 3: Output Tensor Shape
 
 **Topic:** Architecture
 **Difficulty:** Advanced
 
 ### Question
 If Grid $S=7$, Boxes per cell $B=2$, and Classes $C=20$. What is the shape of the final output tensor?
 
 ### Answer
 
 Shape: $(S, S, B \times 5 + C)$
 
 - **$7 \times 7$**: The grid.
 - **$B \times 5$**: For each of the 2 boxes, we predict $(x, y, w, h, \text{confidence})$. Total 10 values.
 - **$C$**: One set of 20 class probabilities for the cell.
 - **Total:** $(7, 7, 30)$.
 
 ---
 
 ## Question 4: Anchor Boxes
 
 **Topic:** Concept
 **Difficulty:** Intermediate
 
 ### Question
 YOLOv2 introduced Anchor Boxes (Priors). Why?
 
 ### Answer
 
 - **Problem (v1):** Predicting exact width/height from scratch is unstable. The model struggles to specialize for "tall" (person) vs "flat" (car) shapes.
 - **Solution (v2):** Define standard template shapes (Anchors) derived from K-means clustering on the dataset.
    - Anchor 1: Tall/Thin.
    - Anchor 2: Short/Wide.
 - The network now predicts **offsets** (shifts) from these anchors instead of raw coordinates. This is much easier to learn.
 
 ---
 
 ## Question 5: YOLO Loss Function
 
 **Topic:** Math
 **Difficulty:** Advanced
 
 ### Question
 The YOLO Loss function is a sum of three parts. What are they?
 
 ### Answer
 
 1. **Localization Loss:** (Regression) Mean Squared Error of box coordinates $(x, y, w, h)$.
 2. **Confidence Loss:** (Binary Cross Entropy)
    - Object exists: 1 vs Predicted Conf.
    - No Object: 0 vs Predicted Conf.
 3. **Classification Loss:** (Cross Entropy) Probability of correct class.
 
 *Note: YOLO weighs these parts differently (e.g., penalizes 'No Object' errors less to handle class imbalance).*
 
 ---
 
 ## Question 6: Evolution (v1 to v8)
 
 **Topic:** General Knowledge
 **Difficulty:** Intermediate
 
 ### Question
 Briefly highlight one key improvement in YOLOv3, v5, and v8.
 
 ### Answer
 
 - **YOLOv3:** Detects at **3 different scales** (Small, Medium, Large heads) using a Feature Pyramid Network (FPN). Solved the "small object" issue.
 - **YOLOv5:** Transition to PyTorch (ultralytics). Auto-anchor generation. Mosaic Data Augmentation.
 - **YOLOv8:** Anchor-free detection (predicts center + distance to walls directly). SOTA accuracy and speed.
 
 ---
 
 ## Question 7: Intersection over Union (IoU) in Training
 
 **Topic:** Training
 **Difficulty:** Intermediate
 
 ### Question
 How does YOLO decide which anchor box is "responsible" for an object during training?
 
 ### Answer
 
 - We have 3 (or 9) anchor boxes at a grid cell.
 - We compare each anchor's IoU with the **Ground Truth** box.
 - The anchor with the **Highest IoU** is assigned the responsibility.
 - It calculates loss against ground truth. Other anchors are told "ignore this object".
 
 ---
 
 ## Question 8: Running YOLO (Inference Code)
 
 **Topic:** Implementation
 **Difficulty:** Basic
 
 ### Question
 Write a snippet to run a pretrained YOLOv8 model using `ultralytics`.
 
 ### Answer
 
 ```python
 from ultralytics import YOLO
 import cv2
 
 # 1. Load model
 model = YOLO('yolov8n.pt')  # n = nano (smallest)
 
 # 2. Predict on image
 results = model('bus.jpg')
 
 # 3. Show Result
 for r in results:
     im_array = r.plot()  # Plot predictions onto image
     cv2.imwrite("output.jpg", im_array)
 ```
 
 ---
 
 ## Question 9: Limitations of YOLO
 
 **Topic:** Concept
 **Difficulty:** Basic
 
 ### Question
 Where does YOLO still struggle compared to Two-Stage detectors?
 
 ### Answer
 
 1. **Small Objects in Groups:** Because of the grid constraint, a single cell can only predict $B$ (e.g., 2) boxes. If a flock of birds has 5 birds in one grid cell, YOLO can only see 2.
 2. **Precision:** Although gap is closing, Faster R-CNN often produces slightly tighter, more precise boxes for high-stakes applications (e.g., medical imaging).
 
 ---
 
 ## Question 10: Mosaic Augmentation
 
 **Topic:** Data Augmentation
 **Difficulty:** Intermediate
 
 ### Question
 YOLOv4 introduced Mosaic Augmentation. What is it?
 
 ### Answer
 
 - **Method:** Stitch **4 training images** together into one giant grid.
 - **Benefit:**
    1. The model sees 4 different contexts at once.
    2. Allows detecting objects outside their normal context.
    3. Solves the Batch Normalization problem (simulates larger batch size).
 
 ---
 
 ## Key Takeaways
 
 - **YOLO** treats detection as a regression problem.
 - **Grid Cell** responsibility is the core concept.
 - **Anchor Boxes** help specialize predictions for shapes.
 - **Multi-scale prediction** (FPN) allows detecting small and large objects.
 - **Real-time Performance** makes YOLO the industry standard for video.
 
 **Next:** [Day 27 - Semantic Segmentation](../Day-27/README.md)
