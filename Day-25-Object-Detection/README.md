# Day 25 - Object Detection Fundamentals
 
 **Topics Covered:** Classification vs Localization vs Detection, IoU, R-CNN, Fast R-CNN, Faster R-CNN, Region Proposal Network (RPN), mAP
 
 ---
 
 ## Question 1: Classification vs Localization vs Detection
 
 **Topic:** Concept
 **Difficulty:** Basic
 
 ### Question
 Distinguish between these three tasks.
 
 ### Answer
 
 1. **Classification:** "What is in this image?" (Output: Label "Cat").
 2. **Localization:** "Where is it?" (Output: Label "Cat" + Bounding Box $(x, y, w, h)$). Assumes **one** object.
 3. **Detection:** "Where are **all** objects?" (Output: List of [Label, Bbox] for multiple objects: Cats, Dogs, Cars).
 
 ---
 
 ## Question 2: Intersection over Union (IoU)
 
 **Topic:** Metric
 **Difficulty:** Basic
 
 ### Question
 How do we decide if a predicted box is "Correct"? Define IoU.
 
 ### Answer
 
 **Definition:** A measure of overlap between the **Predicted Box (A)** and the **Ground Truth Box (B)**.
 
 $$ \text{IoU} = \frac{\text{Area of Overlap}}{\text{Area of Union}} $$
 
 - **Threshold:** Usually, if IoU > 0.5, we count it as a True Positive. Keys:
    - IoU = 1 (Perfect match).
    - IoU = 0 (No overlap).
 
 ---
 
 ## Question 3: Sliding Window Approach
 
 **Topic:** Concept
 **Difficulty:** Basic
 
 ### Question
 The naive approach to detection is "Sliding Window". Why does it fail?
 
 ### Answer
 
 **Method:** Slide a crop box of size $S \times S$ over the image. Feed every crop to a CNN.
 **Failures:**
 1. **Computational Cost:** Thousands of crops per image = Thousands of CNN forward passes. Too slow.
 2. **Aspect Ratio:** A fixed square window cannot detect a standing person (tall) or a car (wide). You'd need millions of windows of all sizes.
 
 ---
 
 ## Question 4: Region Proposals (Selective Search)
 
 **Topic:** Algorithm
 **Difficulty:** Intermediate
 
 ### Question
 How did traditional "Region Proposal" algorithms (like Selective Search) speed up detection?
 
 ### Answer
 
 - Instead of sliding blindly, use a **feature-based** algorithm (grouping pixels by color/texture) to suggest likely places where objects exist.
 - Reduces candidates from 1,000,000 sliding windows to ~2,000 "Region Proposals".
 - **Used in:** R-CNN.
 - **Cons:** Selective Search is a slow CPU algorithm.
 
 ---
 
 ## Question 5: R-CNN Family Evolution
 
 **Topic:** Architecture
 **Difficulty:** Advanced
 
 ### Question
 Briefly track the evolution from R-CNN -> Fast R-CNN -> Faster R-CNN.
 
 ### Answer
 
 1. **R-CNN (2014):**
    - Crop 2000 regions -> Resize -> Feed to CNN **2000 times**. (Very Slow).
 2. **Fast R-CNN (2015):**
    - Feed whole image to CNN **once** -> Get Feature Map.
    - Project region proposals onto the Feature Map (RoI Pooling).
    - Speed up: 100x FASTER than R-CNN.
    - Bottle-neck: Region proposal (Selective Search) is still slow.
 3. **Faster R-CNN (2016):**
    - Replaced Selective Search with a Neural Network (**Region Proposal Network - RPN**).
    - Everything is now a single end-to-end differentiable network.
    - **State of the Art (Two-stage).**
 
 ---
 
 ## Question 6: Region Proposal Network (RPN)
 
 **Topic:** Architecture
 **Difficulty:** Advanced
 
 ### Question
 What is the RPN in Faster R-CNN? How does it suggest boxes?
 
 ### Answer
 
 **Role:** A small network that slides over the Feature Map.
 **Mechanism:**
 - At every location, it predicts:
    1. **Objectness Score:** Probability that an object exists here.
    2. **Box Refinements:** Adjustments to $k$ default "Anchor Boxes".
 - It tells the second stage "Look here, there might be something!"
 
 ---
 
 ## Question 7: RoI (Region of Interest) Pooling
 
 **Topic:** Architecture Component
 **Difficulty:** Advanced
 
 ### Question
 Fast R-CNN introduced RoI Pooling. What problem does it solve?
 
 ### Answer
 
 **Problem:**
 - Region Proposals are of different sizes ($50 \times 50$, $100 \times 30$).
 - Fully Connected layers typically require fixed input size (e.g., $7 \times 7$).
 
 **Solution:**
 - Projects the arbitrary region onto the feature map.
 - Divides it into a grid of $7 \times 7$ sections.
 - Performs Max Pooling in each section.
 - **Result:** Always outputs a fixed $7 \times 7$ tensor, regardless of input proposal size.
 
 ---
 
 ## Question 8: Non-Maximum Suppression (NMS)
 
 **Topic:** Post-processing
 **Difficulty:** Intermediate
 
 ### Question
 A detector often outputs 10 boxes for the same "Cat". How does NMS clean this up?
 
 ### Answer
 
 **Algorithm:**
 1. Sort all boxes by Confidence Score.
 2. Pick the box with the highest score (Top Box).
 3. Compare it with all other boxes. If **IoU(Top Box, Other Box) > threshold** (e.g., 0.5), discard the other box (suppress it).
 4. Repeat until no boxes remain.
 5. **Result:** Only the single best box for each object remains.
 
 ---
 
 ## Question 9: mAP (Mean Average Precision)
 
 **Topic:** Metric
 **Difficulty:** Intermediate
 
 ### Question
 Accuracy is not used in detection. Why? Explain mAP.
 
 ### Answer
 
 **Why no Accuracy:** "Background" class dominates. A model predicting "No Object" everywhere has 99% accuracy but 0 utility.
 
 **mAP Calculation:**
 1. Calculate **Precision-Recall Curve** for each class (at a specific IoU threshold, e.g., 0.5).
 2. Calculate **Average Precision (AP)** = Area Under Curve (AUC).
 3. **mAP** = Mean of AP across all classes (Cat, Dog, Car).
 - mAP@50: mAP at IoU 0.5.
 - mAP@[0.5:0.95]: Average of mAP at steps 0.5, 0.55... 0.95 (COCO Metric).
 
 ---
 
 ## Question 10: One-Stage vs Two-Stage Detectors
 
 **Topic:** High Level Design
 **Difficulty:** Basic
 
 ### Question
 What is the trade-off between Two-Stage (Faster R-CNN) and One-Stage (YOLO/SSD) detectors?
 
 ### Answer
 
 | Type | Examples | Pros | Cons |
 |------|----------|------|------|
 | **Two-Stage** | R-CNN, Faster R-CNN | **High Accuracy**. Better at small objects. | **Slow** (low FPS). Complex to train. |
 | **One-Stage** | YOLO, SSD, RetinaNet | **Real-time Speed**. Simple architecture. | Slightly lower accuracy (historically), struggles with small/crowded objects. |
 
 *Note: Modern YOLOv8 matches or beats Faster R-CNN accuracy.*
 
 ---
 
 ## Key Takeaways
 
 - **IoU** defines what counts as a "Hit".
 - **R-CNN** family represents the "Accuracy First" lineage.
 - **RPN** allows the network to learn *where* to look.
 - **NMS** cleans up duplicate predictions.
 - **mAP** is the gold standard metric.
 
 **Next:** [Day 26 - YOLO](../Day-26/README.md)
