# Day 27 - Semantic Segmentation & U-Net
 
 **Topics Covered:** Semantic vs Instance Segmentation, Fully Convolutional Networks (FCN), U-Net Architecture, Skip Connections, Dice Coefficient, IoU for Segmentation
 
 ---
 
 ## Question 1: Segmentation Types
 
 **Topic:** Concept
 **Difficulty:** Basic
 
 ### Question
 Difference between Semantic, Instance, and Panoptic Segmentation?
 
 ### Answer
 
 - **Semantic Segmentation:** "Label every pixel."
    - All "Cars" are red. All "people" are blue.
    - Cannot distinguish between different cars (e.g., Car 1 and Car 2 are merged into one blob).
 - **Instance Segmentation:** "Label objects and distinguish individuals."
    - Car 1 is red. Car 2 is green. Person 1 is blue.
    - Background is ignored.
 - **Panoptic Segmentation:** Combined. Label *everything* (Background = Semantic, Objects = Instance).
 
 ---
 
 ## Question 2: Fully Convolutional Networks (FCN)
 
 **Topic:** Architecture
 **Difficulty:** Intermediate
 
 ### Question
 How did FCN (2015) enable segmentation using standard classification networks like VGG?
 
 ### Answer
 
 - **Problem:** Classification nets end with Dense layers, which lose spatial info (outputs a single vector 1x1000).
 - **FCN Solution:**
    1. Replace Dense layers with **1x1 Conv** layers.
    2. The output is now a low-resolution map (e.g., $7 \times 7 \times 1000$).
    3. **Upsample** (Transpose Conv) the map back to original image size ($224 \times 224$).
 - **Result:** A pixel-wise map of predictions.
 
 ---
 
 ## Question 3: U-Net Architecture
 
 **Topic:** Architecture
 **Difficulty:** Advanced
 
 ### Question
 Draw the U-Net shape mentally. Why is it the standard for Medical Imaging?
 
 ### Answer
 
 **Shape:** 'U' shaped.
 - **Contracting Path (Encoder):** Captures context/features (Downsampling: Conv + MaxPool).
 - **Expansive Path (Decoder):** Precise localization (Upsampling: Transpose Conv).
 
 **The Killer Feature:**
 - **Skip Connections:** Concatenate high-resolution feature maps from the Encoder directly to the Decoder.
 - Allows the decoder to recover **fine details** (edges) lost during pooling. Critical for precise tumor/cell boundaries.
 
 ---
 
 ## Question 4: Transposed Convolution
 
 **Topic:** Operation
 **Difficulty:** Intermediate
 
 ### Question
 How do we increase resolution (Upsampling) with learnable parameters?
 
 ### Answer
 
 **Transposed Convolution (Deconvolution):**
 - Not just mathematical interpolation (like Bilinear resizing).
 - It learns weights to "paint" pixels onto a larger grid.
 - **Operation:** Takes a single pixel value, multiplies it by a filter kernel, and projects it onto a $2 \times 2$ or $3 \times 3$ output patch.
 
 ---
 
 ## Question 5: Evaluation Metrics (IoU vs Dice)
 
 **Topic:** Metric
 **Difficulty:** Intermediate
 
 ### Question
 Why is Accuracy a terrible metric for segmentation? What is Dice Coefficient?
 
 ### Answer
 
 **Accuracy Problem:** If 99% of pixels are background and 1% are cancer, a model predicting "All Background" has 99% accuracy but 0 utility.
 
 **Dice Coefficient (F1 Score equivalent):**
 
 $$ \text{Dice} = \frac{2 \times |A \cap B|}{|A| + |B|} $$
 
 - Overlap between Prediction (A) and Ground Truth (B).
 - Ranges 0 to 1. 1 = Perfect Overlap.
 - **IoU** = $\frac{|A \cap B|}{|A \cup B|}$. They are positively correlated.
 
 ---
 
 ## Question 6: Loss Functions
 
 **Topic:** Math
 **Difficulty:** Advanced
 
 ### Question
 What loss function do we use? Standard Cross-Entropy?
 
 ### Answer
 
 **Pixel-wise Cross Entropy:** Yes, calculate classification loss for every single pixel and sum it up.
 
 **Problem:** Class Imbalance (Small object, huge background).
 
 **Better Losses:**
 1. **Weighted Cross Entropy:** Penalize mistakes on the "Object" class 10x more.
 2. **Dice Loss:** Minimize $(1 - \text{Dice})$. Directly optimizes overlap.
 3. **Focal Loss:** Heavily penalizes "hard" examples.
 
 ---
 
 ## Question 7: Mask R-CNN
 
 **Topic:** Architecture
 **Difficulty:** Advanced
 
 ### Question
 How does Mask R-CNN achieve Instance Segmentation?
 
 ### Answer
 
 It extends Faster R-CNN.
 - **Branch 1:** Class Label (Cat).
 - **Branch 2:** Bounding Box (Coordinates).
 - **Branch 3 (New):** **Binary Mask**.
    - For each proposed box (RoI), it runs a small FCN to predict a mask *inside that box*.
    - "RoIAlign" layer fixes misalignment issues in RoI Pooling to ensure pixel-perfect masking.
 
 ---
 
 ## Question 8: Dilated (Atrous) Convolutions
 
 **Topic:** Operation
 **Difficulty:** Advanced
 
 ### Question
 Segmentation models like DeepLab use Dilated Convolutions. Why?
 
 ### Answer
 
 - **Goal:** Increase the Receptive Field (see more context) *without* downsampling (pooling).
 - **Method:** Insert "holes" (zeros) between kernel pixels.
 - **Result:** A $3 \times 3$ kernel with dilation rate 2 covers a $5 \times 5$ area but still has only 9 parameters.
 - Keeps feature maps large (high resolution) for better segmentation.
 
 ---
 
 ## Question 9: Data Annotation
 
 **Topic:** Practicality
 **Difficulty:** Basic
 
 ### Question
 Why is Segmentation so expensive to train?
 
 ### Answer
 
 **Annotation Cost.**
 - **Classification:** Human says "Cat". (1 second).
 - **Bounding Box:** Human draws a box. (5 seconds).
 - **Segmentation:** Human must carefully trace the exact outline of the cat, pixel by pixel. (5-10 minutes per image).
 - **Solution:** Use synthetic data or AI-assisted labeling tools (SAM - Segment Anything Model).
 
 ---
 
 ## Question 10: U-Net Implementation (Keras)
 
 **Topic:** Implementation
 **Difficulty:** Advanced
 
 ### Question
 Write pseudo-code for a U-Net blocks.
 
 ### Answer
 
 ```python
 inputs = Input((256, 256, 1))
 
 # Encoder
 c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
 p1 = MaxPool2D((2, 2))(c1)
 
 # Bridge
 c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
 
 # Decoder
 u1 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c2)
 concat = Concatenate()([u1, c1]) # SKIP CONNECTION
 c3 = Conv2D(64, (3, 3), activation='relu', padding='same')(concat)
 
 outputs = Conv2D(1, (1, 1), activation='sigmoid')(c3)
 ```
 
 ---
 
 ## Key Takeaways
 
 - **Type:** Semantic (Classes) vs Instance (Objects).
 - **U-Net:** Encoder-Decoder with Skip Connections (Standard for BioMed).
 - **Metrics:** Accuracy fails; Use **Dice** or **IoU**.
 - **Mask R-CNN:** Adds a mask branch to object detection.
 
 **Next:** [Day 28 - Generative CV](../Day-28/README.md)
