# Day 24 - Transfer Learning
 
 **Topics Covered:** Pre-trained Models, Fine-tuning, Feature Extraction, Freezing Layers, Data Augmentation, Handling Size Mismatch
 
 ---
 
 ## Question 1: What is Transfer Learning?
 
 **Topic:** Concept
 **Difficulty:** Basic
 
 ### Question
 Why do we use Transfer Learning instead of training CNNs from scratch?
 
 ### Answer
 
 **Definition:** Taking a model trained on a large dataset (Source Domain: ImageNet, 14M images) and adapting it to a smaller dataset (Target Domain: Medical X-Rays, 1k images).
 
 **Why?**
 - **Data Scarcity:** Deep CNNs need millions of images to learn low-level features (edges, textures). Small datasets lead to overfitting.
 - **Speed:** Training ResNet-50 from scratch takes days on GPUs. Fine-tuning takes minutes.
 - **Performance:** A model that knows "shapes" (from ImageNet) learns "tumors" faster than a random model.
 
 ---
 
 ## Question 2: Feature Extraction vs Fine-Tuning
 
 **Topic:** Strategy
 **Difficulty:** Intermediate
 
 ### Question
 Explain the two main strategies of Transfer Learning. When to use which?
 
 ### Answer
 
 1. **Feature Extraction (Freeze Base):**
    - **Method:** Freeze the convolutional base. Only train the new Classification Head (Dense layers).
    - **Use Case:** Small dataset + Similar domain (e.g., ImageNet -> Caltech-101).
 
 2. **Fine-Tuning (Unfreeze Base):**
    - **Method:** Train the head first, then unfreeze some top layers of the base and train with a very low learning rate.
    - **Use Case:** Large dataset OR Different domain (e.g., ImageNet -> Satellite Imagery).
 
 ---
 
 ## Question 3: Freezing Layers
 
 **Topic:** Implementation
 **Difficulty:** Basic
 
 ### Question
 How do you "freeze" a layer in detailed PyTorch/Keras?
 
 ### Answer
 
 **Keras:**
 ```python
 base_model = ResNet50(weights='imagenet', include_top=False)
 for layer in base_model.layers:
     layer.trainable = False
 ```
 
 **PyTorch:**
 ```python
 model = models.resnet50(pretrained=True)
 for param in model.parameters():
     param.requires_grad = False
 ```
 
 ---
 
 ## Question 4: Size Mismatch
 
 **Topic:** Implementation
 **Difficulty:** Intermediate
 
 ### Question
 ImageNet models expect $224 \times 224$ images. Your dataset has $1000 \times 1000$ images. What should you do?
 
 ### Answer
 
 1. **Resize (Downsample):** `cv2.resize(img, (224, 224))`. Fastest, but loses small details.
 2. **Crop:** Take the center crop or random crops of $224 \times 224$. Keeps resolution high but might miss objects at edges.
 3. **Global Average Pooling:** If the model consists only of Conv layers + GAP, it can mathematically accept *any* size. However, the *receptive field* learned features at scale 224 might not translate to scale 1000.
 
 **Best Practice:** Resize to something close (e.g., 256 or 512) and fine-tune.
 
 ---
 
 ## Question 5: Catastrophic Forgetting
 
 **Topic:** Concept
 **Difficulty:** Advanced
 
 ### Question
 What happens if you fine-tune the whole network with a large learning rate immediately?
 
 ### Answer
 
 **Catastrophic Forgetting.**
 - The new Classification Head is initialized with random weights.
 - Initial gradients will be massive (high error).
 - These massive gradients verify propagate back to the delicate Conv filters (which were carefully tuned for ImageNet) and destroy them.
 - **Fix:** Always Freeze the base -> Train Head -> Unfreeze Base (Low LR).
 
 ---
 
 ## Question 6: Data Augmentation
 
 **Topic:** Technique
 **Difficulty:** Basic
 
 ### Question
 Why is Data Augmentation critical in Transfer Learning? Name 3 techniques.
 
 ### Answer
 
 **Why:** Transfer learning is usually done on small datasets. Augmentation artificially expands the dataset size to prevent overfitting.
 
 **Techniques:**
 1. **Geometric:** Rotation, Horizontal Flip, Zoom.
 2. **Color:** Brightness jitter, Contrast, Saturation.
 3. **Advanced:**
    - **Cutout:** Randomly masking a square region.
    - **Mixup:** Blending two images: $Im_{new} = 0.7 * Cat + 0.3 * Dog$. Label = $0.7$.
 
 ---
 
 ## Question 7: Pre-processing match
 
 **Topic:** Implementation
 **Difficulty:** Intermediate
 
 ### Question
 A common bug: You load ResNet50, feed your images $(0-255)$, and get 0.1% accuracy. Why?
 
 ### Answer
 
 **Normalization Mismatch.**
 - ResNet-50 (ImageNet) was trained with specific pre-processing:
    - `preprocess_input` typically converts RGB -> BGR.
    - Zero-centers each channel (subtract Mean [103.939, 116.779, 123.68]).
 - VGG might expect 0-1 scale; Inception might expect -1 to 1.
 - **Fix:** *Always* use the specific `preprocess_input` function provided by the library for that model.
 
 ---
 
 ## Question 8: Domain Adaptation
 
 **Topic:** Concept
 **Difficulty:** Advanced
 
 ### Question
 Transfer Learning works best when Source and Target domains are similar. What if they are totally different (e.g., Day photos -> Night photos)?
 
 ### Answer
 
 **Domain Adaptation:**
 - The feature distribution changes ($P(X_{source}) \neq P(X_{target})$).
 - **Technique:** GANs (CycleGAN) to translate Source -> Target style.
 - Use Unsupervised Domain Adaptation to align the feature distributions of source and target.
 
 ---
 
 ## Question 9: Fine-tuning Strategy (ULMFiT)
 
 **Topic:** Strategy
 **Difficulty:** Advanced
 
 ### Question
 What is "Discriminative Fine-tuning" (introduced in ULMFiT)?
 
 ### Answer
 
 - Instead of using the same Learning Rate (LR) for all layers, use **different LRs**.
 - **Early layers** (Edges/gradients): Very low LR (Don't change much).
 - **Middle layers:** Medium LR.
 - **Last layers:** High LR (Task specific).
 - `optim = Adam([{'params': model.base.parameters(), 'lr': 1e-5}, {'params': model.head.parameters(), 'lr': 1e-3}])`
 
 ---
 
 ## Question 10: Code Example
 
 **Topic:** Implementation
 **Difficulty:** Implementation
 
 ### Question
 Write a Keras script to use MobileNetV2 for classification on a custom dataset.
 
 ### Answer
 
 ```python
 from tensorflow.keras.applications import MobileNetV2
 from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
 from tensorflow.keras.models import Model
 
 # 1. Load Base
 base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
 base.trainable = False # Freeze
 
 # 2. Add Head
 x = base.output
 x = GlobalAveragePooling2D()(x)
 x = Dense(1024, activation='relu')(x)
 predictions = Dense(10, activation='softmax')(x) # 10 classes
 
 # 3. Compile
 model = Model(inputs=base.input, outputs=predictions)
 model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
 
 # 4. Train
 # model.fit(...)
 ```
 
 ---
 
 ## Key Takeaways
 
 - **Transfer Learning** is the standard for 99% of real-world CV tasks.
 - **Freeze first, Unfreeze later.**
 - **Preprocessing** must match the original training precisely.
 - **Data Augmentation** fights the small-data overfitting monster.
 
 **Next:** [Day 25 - Object Detection](../Day-25/README.md)
