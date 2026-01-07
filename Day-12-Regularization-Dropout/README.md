# Day 12 - Regularization & Dropout
 
 **Topics Covered:** Overfitting, L1/L2 Regularization, Dropout, Early Stopping, Data Augmentation
 
 ---
 
 ## Question 1: Overfitting in Deep Learning
 
 **Topic:** Concepts
 **Difficulty:** Basic
 
 ### Question
 What are the signs of overfitting in a deep neural network? How does model capacity relate to overfitting?
 
 ### Answer
 
 **Signs of Overfitting:**
 - **Training Loss:** Keeps decreasing (Very low).
 - **Validation Loss:** Decreases initially, then starts increasing.
 - **Gap:** Large gap between Training Accuracy (high) and Validation Accuracy (low).
 - **Behavior:** The model memorizes the training noise instead of learning general patterns.
 
 **Model Capacity:**
 - Capacity = The ability of a model to fit complex functions (related to number of parameters/layers).
 - **High Capacity (Deep/Wide Networks):** Prone to overfitting on small datasets.
 - **Low Capacity (Shallow/Narrow Networks):** Prone to underfitting (high bias).
 
 ---
 
 ## Question 2: L1 vs L2 Regularization (Weight Decay)
 
 **Topic:** Regularization
 **Difficulty:** Intermediate
 
 ### Question
 How do L1 and L2 regularization differ in their effect on the weights? Which one leads to sparse models?
 
 ### Answer
 
 **1. L2 Regularization (Ridge / Weight Decay):**
 - **Penalty:** Adds sum of squared weights to loss: $\lambda \sum w^2$.
 - **Effect:** Pushes weights towards zero but rarely *exactly* zero.
 - **Result:** Diffuse, small weights. Good for preventing large outliers in weights.
 - **Gradient:** Linear shrinkage ($w_{new} = (1 - \alpha\lambda)w_{old}$).
 
 **2. L1 Regularization (Lasso):**
 - **Penalty:** Adds sum of absolute weights to loss: $\lambda \sum |w|$.
 - **Effect:** Pushes weights to *exactly* zero if they are not important.
 - **Result:** **Sparse models** (Feature selection).
 - **Gradient:** Constant shift.
 
 ---
 
 ## Question 3: Dropout
 
 **Topic:** Regularization
 **Difficulty:** Intermediate
 
 ### Question
 Explain how Dropout works. Why is it conceptually effectively training an ensemble of networks?
 
 ### Answer
 
 **Mechanism:**
 - During training, randomly "drop" (set to zero) a fraction $p$ (e.g., 0.5) of neurons in a layer.
 - Do this for every forward/backward pass.
 - During testing/inference, use **all** neurons but scale weights by $(1-p)$ (or scale up during training).
 
 **Ensemble Interpretation:**
 - A network with $N$ units has $2^N$ possible sub-networks.
 - Each training step effectively trains a different random sub-network.
 - These sub-networks share weights.
 - At test time, using the full network approximates averaging the predictions of these exponentially many sub-networks (Ensemble averaging).
 - Prevents neurons from co-adapting (relying too much on specific peers).
 
 ---
 
 ## Question 4: Early Stopping
 
 **Topic:** Training
 **Difficulty:** Basic
 
 ### Question
 What is Early Stopping? Why is it considered a form of regularization?
 
 ### Answer
 
 **Mechanism:**
 - Monitor the validation loss during training.
 - Stop training when validation loss stops improving (or starts getting worse) for a set number of epochs ("patience").
 - Restore the weights from the epoch with the best validation loss.
 
 **Regularization Effect:**
 - As training proceeds, weights grow to fit the training data better (and potentially the noise).
 - Stopping early restricts the optimization process to a region of parameter space closer to the initialization (0), effectively limiting the complexity of the final model similar to L2 regularization.
 
 ---
 
 ## Question 5: Batch Normalization
 
 **Topic:** Optimization/Regularization
 **Difficulty:** Advanced
 
 ### Question
 How does Batch Normalization work? Does it act as a regularizer?
 
 ### Answer
 
 **Mechanism:**
 1. **Normalize:** For each mini-batch, calculate mean $\mu$ and variance $\sigma^2$ per feature. Normalize inputs to have 0 mean, 1 variance.
    $$ \hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} $$
 2. **Scale & Shift:** Learn parameters $\gamma$ (scale) and $\beta$ (shift) to allow the network to undo the normalization if needed.
    $$ y = \gamma \hat{x} + \beta $$
 
 **Regularization Effect:**
 - **Yes, typically.**
 - Because means/variances are calculated on mini-batches, they introduce stochastic noise (like Dropout).
 - This noise prevents the network from relying too heavily on any specific neuron activation.
 - Often allows removing Dropout from the network.
 
 ---
 
 ## Question 6: Data Augmentation
 
 **Topic:** Data
 **Difficulty:** Basic
 
 ### Question
 How does Data Augmentation help prevent overfitting, especially in computer vision?
 
 ### Answer
 
 **Concept:**
 - Artificially increase the size of the training dataset by creating modified versions of existing data.
 - **Techniques (Images):**
    - Rotation, Flipping (Horizontal/Vertical)
    - Zooming, Cropping
    - Color Jittering (brightness, contrast)
    - Noise injection
 
 **Why it helps:**
 - The network sees more "unique" examples.
 - It Forces the model to learn **invariant features** (e.g., a cat is still a cat if rotated or darker) rather than memorizing specific pixels.
 - Directly attacks the root cause of overfitting (lack of data).
 
 ---
 
 ## Question 7: Internal Covariate Shift
 
 **Topic:** Theory
 **Difficulty:** Advanced
 
 ### Question
 What is "Internal Covariate Shift", and how does Batch Norm address it?
 
 ### Answer
 
 **Definition:**
 - **Covariate Shift:** The distribution of input data changes.
 - **Internal:** In a deep network, the distribution of inputs to layer $L$ depends on the parameters of all layers before it ($1...L-1$).
 - As weights change during training, the distribution of activations (inputs for the next layer) keeps shifting.
 - The next layer has to constantly "chase" a moving target, slowing down training.
 
 **Batch Norm Solution:**
 - Forces the inputs to every layer to have a stable distribution (mean 0, var 1) throughout training.
 - Decouples layers (somewhat), allowing higher learning rates and faster convergence.
 
 ---
 
 ## Question 8: Regularization Implementation Check
 
 **Topic:** Implementation
 **Difficulty:** Intermediate
 
 ### Question
 How would you add L2 Regularization and Dropout to a Keras Dense layer?
 
 ### Answer
 
 ```python
 from tensorflow.keras import layers, regularizers
 
 # Adding L2 and Dropout
 model.add(layers.Dense(
     units=64,
     activation='relu',
     # L2 Regularization on weights (kernel)
     kernel_regularizer=regularizers.l2(0.01)  # Lambda = 0.01
 ))
 
 # Dropout Layer (applied AFTER activation typically)
 # 0.5 means 50% of neurons are dropped
 model.add(layers.Dropout(0.5))
 ```
 
 ---
 
 ## Question 9: High Bias vs High Variance
 
 **Topic:** Diagnostics
 **Difficulty:** Intermediate
 
 ### Question
 You train a model and get the following results:
 - Train Accuracy: 99%
 - Dev (Validation) Accuracy: 75%
 
 What problem does this indicate? What are 3 potential solutions?
 
 ### Answer
 
 **Diagnosis:**
 - **High Train Acc:** Low Bias (fits data well).
 - **Low Dev Acc:** High Variance (generalizes poorly).
 - **Problem:** **Overfitting**.
 
 **Solutions:**
 1. **Add Regularization:** Add L2 (Weight Decay) or Dropout.
 2. **Get More Data:** Augmentation or collect more samples.
 3. **Simplify Architecture:** Reduce number of layers or neurons/layer.
 4. **Early Stopping:** Stop before variance spikes.
 
 ---
 
 ## Question 10: Dropout at Test Time (Monte Carlo Dropout)
 
 **Topic:** Advanced
 **Difficulty:** Advanced
 
 ### Question
 Normally Dropout is turned off during inference. What happens if you keep it ON during inference? What is "Monte Carlo Dropout"?
 
 ### Answer
 
 **Standard Inference:** Dropout OFF. Weights scaled. Deterministic output.
 
 **MC Dropout (Inference with Dropout ON):**
 - Run the input through the network $T$ times with Dropout ON.
 - Because random neurons are dropped each time, you get $T$ slightly different predictions.
 - **Result:** You obtain a **distribution** of predictions.
 - **Use Case:** Estimating **Model Uncertainty**.
    - Mean of predictions = Final prediction.
    - Variance of predictions = Uncertainty measure.
    - Useful for safety-critical apps (e.g., medical diagnosis, self-driving cars) to know when the model is "unsure".
 
 ---
 
 ## Key Takeaways
 
 - **Overfitting** differs high training performance from low validation performance.
 - **L2 (Weight Decay)** penalizes large weights, preventing complexity.
 - **Dropout** forces redundancy and robustness, acting like an ensemble.
 - **Batch Norm** speeds up training and provides mild regularization.
 - **Early Stopping** is the simplest way to prevent overfitting without tuning parameters.
 
 **Next:** [Day 13 - CNNs](../Day-13/README.md)
