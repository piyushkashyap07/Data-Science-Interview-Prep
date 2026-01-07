# Day 10 - Neural Network Architecture
 
 **Topics Covered:** Hidden Layers, Weights & Biases, Architecture Design, Universal Approximation, MLP from Scratch
 
 ---
 
 ## Question 1: Components of a Neural Network
 
 **Topic:** Neural Network
 **Difficulty:** Basic
 
 ### Question
 Breakdown the anatomical components of a fully connected neural network (MLP). What role does each component play?
 
 ### Answer
 
 1. **Input Layer:**
    - Receives raw data (features).
    - No computation happens here.
    - Number of neurons = Number of features ($n_x$).
 
 2. **Hidden Layers:**
    - Layers between input and output.
    - Perform feature extraction and transformation.
    - "Deep" learning implies >1 hidden layer.
    - Neurons here use activation functions (ReLU, Tanh) to introduce non-linearity.
 
 3. **Weights ($W$):**
    - Learnable parameters that scale the input signal.
    - High weight = Strong influence of that input on the output.
 
 4. **Biases ($b$):**
    - Learnable constant added to the weighted sum.
    - Allows shifting the activation curve (like the intercept $c$ in $y=mx+c$).
 
 5. **Output Layer:**
    - Produces the final prediction.
    - Activation depends on task (Sigmoid for binary, Softmax for multi-class, Linear for regression).
 
 ---
 
 ## Question 2: Designing Neural Network Architecture
 
 **Topic:** Neural Network
 **Difficulty:** Intermediate
 
 ### Question
 How do you decide the number of hidden layers and the number of neurons in each layer?
 
 ### Answer
 
 There is no hard rule, but here are common heuristics:
 
 **1. Number of Hidden Layers:**
 - **0 Layers:** Linear separation only.
 - **1 Layer:** Can approximate any continuous function (Universal Approximation Theorem).
 - **2+ Layers:** Better for complex patterns (images, NLP). Deep narrow networks often generalize better than shallow wide ones.
 - **Start:** Start with 1-2 layers, increase if underfitting.
 
 **2. Number of Neurons per Layer:**
 - **Input Layer:** Fixed by data shape.
 - **Output Layer:** Fixed by task (1 for binary/regression, $K$ for $K$-class classification).
 - **Hidden Neurons:**
    - Common practice: Size often between input and output size.
    - Powers of 2 (32, 64, 128) are common for memory efficiency.
    - **Rule of Thumb:** Start small (e.g., 64), monitor overfitting (if training acc >> validation acc, reduce size or regularize).
 
 ---
 
 ## Question 3: Parameter Counting
 
 **Topic:** Neural Network
 **Difficulty:** Intermediate
 
 ### Question
 Calculate the number of learnable parameters in this network:
 - Input: 10 features
 - Hidden Layer 1: 32 neurons
 - Hidden Layer 2: 16 neurons
 - Output: 1 neuron
 
 ### Answer
 
 **Formula per layer:** $(Inputs \times Neurons) + Biases$
 
 **Calculation:**
 1. **Input to Hidden 1:**
    - Weights: $10 \times 32 = 320$
    - Biases: $32$
    - Total: $352$
 
 2. **Hidden 1 to Hidden 2:**
    - Weights: $32 \times 16 = 512$
    - Biases: $16$
    - Total: $528$
 
 3. **Hidden 2 to Output:**
    - Weights: $16 \times 1 = 16$
    - Biases: $1$
    - Total: $17$
 
 **Grand Total:** $352 + 528 + 17 = 897$ parameters.
 
 ---
 
 ## Question 4: Vectorization
 
 **Topic:** Implementation
 **Difficulty:** Intermediate
 
 ### Question
 Why is vectorization important in Deep Learning? How does it differ from partial loops?
 
 ### Answer
 
 - **Vectorization** allows performing matrix operations on entire approaches of data simultaneously, leveraging **SIMD (Single Instruction, Multiple Data)** instructions in CPUs and parallel architecture of GPUs.
 
 **Example:** Calculating $Z = WX + b$
 - **Loop Approach:** Iterating through each feature and sample is slow ($O(n)$ in Python).
 - **Vectorized Approach:** `np.dot(W, X)` delegates the computation to highly optimized, low-level BLAS/LAPACK libraries.
 
 **Performance Difference:** Vectorized code is typically 100-1000x faster than Python loops.
 
 ---
 
 ## Question 5: Weight Initialization
 
 **Topic:** Optimization
 **Difficulty:** Intermediate
 
 ### Question
 Why shouldn't we initialize all weights to zero? What are better alternatives?
 
 ### Answer
 
 **Why not Zero?**
 - If $W = 0$, every neuron in a hidden layer performs the exact same calculation ($Z=0$, $A=0$ or $0.5$).
 - During backpropagation, all neurons compute the same gradient.
 - They update identically and remain identical (symmetry constraint). The network behaves like a single neuron.
 
 **Better Alternatives:**
 1. **Random Initialization:** Small random values (Gaussian/Uniform).
 2. **Xavier/Glorot Initialization (for Sigmoid/Tanh):** Variance scaled by $\frac{1}{n_{in} + n_{out}}$. Keeps signal variance consistent.
 3. **He Initialization (for ReLU):** Variance scaled by $\frac{2}{n_{in}}$.
 
 ---
 
 ## Question 6: The "Deep" in Deep Learning
 
 **Topic:** Concepts
 **Difficulty:** Basic
 
 ### Question
 What is the benefit of adding depth (more layers) versus adding width (more neurons)?
 
 ### Answer
 
 - **Hierarchical Feature Learning:**
    - **Layer 1:** Detects simple edges/lines.
    - **Layer 2:** Shapes (circles, squares).
    - **Layer 3:** Complex objects (faces, cars).
 - **Efficiency:** Deep networks can represent complex functions with exponentially fewer parameters than a very wide shallow network.
 - **Generalization:** Deeper networks force the model to learn compositionality, which often generalizes better to unseen data.
 
 ---
 
 ## Question 7: Architecture Types
 
 **Topic:** Neural Network
 **Difficulty:** Basics
 
 ### Question
 Briefly describe when to use MLP, CNN, and RNN architectures.
 
 ### Answer
 
 1. **Multilayer Perceptron (MLP):**
    - **Data:** Tabular/Structured data.
    - **Use Case:** Classification, Regression on CSV-like data.
    - **Limitation:** Ignores spatial/temporal structure (flattens inputs).
 
 2. **Convolutional Neural Network (CNN):**
    - **Data:** Image, Spatial data.
    - **Use Case:** Computer Vision, Image Classification.
    - **Key Feature:** Local connectivity, parameter sharing (filters), translation invariance.
 
 3. **Recurrent Neural Network (RNN/LSTM/GRU):**
    - **Data:** Sequential/Time-series data.
    - **Use Case:** NLP, Time series forecasting, Audio.
    - **Key Feature:** Has "memory" (hidden state) to process sequences.
 
 ---
 
 ## Question 8: Input Normalization
 
 **Topic:** Preprocessing
 **Difficulty:** Intermediate
 
 ### Question
 Why is it crucial to normalize inputs (e.g., standard scaler) before feeding them into a Neural Network?
 
 ### Answer
 
 1. **Convergence Speed:**
    - If Feature 1 ranges (0, 1) and Feature 2 ranges (0, 1000), weights for Feature 2 will need to be very small.
    - The loss contours become elongated ellipses. Gradient descent oscillates and takes longer to reach the minimum.
    - Normalized data creates spherical contours, allowing faster, direct convergence.
 
 2. **Vanishing Gradients:**
    - Large inputs can saturate activation functions (Sigmoid/Tanh) immediately, stopping learning.
 
 ---
 
 ## Question 9: Neural Network in Keras/TensorFlow
 
 **Topic:** Implementation
 **Difficulty:** Basic
 
 ### Question
 Write code to define a simple MLP for binary classification using Keras Sequential API.
 
 ### Answer
 
 ```python
 import tensorflow as tf
 from tensorflow.keras import models, layers
 
 # 1. Define Model
 model = models.Sequential()
 
 # 2. Add Layers
 # Input layer (indirectly defined) + First Hidden Layer
 # 16 Neurons, ReLU activation, Input shape depends on features (e.g., 8 features)
 model.add(layers.Dense(16, activation='relu', input_shape=(8,)))
 
 # Second Hidden Layer
 model.add(layers.Dense(8, activation='relu'))
 
 # Output Layer
 # 1 Neuron for binary classification, Sigmoid activation
 model.add(layers.Dense(1, activation='sigmoid'))
 
 # 3. Compile Model
 model.compile(optimizer='adam',
               loss='binary_crossentropy',
               metrics=['accuracy'])
 
 # 4. Summary
 model.summary()
 ```
 
 ---
 
 ## Question 10: MLP from Scratch (Concept)
 
 **Topic:** Implementation
 **Difficulty:** Advanced
 
 ### Question
 If you were building a generic `Dense` layer class from scratch, what variables would you need to initialize in the `__init__` method and what calculation occurs in `forward`?
 
 ### Answer
 
 ```python
 import numpy as np
 
 class DenseLayer:
     def __init__(self, n_inputs, n_neurons):
         # Initialize weights (random)
         # Shape: (n_inputs, n_neurons) so input dot weights works
         self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
         
         # Initialize biases (zeros)
         # Shape: (1, n_neurons)
         self.biases = np.zeros((1, n_neurons))
         
     def forward(self, inputs):
         # inputs shape: (batch_size, n_inputs)
         self.inputs = inputs
         
         # Linear transformation: Z = XW + b
         # Output shape: (batch_size, n_neurons)
         self.output = np.dot(inputs, self.weights) + self.biases
         return self.output
 ```
 Note: This is just the linear step. An activation object would normally follow this layer.
 
 ---
 
 ## Key Takeaways
 
 - **Architecture:** Input -> Hidden (Feature Extraction) -> Output (Prediction).
 - **Heuristics:** Start simple (1-2 layers), scale up. Powers of 2 for neuron counts.
 - **Weights & Biases:** The actual "knowledge" learned by the network.
 - **Normalization:** Critical for fast and stable training.
 - **Vectorization:** Essential for computational efficiency.
 
 **Next:** [Day 11 - Backpropagation](../Day-11/README.md)
