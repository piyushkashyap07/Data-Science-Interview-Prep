# Day 09 - Deep Learning Basics
 
 **Topics Covered:** Deep Learning vs Machine Learning, Perceptrons, Neural Network Basics, Activation Functions, Forward Propagation
 
 ---
 
 ## Question 1: Deep Learning vs Machine Learning
 
 **Topic:** Deep Learning
 **Difficulty:** Intermediate
 
 ### Question
 What are the fundamental differences between Deep Learning and traditional Machine Learning? When should you prioritize one over the other?
 
 ### Answer
 
 **Deep Learning** is a subset of Machine Learning that uses multi-layered neural networks to learn representations from data.
 
 | Feature | Machine Learning | Deep Learning |
 |---------|------------------|---------------|
 | **Data Dependency** | PERFORMS well with small to medium data | REQUIRES large amounts of data to outperform ML |
 | **Hardware** | Can run on potential CPUs | Typically requires GPUs/TPUs for training |
 | **Feature Engineering** | Manual feature extraction is critical | Automated feature extraction (learns features itself) |
 | **Training Time** | Short (minutes to hours) | Long (hours to weeks) |
 | **Interpretability** | High (e.g., Decision Trees, Linear Regression) | Low (Black-box models) |
 
 **When to use Deep Learning:**
 - Unstructured data (Images, Audio, Text)
 - Complex patterns where feature engineering is difficult
 - Large scale data availability
 
 **When to use Machine Learning:**
 - Structured/Tabular data
 - Limited data availability
 - Interpretability is required
 - Low latency/resource constraints
 
 ---
 
 ## Question 2: The Perceptron
 
 **Topic:** Deep Learning
 **Difficulty:** Basic
 
 ### Question
 Explain the architecture of a Perceptron and its limitations. Why can't a single perceptron solve the XOR problem?
 
 ### Answer
 
 A **Perceptron** is the fundamental building block of a neural network (a single artificial neuron).
 
 **Architecture:**
 1. **Inputs ($x_1, x_2, ..., x_n$):** Features
 2. **Weights ($w_1, w_2, ..., w_n$):** Importance of each feature
 3. **Bias ($b$):** Shift parameters to adjust the activation threshold
 4. **Weighted Sum ($z$):** $z = \sum w_i x_i + b$
 5. **Activation Function:** Step function (originally) to map $z$ to output $y$
 
 $$ y = \begin{cases} 1 & \text{if } w \cdot x + b > 0 \\ 0 & \text{otherwise} \end{cases} $$
 
 **XOR Problem Limitation:**
 - A single perceptron is a **linear classifier** (it draws a straight line to separate classes).
 - The XOR function (Exclusive OR) is **not linearly separable**.
 - You cannot draw a single straight line to separate (0,0) & (1,1) from (0,1) & (1,0).
 - **Solution:** Multi-Layer Perceptrons (MLPs) introduce hidden layers and non-linearities to solve this.
 
 ---
 
 ## Question 3: Activation Functions - Sigmoid & Tanh
 
 **Topic:** Deep Learning
 **Difficulty:** Intermediate
 
 ### Question
 Compare Sigmoid and Tanh activation functions. What is the "Vanishing Gradient" problem associated with them?
 
 ### Answer
 
 **1. Sigmoid Function:**
 - **Formula:** $\sigma(z) = \frac{1}{1 + e^{-z}}$
 - **Range:** $(0, 1)$
 - **Usage:** Binary classification output layer
 - **Pros:** Smooth gradient, clear probabilistic interpretation
 
 **2. Hyperbolic Tangent (Tanh):**
 - **Formula:** $\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$
 - **Range:** $(-1, 1)$
 - **Usage:** Hidden layers (historically)
 - **Pros:** Zero-centered (unlike Sigmoid), often converges faster
 
 **The Vanishing Gradient Problem:**
 - Both functions saturate (flatten out) at extreme values ($z \to \pm \infty$).
 - The derivatives become very small (close to 0).
 - During backpropagation, these small gradients are multiplied essentially killing the gradient for early layers.
 - **Consequence:** Deep networks fail to learn effectively.
 
 ---
 
 ## Question 4: ReLU (Rectified Linear Unit)
 
 **Topic:** Deep Learning
 **Difficulty:** Intermediate
 
 ### Question
 Why is ReLU the most popular activation function for hidden layers? What is the "Dying ReLU" problem?
 
 ### Answer
 
 **ReLU Formula:** $f(z) = \max(0, z)$
 
 **Why it's popular:**
 1. **Computational Efficiency:** Simple thresholding (no expensive exp() calculations).
 2. **Sparsity:** Outputs exact 0 for negative inputs, leading to sparse activations.
 3. **Solves Vanishing Gradient (partially):** For positive inputs, the derivative is constant (1), preventing gradients from vanishing.
 
 **Dying ReLU Problem:**
 - If a neuron learns weights such that it always outputs a negative value for all typically inputs, it essentially "dies".
 - The gradient becomes 0, and weights never update again.
 - **Solution:** Leaky ReLU or Parametric ReLU (PReLU), which allows a small gradient ($0.01z$) for negative inputs.
 
 ```python
 import numpy as np
 
 def relu(z):
     return np.maximum(0, z)
 
 def leaky_relu(z, alpha=0.01):
     return np.where(z > 0, z, z * alpha)
 ```
 
 ---
 
 ## Question 5: Forward Propagation
 
 **Topic:** Deep Learning
 **Difficulty:** Intermediate
 
 ### Question
 Describe the mathematical steps of Forward Propagation in a basic Neural Network with one hidden layer.
 
 ### Answer
 
 Let:
 - $X$: Input matrix
 - $W^{[1]}, b^{[1]}$: Weights/Biases for hidden layer
 - $W^{[2]}, b^{[2]}$: Weights/Biases for output layer
 - $g^{[1]}$: Hidden layer activation (e.g., ReLU)
 - $g^{[2]}$: Output layer activation (e.g., Sigmoid)
 
 **Steps:**
 
 1. **Hidden Layer Linear Step:**
    $$ Z^{[1]} = W^{[1]} \cdot X + b^{[1]} $$
 
 2. **Hidden Layer Activation:**
    $$ A^{[1]} = g^{[1]}(Z^{[1]}) $$
 
 3. **Output Layer Linear Step:**
    $$ Z^{[2]} = W^{[2]} \cdot A^{[1]} + b^{[2]} $$
 
 4. **Output Prediction:**
    $$ \hat{Y} = A^{[2]} = g^{[2]}(Z^{[2]}) $$
 
 5. **Loss Calculation:**
    Compare $\hat{Y}$ with true labels $Y$ using a loss function $L(\hat{Y}, Y)$.
 
 ---
 
 ## Question 6: Softmax Activation
 
 **Topic:** Deep Learning
 **Difficulty:** Intermediate
 
 ### Question
 What is the Softmax activation function, and why is it used in the output layer for multi-class classification?
 
 ### Answer
 
 **Softmax** converts a vector of raw scores (logits) into a probability distribution.
 
 **Formula:**
 $$ \sigma(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}} $$
 for $i = 1, \dots, K$ classes.
 
 **Why Multi-class Classification?**
 1. **Probabilities:** It ensures all output values sum to 1 and are between 0 and 1.
 2. **Exclusivity:** Because they sum to 1, an increase in the probability of one class forces a decrease in others (unlike independent Sigmoids).
 3. **Interpretability:** Outputs can be directly interpreted as the model's confidence for each class.
 
 ```python
 import numpy as np
 
 def softmax(logits):
     exp_scores = np.exp(logits)
     return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
 
 logits = np.array([[2.0, 1.0, 0.1]])
 probs = softmax(logits)
 print(probs) # Output sums to 1
 ```
 
 ---
 
 ## Question 7: Cost Functions (Loss Functions)
 
 **Topic:** Deep Learning
 **Difficulty:** Intermediate
 
 ### Question
 Explain the difference between Mean Squared Error (MSE) and Cross-Entropy Loss. When would you use each?
 
 ### Answer
 
 **1. Mean Squared Error (MSE):**
 - **Formula:** $L = \frac{1}{n} \sum (y - \hat{y})^2$
 - **Use Case:** **Regression** problems (predicting continuous values like house prices).
 - **Why not for classification?** When used with Sigmoid/Softmax, it results in a non-convex error surface, making optimization difficult (many local minima).
 
 **2. Cross-Entropy Loss (Log Loss):**
 - **Formula (Binary):** $L = - [y \log(\hat{y}) + (1-y) \log(1-\hat{y})]$
 - **Use Case:** **Classification** problems.
 - **Why?** Heavily penalizes confident wrong predictions. It works well with Softmax/Sigmoid to create a convex optimization landscape.
 
 ---
 
 ## Question 8: Universal Approximation Theorem
 
 **Topic:** Deep Learning
 **Difficulty:** Advanced
 
 ### Question
 What is the Universal Approximation Theorem? What does it imply about the power of Neural Networks?
 
 ### Answer
 
 **Theorem Statement:**
 A feedforward neural network with a **single hidden layer** containing a finite number of neurons can approximate almost **any continuous function** to an arbitrary degree of precision, provided the activation function is non-linear.
 
 **Implications:**
 - Neural Networks are **universal function approximators**.
 - In theory, a simple 1-hidden-layer network can solve any problem.
 - **Caveat:** It doesn't tell us *how* to find the correct weights or how *many* neurons are needed. Often, the number of neurons required is exponentially large, which is why **deep** networks (multiple layers) are preferred—they represent complex functions more efficiently.
 
 ---
 
 ## Question 9: Neural Network Hyperparameters
 
 **Topic:** Deep Learning
 **Difficulty:** Basic
 
 ### Question
 List the key hyperparameters in a Neural Network. Which ones are learned by the model and which are set by the user?
 
 ### Answer
 
 **Parameters (Learned by the Model):**
 - **Weights ($W$):** Connection strengths between neurons.
 - **Biases ($b$):** Activation thresholds.
 *The user does NOT set these; the optimizer finds them.*
 
 **Hyperparameters (Set by the User):**
 1. **Network Architecture:** Number of hidden layers, Number of neurons per layer.
 2. **Learning Rate ($\alpha$):** Step size for gradient descent.
 3. **Activation Functions:** ReLU, Sigmoid, etc.
 4. **Number of Epochs:** How many times to pass the entire dataset.
 5. **Batch Size:** Number of samples processed before updating weights.
 6. **Optimizer Choice:** SGD, Adam, RMSprop.
 7. **Regularization:** Dropout rate, L2 penalty.
 
 ---
 
 ## Question 10: Simple Neural Network Implementation
 
 **Topic:** Deep Learning
 **Difficulty:** Intermediate
 
 ### Question
 Write a simple Python class to implement a single neuron (Perceptron) that can learn the OR gate using the step activation function.
 
 ### Answer
 
 ```python
 import numpy as np
 
 class Perceptron:
     def __init__(self, input_size, lr=0.1, epochs=100):
         self.weights = np.zeros(input_size + 1) # +1 for bias
         self.lr = lr
         self.epochs = epochs
         
     def activation(self, z):
         return 1 if z >= 0 else 0
         
     def predict(self, x):
         z = np.dot(x, self.weights[1:]) + self.weights[0]
         return self.activation(z)
         
     def fit(self, X, y):
         for _ in range(self.epochs):
             for i in range(len(X)):
                 prediction = self.predict(X[i])
                 error = y[i] - prediction
                 # Weight update rule: w = w + lr * error * input
                 self.weights[1:] += self.lr * error * X[i]
                 self.weights[0] += self.lr * error # Update bias
 
 # OR Gate Data
 X = np.array([[0,0], [0,1], [1,0], [1,1]])
 y = np.array([0, 1, 1, 1])
 
 model = Perceptron(input_size=2)
 model.fit(X, y)
 
 # Test
 for x_in in X:
     print(f"Input: {x_in}, Pred: {model.predict(x_in)}")
 # Expected Output: 0, 1, 1, 1
 ```
 
 ---
 
 ## Key Takeaways
 
 - Deep Learning excels at unstructured data but requires more resources.
 - Activation functions introduce non-linearity, essential for learning complex patterns.
 - **ReLU** is the default for hidden layers; **Sigmoid/Softmax** for outputs.
 - **Forward Propagation** calculates predictions; **Backpropagation** (Day 11) calculates errors to update weights.
 
 **Next:** [Day 10 - Neural Network Architecture](../Day-10/README.md)
