# Day 34 - Dimensionality Reduction
 
 **Topics Covered:** Curse of Dimensionality, PCA (Principal Component Analysis), Explained Variance, t-SNE, UMAP, Linear vs Non-Linear Methods.
 
 ---
 
 ## Question 1: The Curse of Dimensionality
 
 **Topic:** Concept
 **Difficulty:** Basic
 
 ### Question
 Explain why having 1,000 features is often worse than having 10, even if the 1,000 features contain more info.
 
 ### Answer
 
 **Concept:** As dimensions increase, the volume of the space increases exponentially.
 1. **Sparsity:** Data becomes incredibly sparse. To maintain the same density of samples in 1000D space as 10D, you'd need exponentially more data.
 2. **Distance Failure:** Distance metrics (Euclidean) lose meaning (everything is equally far away).
 3. **Overfitting:** Models find coincidental patterns in the noise of the extra dimensions.
 
 ---
 
 ## Question 2: Principal Component Analysis (PCA)
 
 **Topic:** Algorithm
 **Difficulty:** Intermediate
 
 ### Question
 PCA aims to find "Principal Components". What physical property of the data do these components maximize?
 
 ### Answer
 
 **Variance.**
 - PCA finds the direction (axis) along which the data varies the most.
 - **PC1:** The line that captures the maximum spread of the data.
 - **PC2:** The line orthogonal (90 degrees) to PC1 that captures the second most variance.
 - By projecting data onto just PC1 and PC2, we preserve the "structure" while dropping noise.
 
 ---
 
 ## Question 3: Eigenvalues and Eigenvectors
 
 **Topic:** Math
 **Difficulty:** Advanced
 
 ### Question
 How does the Covariance Matrix relate to PCA?
 
 ### Answer
 
 1. Calculate the **Covariance Matrix** of the features.
 2. Compute its **Eigenvectors** and **Eigenvalues**.
    - **Eigenvector:** The Direction of the new axis (PC1, PC2...).
    - **Eigenvalue:** The Magnitude of variance explained by that axis.
 3. Sort pairs by Eigenvalue (Highest to Lowest). Keep the top $k$.
 
 ---
 
 ## Question 4: Explained Variance Ratio
 
 **Topic:** Evaluation
 **Difficulty:** Basic
 
 ### Question
 You run PCA and get `[0.70, 0.20, 0.05, ...]`. What does this mean?
 
 ### Answer
 
 - **Component 1** explains 70% of the information (variance) in the original dataset.
 - **Component 2** explains 20%.
 - **Together:** Just 2 dimensions capture 90% of the total patterns. You can safely drop the other 98 dimensions (which are likely noise) and still model the data accurately.
 
 ---
 
 ## Question 5: Linear vs Non-Linear Reduction
 
 **Topic:** Concept
 **Difficulty:** Intermediate
 
 ### Question
 PCA is a Linear method. When does it fail? What should you use instead?
 
 ### Answer
 
 - **Linear (PCA):** Can only flatten data like a pancake.
 - **Failure:** If data lies on a curved manifold (e.g., a "Swiss Roll" shape). PCA smashing a Swiss Roll flat destroys the spiral structure.
 - **Non-Linear (t-SNE/UMAP):** Can "unroll" the Swiss Roll. They preserve the **local neighborhood structure** rather than global variance.
 
 ---
 
 ## Question 6: t-SNE (t-Distributed Stochastic Neighbor Embedding)
 
 **Topic:** Algorithm
 **Difficulty:** Advanced
 
 ### Question
 Why is t-SNE widely used for visualization but not for clustering/preprocessing?
 
 ### Answer
 
 - **Pros:** Creates beautiful 2D clusters from high-dimensional data (e.g., MNIST digits separate perfectly).
 - **Cons:**
    1. **No learned function:** You cannot transform *new* data points (N/A for production).
    2. **Distorted Global Geometry:** It preserves local neighbors well, but the distance between far-away clusters is meaningless.
    3. **Slow:** $O(N^2)$.
 
 ---
 
 ## Question 7: UMAP (Uniform Manifold Approximation)
 
 **Topic:** Algorithm
 **Difficulty:** Advanced
 
 ### Question
 How is UMAP an improvement over t-SNE?
 
 ### Answer
 
 - **Speed:** Much faster (can handle millions of points).
 - **Global Structure:** Better at preserving the global relationship between clusters (unlike t-SNE).
 - **Transform:** Can learn a transformation to apply to specific new data.
 - **Conclusion:** It is effectively the modern replacement for t-SNE.
 
 ---
 
 ## Question 8: Autoencoders for Reduction
 
 **Topic:** Deep Learning
 **Difficulty:** Intermediate
 
 ### Question
 How can a Neural Network perform dimensionality reduction?
 
 ### Answer
 
 **Bottleneck Layer:**
 - Input (100) -> Dense (50) -> **Dense (2)** -> Dense (50) -> Output (100).
 - The middle layer (2 neurons) forces the network to compress the 100 inputs into 2 coordinates (Latent Space) and then reconstruct them.
 - This is essentially **Non-Linear PCA**.
 
 ---
 
 ## Question 9: Pre-processing for PCA
 
 **Topic:** Implementation
 **Difficulty:** Basic
 
 ### Question
 What happens if you run PCA without scaling your data first?
 
 ### Answer
 
 - **Disaster.**
 - Feature A: "Salary" (Range 0 - 100,000). Variance is huge.
 - Feature B: "Age" (Range 0 - 100). Variance is tiny.
 - PCA looks for Maximum Variance. It will think "Salary" is the **only** important feature (PC1 will just completely align with Salary).
 - **Fix:** Always `StandardScaler` (Mean=0, Var=1) before PCA.
 
 ---
 
 ## Question 10: Implementation (PCA)
 
 **Topic:** Implementation
 **Difficulty:** Basic
 
 ### Question
 Write code to reduce the Iris dataset to 2 dimensions.
 
 ### Answer
 
 ```python
 from sklearn.decomposition import PCA
 from sklearn.preprocessing import StandardScaler
 from sklearn.datasets import load_iris
 import matplotlib.pyplot as plt
 
 # 1. Load & Scale
 iris = load_iris()
 X = iris.data
 X_scaled = StandardScaler().fit_transform(X)
 
 # 2. PCA
 pca = PCA(n_components=2)
 X_pca = pca.fit_transform(X_scaled)
 
 # 3. Check Variance
 print(pca.explained_variance_ratio_) 
 # [0.72, 0.23] -> 95% total explained
 
 # 4. Plot
 plt.scatter(X_pca[:, 0], X_pca[:, 1], c=iris.target)
 plt.xlabel('PC1')
 plt.ylabel('PC2')
 plt.show()
 ```
 
 ---
 
 ## Key Takeaways
 
 - **PCA** removes correlation and compresses data by maximizing variance.
 - **Always Scale** data before reduction.
 - **t-SNE/UMAP** are great for 2D visualization of complex data.
 - **Curse of Dimensionality** is the reason we need these tools.
 
 **Next:** [Day 35 - MLOps](../Day-35/README.md)
