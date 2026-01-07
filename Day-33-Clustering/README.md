# Day 33 - Clustering (Unsupervised Learning)
 
 **Topics Covered:** K-Means, Elbow Method, Hierarchical Clustering (Dendrograms), DBSCAN, Gaussian Mixture Models (GMM), Silhouette Score.
 
 ---
 
 ## Question 1: Unsupervised Learning
 
 **Topic:** Concept
 **Difficulty:** Basic
 
 ### Question
 How does Clustering differ from Classification?
 
 ### Answer
 
 - **Classification (Supervised):** You have labels ($y$). "Teach me to separate Cats from Dogs."
 - **Clustering (Unsupervised):** You have No labels. "Here is a pile of data. Group similar things together."
 - **Goal:** Minimize intra-cluster distance (items inside group are similar) and maximize inter-cluster distance (groups are distinct).
 
 ---
 
 ## Question 2: K-Means Algorithm
 
 **Topic:** Algorithm
 **Difficulty:** Basic
 
 ### Question
 Explain the iterative steps of K-Means.
 
 ### Answer
 
 1. **Initialize:** Pick $K$ random centroids.
 2. **Assign:** For every point, calculate distance to all centroids. Assign point to the closest centroid.
 3. **Update:** Calculate the *mean* of all points assigned to a cluster. Move the centroid to this new mean.
 4. **Repeat:** Loop steps 2-3 until centroids stop moving (Convergence).
 
 ---
 
 ## Question 3: Choosing K (Elbow Method)
 
 **Topic:** Heuristic
 **Difficulty:** Intermediate
 
 ### Question
 How do you decide the optimal number of clusters ($K$)?
 
 ### Answer
 
 **Elbow Method:**
 1. Run K-Means for $K=1$ to $10$.
 2. Calculate **Inertia** (Sum of squared distances from points to their centroids).
 3. Plot Inertia vs $K$.
 4. As $K$ increases, Inertia decreases.
 5. Pick the "Elbow" point where the rate of decrease sharpens/flattens. This is the point of diminishing returns.
 
 ---
 
 ## Question 4: K-Means++
 
 **Topic:** Initialization
 **Difficulty:** Intermediate
 
 ### Question
 Standard K-Means initialization is random, which can lead to bad results. How does K-Means++ fix this?
 
 ### Answer
 
 - **Problem:** If two initial centroids are placed very close to each other, they might split a single natural cluster into two.
 - **K-Means++:**
    1. Pick 1st centroid perfectly randomly.
    2. Pick subsequent centroids with probability proportional to the **squared distance** from existing centroids.
    3. This ensures initial centroids are **spread out** across the data space.
 
 ---
 
 ## Question 5: Hierarchical Clustering
 
 **Topic:** Algorithm
 **Difficulty:** Intermediate
 
 ### Question
 What is the difference between Agglomerative and Divisive Hierarchical Clustering? What is a Dendrogram?
 
 ### Answer
 
 - **Agglomerative (Bottom-Up):** Start with $N$ clusters (every point is a cluster). Merge closest pair. Repeat until 1 cluster remains.
 - **Divisive (Top-Down):** Start with 1 big cluster. Split recursively.
 - **Dendrogram:** A tree diagram showing the sequence of merges. The height of the branch represents the distance at which the merge happened. Cutting the tree at a specific height gives you $K$ clusters.
 
 ---
 
 ## Question 6: DBSCAN (Density-Based)
 
 **Topic:** Algorithm
 **Difficulty:** Advanced
 
 ### Question
 Why would you choose DBSCAN over K-Means?
 
 ### Answer
 
 1. **Arbitrary Shapes:** K-Means assumes spherical clusters. DBSCAN can find "moons", "rings", or "snakes" because it follows density.
 2. **No K:** You don't need to specify number of clusters.
 3. **Outliers:** DBSCAN explicitly labels noise points as -1 (Outliers), where K-Means forces every point into a cluster.
 
 ---
 
 ## Question 7: Gaussian Mixture Models (GMM)
 
 **Topic:** Algorithm
 **Difficulty:** Advanced
 
 ### Question
 K-Means is a "Hard Clustering" method. How is GMM a "Soft Clustering" method?
 
 ### Answer
 
 - **Hard (K-Means):** Point A belongs to Cluster 1 with 100% certainty.
 - **Soft (GMM):**
    - Assumes data comes from a mixture of $K$ Gaussian distributions.
    - Yields **Probability**: Point A belongs to Cluster 1 (80%) and Cluster 2 (20%).
    - Can model elliptical clusters (Using covariance matrix) unlike K-Means' spheres.
 
 ---
 
 ## Question 8: Evaluation (Silhouette Score)
 
 **Topic:** Metric
 **Difficulty:** Intermediate
 
 ### Question
 Explain the Silhouette Score. What does a score of +1 vs -1 mean?
 
 ### Answer
 
 $$ S = \frac{b - a}{\max(a, b)} $$
 - $a$: Mean distance to other points in *same* cluster (Cohesion).
 - $b$: Mean distance to points in *nearest neighbor* cluster (Separation).
 
 **Interpretation:**
 - **+1:** Perfectly clustered (Far from neighbor, close to self).
 - **0:** On the border.
 - **-1:** Wrongly assigned (Closer to neighbor than self).
 
 ---
 
 ## Question 9: Curse of Dimensionality in Clustering
 
 **Topic:** Concept
 **Difficulty:** Advanced
 
 ### Question
 Why does Euclidean Distance (and thus K-Means) fail in high dimensions (e.g., 1000 features)?
 
 ### Answer
 
 - In high-dimensional space, **all points become equidistant** from each other.
 - The contrast between "closest" and "farthest" neighbor vanishes.
 - The concept of "density" (for DBSCAN) or "distance" (for K-Means) becomes meaningless.
 - **Fix:** Use Dimensionality Reduction (PCA/t-SNE) *before* clustering.
 
 ---
 
 ## Question 10: Implementation (K-Means)
 
 **Topic:** Implementation
 **Difficulty:** Basic
 
 ### Question
 Write code to cluster a dataset into 3 groups.
 
 ### Answer
 
 ```python
 from sklearn.cluster import KMeans
 from sklearn.datasets import make_blobs
 import matplotlib.pyplot as plt
 
 # 1. Data
 X, y = make_blobs(n_samples=300, centers=4, random_state=42)
 
 # 2. Model
 kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42)
 y_kmeans = kmeans.fit_predict(X)
 
 # 3. Visualize
 plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis')
 plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
 plt.show()
 ```
 
 ---
 
 ## Key Takeaways
 
 - **K-Means** is fast and simple but assumes spherical blobs.
 - **DBSCAN** is great for weird shapes and outlier detection.
 - **Hierarchical** gives you a taxonomy (Dendrogram).
 - **Silhouette Score** validates the specific cluster count without labels.
 - **Standardize** data before clustering (Distance metrics are sensitive to scale!).
 
 **Next:** [Day 34 - Dimensionality Reduction](../Day-34/README.md)
