# Day 32 - Matrix Factorization
 
 **Topics Covered:** Latent Factors, SVD (Singular Value Decomposition), SVD++, ALS (Alternating Least Squares), Netflix Prize, Hybrid Systems.
 
 ---
 
 ## Question 1: Latent Factors
 
 **Topic:** Concept
 **Difficulty:** Basic
 
 ### Question
 Matrix Factorization decomposes Users and Items into "Latent Factors". What does this mean intuitively?
 
 ### Answer
 
 - **Latent Factors:** Hidden/Unobserved features that explain why a user likes a movie.
 - **Example:**
    - Factor 1: High value = Comedy, Low value = Horror.
    - Factor 2: High value = For Kids, Low value = For Adults.
 - A User Vector `[0.9, -0.8]` means they love Comedy and Adult movies.
 - An Item Vector `[0.9, -0.9]` means it is "The Hangover" (Comedy + Adult).
 - **Dot Product:** `0.81 + 0.72 = 1.53` (High Match).
 
 ---
 
 ## Question 2: SVD (Singular Value Decomposition)
 
 **Topic:** Math
 **Difficulty:** Intermediate
 
 ### Question
 How does SVD decompose the Rating Matrix $R$?
 
 ### Answer
 
 $$ R \approx U \times V^T $$
 
 - $U$: User Matrix (Users $\times$ Factors).
 - $V$: Item Matrix (Items $\times$ Factors).
 - **Prediction:** $\hat{r}_{ui} = u_i \cdot v_j$
 - **Training:** Since $R$ is sparse (missing values), we cannot use standard Linear Algebra SVD. We use **Gradient Descent** to minimize the squared error only on *observed* ratings.
 
 ---
 
 ## Question 3: Bias Terms
 
 **Topic:** Algorithm
 **Difficulty:** Intermediate
 
 ### Question
 The simple dot product $u \cdot v$ is not enough. Why do we add Bias terms?
 
 ### Answer
 
 $$ \hat{r}_{ui} = \mu + b_u + b_i + u_i \cdot v_j $$
 
 - $\mu$: Global Average rating (e.g., 3.5 stars).
 - $b_u$: **User Bias**. User A is grumpy and rates everything 1 star lower than average.
 - $b_i$: **Item Bias**. "The Shawshank Redemption" is universally loved, so it gets +1 star regardless of user.
 - Captures effects independent of interaction.
 
 ---
 
 ## Question 4: SVD++ (Netflix Prize Winner)
 
 **Topic:** Algorithm
 **Difficulty:** Advanced
 
 ### Question
 What did SVD++ add to standard SVD to improve accuracy?
 
 ### Answer
 
 **Implicit Feedback Integration.**
 - Even if a user hasn't rated a movie, the *fact that they rated it at all* provides information.
 - SVD++ adds a vector $y_j$ for every item $j$ rated by the user.
 - **User Representation:** $P_u + \frac{1}{\sqrt{|N(u)|}} \sum_{j \in N(u)} y_j$
 - It models the user not just by their ID, but by the set of items they interacted with.
 
 ---
 
 ## Question 5: ALS (Alternating Least Squares)
 
 **Topic:** Algorithm
 **Difficulty:** Advanced
 
 ### Question
 Why is ALS preferred over Gradient Descent for parallelization (Spark)?
 
 ### Answer
 
 **Mechanism:**
 1. Fix User Matrix $U$, solve for Item Matrix $V$ (Becomes a Convex Linear Regression problem).
 2. Fix $V$, solve for $U$.
 3. Repeat.
 
 **Pros:**
 - Each step can be perfectly **Parallelized** (Solving for User 1 is independent of User 2).
 - Works great on Spark clusters for massive datasets.
 
 ---
 
 ## Question 6: Neural Collaborative Filtering (NCF)
 
 **Topic:** Deep Learning
 **Difficulty:** Intermediate
 
 ### Question
 How does Deep Learning replace Matrix Factorization?
 
 ### Answer
 
 - MF assumes a linear relationship (Dot Product).
 - **NCF:** Replaces the Dot Product with a **Neural Network**.
 - Input: `[User Embedding, Item Embedding]`.
 - Layers: Concatenate -> Dense -> ReLU -> Dense -> Output.
 - **Benefit:** Can learn complex, non-linear interactions between users and items.
 
 ---
 
 ## Question 7: Hybrid Systems
 
 **Topic:** System Design
 **Difficulty:** Basic
 
 ### Question
 How do you combine Content-Based and Collaborative Filtering (Hybrid)?
 
 ### Answer
 
 **Methods:**
 1. **Weighted:** Average the scores ($0.5 \times \text{Content} + 0.5 \times \text{CF}$).
 2. **Switching:** Use Content-Based for Cold Start users, switch to CF once they have >10 ratings.
 3. **Feature Augmentation:** Feed Content features (Genre, Director) directly into the Matrix Factorization model (e.g., LightFM).
 
 ---
 
 ## Question 8: Factorization Machines
 
 **Topic:** Algorithm
 **Difficulty:** Advanced
 
 ### Question
 What are Factorization Machines designed to handle that MF cannot?
 
 ### Answer
 
 **Side Features (Context).**
 - MF only knows User ID and Item ID.
 - FM can take a sparse vector including: `{User=A, Item=B, Time=Night, Device=Mobile}`.
 - It models interactions between **all variables** (e.g., User A behaves differently at Night vs Day).
 - Generalizes MF to any number of features.
 
 ---
 
 ## Question 9: Implementing SVD (Surprise Library)
 
 **Topic:** Implementation
 **Difficulty:** Basic
 
 ### Question
 Write Python code to train SVD using the `surprise` library.
 
 ### Answer
 
 ```python
 from surprise import SVD, Dataset, Reader
 from surprise.model_selection import cross_validate
 
 # 1. Prepare Data
 reader = Reader(rating_scale=(1, 5))
 data = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)
 
 # 2. Algorithm
 algo = SVD()
 
 # 3. Train/Evaluate
 cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
 ```
 
 ---
 
 ## Question 10: Evaluation (Precision@K)
 
 **Topic:** Metric
 **Difficulty:** Intermediate
 
 ### Question
 RMSE is good for rating prediction. What if we only care about the "Top 10" recommendations?
 
 ### Answer
 
 **Precision at K (P@K):**
 - Of the top $K$ items we recommended, how many were relevant (actually liked/clicked)?
 - $$ P@K = \frac{\text{# Relevant in top K}}{K} $$
 - **Recall@K:** How many of the total relevant items appeared in the top K?
 - These metrics are far more important for ranking systems (Search/Feeds).
 
 ---
 
 ## Key Takeaways
 
 - **Latent Factors** uncover hidden semantics.
 - **Bias Terms** correct for user grumpiness or item popularity.
 - **ALS** is the king of parallel MF (Spark).
 - **Hybrid Systems** solve the Cold Start problem.
 - **Precision@K** is superior to RMSE for ranking tasks.
 
 **Next:** [Day 33 - Clustering](../Day-33/README.md)
