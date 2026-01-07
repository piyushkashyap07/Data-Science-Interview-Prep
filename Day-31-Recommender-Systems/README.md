# Day 31 - Recommender Systems Basics
 
 **Topics Covered:** Types of Recommendation, Content-Based Filtering, Collaborative Filtering, User-User vs Item-Item, Cold Start Problem, Hybrid Systems.
 
 ---
 
 ## Question 1: Types of Recommenders
 
 **Topic:** Concept
 **Difficulty:** Basic
 
 ### Question
 What are the three main approaches to building a Recommender System?
 
 ### Answer
 
 1. **Content-Based Filtering:**
    - Recommend items *similar* to what the user liked before.
    - "You liked 'Batman', so here is 'Superman' (both are Action/Comics)."
 2. **Collaborative Filtering:**
    - Recommend items liked by *similar users*.
    - "Users who liked 'Batman' also liked 'Inception', so try 'Inception'."
 3. **Hybrid:**
    - Combination of both. (e.g., Netflix Uses everything).
 
 ---
 
 ## Question 2: Content-Based Filtering Logic
 
 **Topic:** Algorithm
 **Difficulty:** Intermediate
 
 ### Question
 How does Content-Based filtering work mathematically?
 
 ### Answer
 
 1. **Item Profile:** Create a feature vector for each movie (Genre: Action, Director: Nolan, Year: 2010).
 2. **User Profile:** Create a vector for the user based on their history (Average of vectors of movies they liked).
 3. **Process:** Calculate **Cosine Similarity** between User Vector and all Item Vectors.
 4. **Recommend:** Top K most similar items.
 
 ---
 
 ## Question 3: Collaborative Filtering (User-User)
 
 **Topic:** Algorithm
 **Difficulty:** Basic
 
 ### Question
 Explain User-based Collaborative Filtering.
 
 ### Answer
 
 "People who are similar to me also liked..."
 
 1. Find users who have a similar rating history to Alice. (Similarity Measure).
 2. If similar users rated Movie X highly, and Alice hasn't seen it yet...
 3. Recommend Movie X to Alice.
 
 ---
 
 ## Question 4: Item-Item Collaborative Filtering
 
 **Topic:** Algorithm
 **Difficulty:** Advanced
 
 ### Question
 Why did Amazon popularize Item-Item CF over User-User CF?
 
 ### Answer
 
 "Users who bought X also bought Y."
 
 **Reason:**
 - **Stability:** Item ratings don't change much. User tastes change often.
 - **Scale:** There are usually way more Users (Millions) than Items (Products). Calculating a $User \times User$ matrix is computationally expensive ($N^2$). $Item \times Item$ is smaller and pre-computable.
 
 ---
 
 ## Question 5: Explicit vs Implicit Feedback
 
 **Topic:** Data
 **Difficulty:** Basic
 
 ### Question
 Difference between Explicit and Implicit feedback? Which is more common?
 
 ### Answer
 
 1. **Explicit:** Direct rating.
    - User gives 5 stars. Thumbs up/down.
    - High quality, but **Very Rare** (Most users don't rate).
 2. **Implicit:** Indirect behavior.
    - User clicked, watched 50% of video, added to cart.
    - Lower quality signal (Click $\neq$ Like), but **Abundant Volume**.
 
 *Implicit is the standard for modern systems.*
 
 ---
 
 ## Question 6: The Cold Start Problem
 
 **Topic:** Challenge
 **Difficulty:** Intermediate
 
 ### Question
 What is the Cold Start Problem? How to solve it?
 
 ### Answer
 
 **Problem:**
 - **New User:** We know nothing about them. Cannot calculate similarity.
 - **New Item:** No one has rated it yet. Collaborative filtering ignores it.
 
 **Solutions:**
 - **New User:** Ask for preferences on sign-up ("Choose 3 genres"). Recommend popular items (non-personalized).
 - **New Item:** Use **Content-Based** features (It's an Action movie, let's show it to Action fans) until it gets interaction data.
 
 ---
 
 ## Question 7: Similarity Metrics
 
 **Topic:** Math
 **Difficulty:** Intermediate
 
 ### Question
 Which distance metric is best for calculating similarity between two users' ratings?
 
 ### Answer
 
 **Pearson Correlation Coefficient.**
 - **Why not Cosine?**
    - User A rates everything 5/5.
    - User B rates everything 3/5.
    - Cosine might say they are distinct.
 - **Pearson:** Centers the data (subtracts mean rating).
    - It realizes that for User A, 5 is "Average", but for User B, 3 is "Average".
    - If both rate Movie X above *their* average, they are correlated.
 
 ---
 
 ## Question 8: Jaccard Similarity
 
 **Topic:** Math
 **Difficulty:** Basic
 
 ### Question
 When would you use Jaccard Similarity?
 
 ### Answer
 
 **Use Case:** Binary Data (Implicit Feedback).
 - Did they buy X? (Yes/No).
 - No ratings (Stars).
 
 $$ J(A, B) = \frac{|A \cap B|}{|A \cup B|} $$
 - (Items both bought) / (Items either bought).
 
 ---
 
 ## Question 9: Scalability Challenge
 
 **Topic:** System Design
 **Difficulty:** Advanced
 
 ### Question
 Nearest Neighbor search is $O(N)$ (slow). How do Spotify/YouTube search millions of items instantly?
 
 ### Answer
 
 **ANN (Approximate Nearest Neighbors).**
 - Algorithms like **HNSW**, **Faiss** (Facebook), or **Annoy** (Spotify).
 - They build a graph or tree index.
 - They find items that are *close enough* to the query vector extremely fast, trading a tiny bit of accuracy for massive speedups ($O(\log N)$).
 
 ---
 
 ## Question 10: Utility Matrix
 
 **Topic:** Concept
 **Difficulty:** Basic
 
 ### Question
 What is the Utility Matrix? Why is it "Sparse"?
 
 ### Answer
 
 - **Matrix:** Rows = Users, Cols = Items.
 - **Value:** Rating (1-5).
 - **Sparsity:** A typical user has watched 100 movies out of 10,000 available.
 - 99% of the matrix is Empty (NaN).
 - The goal of Recommender Systems is essentially **Matrix Completion** (Predicting the empty cells).
 
 ---
 
 ## Key Takeaways
 
 - **Content-Based:** Best for cold-start.
 - **Collaborative Filtering:** Best for serendipity (finding unexpected likes).
 - **Item-Item:** Preferred for E-commerce (Amazon).
 - **Implicit Data:** The fuel for modern engines.
 
 **Next:** [Day 32 - Matrix Factorization](../Day-32/README.md)
