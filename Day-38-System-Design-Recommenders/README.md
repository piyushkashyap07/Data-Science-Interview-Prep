# Day 38 - ML System Design: Recommender Systems
 
 **Case Study:** Design a Recommendation System for a Video Streaming Platform (e.g., Netflix/YouTube).
 **Focus:** The Funnel Architecture (Candidates -> Retrieval -> Ranking -> Re-ranking), Online vs Offline evaluation, Data Pipelines.
 
 ---
 
 ## 1. Requirements Clarification
 
 **Functional:**
 - Personalization: Show videos relevant to *this* user.
 - Freshness: Show new videos quickly.
 - Latency: Feed must load in < 200ms.
 
 **Non-Functional:**
 - Scale: 1 Billion users, 10 Billion videos.
 - Reliability: System shouldn't crash if one model fails (Fallback to popular videos).
 
 ---
 
 ## 2. High-Level Architecture (The Funnel)
 
 Because we cannot score 10 Billion videos for every user in 200ms, we use a Multi-Stage Funnel.
 
 1. **Candidate Generation (Retrieval):**
    - Input: All 10 Billion videos.
    - Output: ~1000 relevant candidates.
    - Method: Fast, computationally cheap algorithms (Two-Tower, Collaborative Filtering, Heuristics).
 
 2. **Ranking (Scoring):**
    - Input: 1000 candidates.
    - Output: Top 100.
    - Method: Heavy neural networks (Deep & Wide) that predict exact p(Click) or p(Watch).
 
 3. **Re-Ranking (Business Logic):**
    - Input: Top 100.
    - Output: Final 10 for the screen.
    - Logic: Diversity (don't show 10 Cat videos), Filter watched, Harmful content filtration.
 
 ---
 
 ## 3. Candidate Generation Options
 
 **Goal:** High Recall (Don't miss good stuff). Low Precision is okay.
 
 1. **Two-Tower Model (Neural Retrieval):**
    - User Tower computes User Embedding ($U$).
    - Video Tower computes Video Embedding ($V$).
    - Similarity = $U \cdot V$.
    - Use FAISS (ANN) to retrieve top $K$ neighbors in milliseconds.
 2. **Matrix Factorization (ALS):**
    - Classic User-Item similarity.
 3. **Graph-Based:**
    - Random walks on the User-Video graph.
 
 ---
 
 ## 4. The Ranking Layer
 
 **Goal:** High Precision (Show the absolute best).
 **Model:** **Multi-Task Deep & Wide Network**.
 
 - **Inputs:**
    - User Features: Age, Gender, Past History (Last 50 watched).
    - Video Features: Genre, Duration, Embeddings.
    - Context Features: Time of Day, Device (Mobile/TV).
 
 - **Outputs (Multi-Head):**
    - Head 1: Probability of Click (pCTR).
    - Head 2: Probability of Watch completion (pWatch).
 
 - **Final Score:** $Score = p(Click) \times p(Watch)$. (Optimizes for Watch Time, not just clickbait).
 
 ---
 
 ## 5. Training Data Generation
 
 **Labeling is tricky.**
 - **Positive Labels:**
    - Clicked AND Watched > 30 seconds. (Implicit Positive).
 - **Negative Labels:**
    - Impressed (Shown) but NOT Clicked. (Implicit Negative).
 
 **Point-in-Time Correctness:**
 - When training on yesterday's data, use the User History *as it was yesterday*, not today. (Avoid Data Leakage).
 
 ---
 
 ## 6. Evaluation
 
 **Offline Metrics:**
 - Retrieval: Recall@K.
 - Ranking: ROC-AUC, NDCG (Normalized Discounted Cumulative Gain), Precision@K.
 
 **Online Metrics (A/B Testing):**
 - **Primary:** Total Watch Time per User.
 - **Secondary:** Daily Active Users (DAU), Retention Rate.
 - **Guardrail:** Latency (Must not increase > 10%).
 
 ---
 
 ## 7. Handling Freshness
 
 **Problem:** New video uploaded 5 mins ago. Embeddings/Training Data don't exist yet.
 **Solution:**
 1. **Separate "Fresh" Candidate Source:** A simple heuristic rule ("Show trending videos in user's country").
 2. **Content-Based Embedding:** Generate embedding from Video Title/Thumbnail immediately (using BERT/ResNet) without waiting for interaction data.
 3. **Online Learning:** Update Bandit algorithms in real-time.
 
 ---
 
 ## 8. Interview Questions
 
 **Q: Users complain they see the same videos over and over. How to fix?**
 A: Add a "Bloom Filter" or a History Cache. During Re-ranking, filter out disjoint set of {User's Watched History}. Also, encourage **Serendipity** by injecting random exploration items.
 
 **Q: Ranker is too slow (>500ms). What do you do?**
 A: 
 1. Reduce number of candidates (Recalibrate Retrieval).
 2. Quantize Model (Float32 -> Int8).
 3. Distill complex model into a smaller Student model.
 
 ---
 
 ## Key Takeaways
 
 - **Funnel:** Retrieval (Fast/Broad) -> Ranking (Slow/Precise).
 - **Two-Tower:** The standard for modern retrieval.
 - **Multi-Objective:** Optimizing for Clicks gives Clickbait. Optimize for Watch Time/Satisfaction.
 - **Data:** Use Point-in-Time joins to prevent leakage.
 
 **Next:** [Day 39 - System Design: Search & Ads](../Day-39/README.md)
