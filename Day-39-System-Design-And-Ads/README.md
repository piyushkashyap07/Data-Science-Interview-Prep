# Day 39 - ML System Design: Search & AdTech
 
 **Case Study:** Design a Search Engine Ranking System or an Ad Click Prediction System.
 **Focus:** Text Relevance (BM25), Learning to Rank (LTR), Ad Auctions (eCPM), and Calibration.
 
 ---
 
 ## 1. Problem Statement
 
 **Search:**
 - Input: Query "Best running shoes".
 - Output: Ranked list of webpages.
 - Goal: Maximize NDCG (Relevance).
 
 **Ads:**
 - Input: User + Context + Candidate Ads.
 - Output: One Ad to show.
 - Goal: Maximize Revenue = $p(Click) \times Bid$.
 
 ---
 
 ## 2. Search Architecture (Inverted Index)
 
 You cannot scan all documents.
 
 1. **Indexer:** Creates a Map `Word -> [List of DocIDs]`.
    - "Running": [Doc1, Doc5, Doc99].
 2. **Retrieval (Recall):**
    - Query "Running Shoes" -> Intersection of `List(Running)` and `List(Shoes)`.
    - Returns 10,000 docs.
 3. **Ranking (Precision):**
    - Score these 10,000 docs using ML.
 
 ---
 
 ## 3. Ranking Features
 
 1. **Query-Dependent:**
    - **BM25 / TF-IDF Score:** Text match.
    - **Semantic match:** BERT embedding cosine similarity.
    - **Proximity:** Do "Running" and "Shoes" appear close together in the doc?
 2. **Query-Independent (Static):**
    - **PageRank:** Importance of the page (Backlinks).
    - **Quality Score:** Spam score, Load time, Domain Authority.
 3. **User-Dependent:**
    - **Personalization:** Does this user usually buy shoes? Location (Show local stores).
 
 ---
 
 ## 4. Learning to Rank (LTR)
 
 How to train the model?
 
 1. **Pointwise:** Regress score for each doc. ($Doc_A = 0.9, Doc_B = 0.4$).
    - Problem: Doesn't care about relative order.
 2. **Pairwise (RankNet/LambdaMART):**
    - Train on pairs $(Doc_A, Doc_B)$.
    - Loss: Minimize error if $Doc_A$ is more relevant than $Doc_B$ but ranked lower.
    - **Standard Industry Approach.**
 3. **Listwise:** Optimize the entire list (NDCG) directly. (Hard to train).
 
 ---
 
 ## 5. AdTech: The Auction (eCPM)
 
 Unlike Search, Ads are about Money.
 
 - **Bid:** Advertiser pays $1.00 per click (CPC).
 - **Problem:** If I show an Ad with Bid $100 but 0% chance of click, I make $0.
 - **Solution:** Rank by **eCPM** (Effective Cost Per Mille).
    - $$ eCPM = p(Click) \times Bid $$
 - **Model Goal:** Accurately predict $p(Click)$ (CTR).
 
 ---
 
 ## 6. Calibration
 
 **Topic:** Accuracy vs Probability
 **Difficulty:** Advanced
 
 ### Question
 Your XGBoost model outputs 0.8. Does that mean there is an 80% chance of a click?
 
 ### Answer
 
 - Not necessarily. Trees/Neural Nets are not naturally probabilistic.
 - **Calibration:** Aligning predicted probabilities with actual observed frequencies.
 - If model says 0.8 for 100 items, and exactly 80 of them are clicked, it is **Calibrated**.
 - **Method:** Isotonic Regression or Platt Scaling.
 - **Why Critical?** In Ads, $p(Click)$ is a multiplier for money. If it's off by 10%, Revenue is off by 10%.
 
 ---
 
 ## 7. Online Learning for Ads
 
 Ads have a short lifespan (Black Friday Sale).
 - **Batch Learning:** Train once a day. (Too slow. The sale is over).
 - **Online Learning:** FTRL (Follow The Regularized Leader) Optimizer.
    - Updates weights *immediately* after every click/no-click.
    - Handles sparse features (Millions of keywords) efficiently.
 
 ---
 
 ## 8. Interview Questions
 
 **Q: How to handle "Position Bias"?**
 A: Top results get clicked just because they are on top, not because they are relevant.
 - **Training:** Add "Position" as a feature.
 - **Inference:** Set "Position = 1" for all candidates to compare them fairly.
 
 **Q: What metric optimizes for User Satisfaction in Search?**
 A: **NDCG (Normalized Discounted Cumulative Gain)**. It rewards correct ranking, and heavily penalizes putting relevant items at the bottom.
 
 ---
 
 ## Key Takeaways
 
 - **Search:** Driven by Relevance (BM25 + Semantic).
 - **Ads:** Driven by Revenue (CTR $\times$ Bid).
 - **Calibration:** Essential when probabilities map to real money.
 - **Pairwise Loss:** The standard for Ranking problems.
 
 **Next:** [Day 40 - Final Prep](../Day-40/README.md)
