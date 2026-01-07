# Day 17 - Word Embeddings
 
 **Topics Covered:** Distributed Representations, Word2Vec, Skip-grams, CBOW, GloVe, FastText, Cosine Meaning
 
 ---
 
 ## Question 1: Distributed Representations
 
 **Topic:** Concept
 **Difficulty:** Basic
 
 ### Question
 What is the core idea of "Distributed Representations" or "Embeddings" compared to One-Hot Encoding?
 
 ### Answer
 
 - **One-Hot:** Each word is an island. "King" $[1, 0]$ and "Queen" $[0, 1]$ share nothing.
 - **Embeddings:**
    - Words are mapped to dense vectors of real numbers (e.g., length 100 or 300).
    - The meaning is **distributed** across these dimensions.
    - Example:
       - King: $[0.9, 0.2, 0.5]$ (High "Royalty", Low "Femininity", Medium "Power")
       - Queen: $[0.9, 0.9, 0.5]$ (High "Royalty", High "Femininity", Medium "Power")
    - This allows measuring similarity (King is closer to Queen than to Apple).
 
 ---
 
 ## Question 2: Word2Vec Architecture
 
 **Topic:** Algorithm
 **Difficulty:** Intermediate
 
 ### Question
 Explain the two architectures of Word2Vec: CBOW and Skip-gram. Which one works better for rare words?
 
 ### Answer
 
 **1. CBOW (Continuous Bag of Words):**
 - **Goal:** Predict the **target word** based on context words.
 - **Input:** ["The", "cat", "___", "on", "mat"] -> Output: "sat".
 - **Pros:** Faster training. Better frequency accuracy for frequent words.
 
 **2. Skip-gram:**
 - **Goal:** Predict **context words** given a target word.
 - **Input:** "sat" -> Output: ["The", "cat", "on", "mat"].
 - **Pros:** **Works better for rare words**. Captures fine-grained semantic relationships.
 
 ---
 
 ## Question 3: The "King - Man + Woman = Queen" Equation
 
 **Topic:** Concept
 **Difficulty:** Basic
 
 ### Question
 What does the famous vector operation `Vector("King") - Vector("Man") + Vector("Woman")` result in? What does this demonstrate?
 
 ### Answer
 
 - **Result:** The resulting vector is computationally closest to `Vector("Queen")`.
 - **Demonstration:**
    - It shows that the embedding space captures **linear substructures** of meaning.
    - The difference `King - Man` captures the concept of "Royalty" (or perhaps "Monarch").
    - Adding "Woman" applies the "Gender" direction to that concept.
    - It proves the model understands semantic analogies.
 
 ---
 
 ## Question 4: Negative Sampling
 
 **Topic:** Optimization
 **Difficulty:** Advanced
 
 ### Question
 Calculating the Softmax over a vocabulary of 50,000 words is expensive for every training step. How does Word2Vec solve this?
 
 ### Answer
 
 **Hierarchical Softmax** or **Negative Sampling**.
 
 **Negative Sampling:**
 - Instead of updating weights for *all* 50,000 output neurons (one true 1 and 49,999 zeros).
 - We update the **True context word** (label 1).
 - We randomly select $k$ (e.g., 5-20) **"Negative" words** (words not in context, label 0).
 - We only update weights for these $k+1$ words.
 - This turns a massive classification problem into a cheap binary logistic regression task.
 
 ---
 
 ## Question 5: GloVe (Global Vectors)
 
 **Topic:** Algorithm
 **Difficulty:** Intermediate
 
 ### Question
 How does GloVe differ from Word2Vec (implementation-wise)?
 
 ### Answer
 
 - **Word2Vec (Predictive):** Learns vectors by sequentially predicting context windows (streaming). It learns iteratively (SGD).
 - **GloVe (Count-based):**
    1. First, builds a massive **Co-occurrence Matrix** ($\text{Word} \times \text{Word}$) for the entire corpus. Cell $(i, j)$ = how often word $i$ appears with word $j$.
    2. Factorizes this matrix to reduce dimensions.
 - **Result:** Hybrid approach. Captures global statistics (counts) better than Word2Vec's local window approach.
 
 ---
 
 ## Question 6: FastText
 
 **Topic:** Algorithm
 **Difficulty:** Intermediate
 
 ### Question
 Why is FastText better than Word2Vec for morphologically rich languages (like German or Turkish)?
 
 ### Answer
 
 - **Sub-word Information:** FastText breaks words into character n-grams.
 - **Word2Vec:** Learns "Apple" and "Apples" as two totally unrelated vectors.
 - **FastText:**
    - "Apple" = `<ap, app, ppl, ple, le>`
    - "Apples" = `<ap, app, ppl, ple, les, es>`
    - They share many n-grams, so their vectors will be similar.
 - **OOV:** Can construct a vector for a word never seen before by summing its n-gram vectors.
 
 ---
 
 ## Question 7: Pre-trained Embeddings
 
 **Topic:** Practical
 **Difficulty:** Basic
 
 ### Question
 When should you train your own embeddings vs using pre-trained ones (like Google News Word2Vec)?
 
 ### Answer
 
 **Use Pre-trained:**
 - **General Tasks:** Sentiment analysis, classification on standard English text.
 - **Small Data:** You don't have enough text (millions of words) to learn good quality embeddings yourself.
 
 **Train Your Own:**
 - **Specialized Domain:** Medical (BioBERT/Med), Legal, Finance. "Python" means snake in Wikipedia but code in StackOverflow. General embeddings won't capture your domain jargon.
 - **Large Corpus:** You have a massive dataset available.
 
 ---
 
 ## Question 8: Embedding Layer in Keras
 
 **Topic:** Implementation
 **Difficulty:** Intermediate
 
 ### Question
 How do you load a pre-trained embedding matrix into a Keras Embedding layer and freeze it?
 
 ### Answer
 
 ```python
 from tensorflow.keras.layers import Embedding
 import numpy as np
 
 # Assume embedding_matrix is prepared (shape: Vocab x Dimension)
 # vocab_size = 10000, embedding_dim = 100
 
 layer = Embedding(
     input_dim=vocab_size,
     output_dim=embedding_dim,
     embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
     trainable=False  # FREEZE weights
 )
 ```
 
 ---
 
 ## Question 9: Contextual Limitations
 
 **Topic:** Challenges
 **Difficulty:** Intermediate
 
 ### Question
 What is the fatal flaw of Word2Vec/GloVe regarding polysemy (words with multiple meanings)?
 
 ### Answer
 
 **Static Embeddings:**
 - Word2Vec assigns **one fixed vector** to the word "Bank".
 - **Context 1:** "I sat on the river bank".
 - **Context 2:** "I deposited money in the bank".
 - The model averages these two meanings into a single vector that is neither fully "nature" nor fully "finance".
 - **Solution:** **ELMo** and **BERT** (Contextualized Embeddings) generate dynamic vectors based on the surrounding sentence.
 
 ---
 
 ## Question 10: Visualizing Embeddings
 
 **Topic:** Analysis
 **Difficulty:** Basic
 
 ### Question
 Since embeddings are 300D, how can we visualize them on a 2D screen?
 
 ### Answer
 
 **Dimensionality Reduction Techniques:**
 1. **t-SNE (t-Distributed Stochastic Neighbor Embedding):** Best for preserving local clusters (similar words stay together).
 2. **PCA (Principal Component Analysis):** Preserves global variance but linear.
 3. **UMAP:** Newer, often faster substitute for t-SNE.
 
 **Process:** Project 300D -> 2D, then plot scatter graph.
 
 ---
 
 ## Key Takeaways
 
 - **Embeddings** carry semantic meaning in vector space.
 - **Word2Vec** predicts context; **GloVe** factorizes co-occurrence counts.
 - **FastText** handles unknown words using sub-word n-grams.
 - **Static Embeddings** fail at multiple meanings (Polysemy), leading to the Transformer revolution (Day 18).
 
 **Next:** [Day 18 - Attention & Transformers](../Day-18/README.md)
