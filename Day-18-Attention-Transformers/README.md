# Day 18 - Attention & Transformers
 
 **Topics Covered:** Seq2Seq bottleneck, Attention Mechanism, Self-Attention, Multi-Head Attention, Transformers Architecture, Positional Encoding
 
 ---
 
 ## Question 1: The Bottleneck of RNNs
 
 **Topic:** Concept
 **Difficulty:** Basic
 
 ### Question
 Before Attention, Encoder-Decoder RNNs (Seq2Seq) had a major limitation for long sequences. What was it?
 
 ### Answer
 
 **The Fixed-Length Context Vector.**
 - The Encoder had to compress the entire input sentence (whether 5 words or 100 words) into a single static vector ($h_T$).
 - This context vector acted as a bottleneck.
 - **Result:** The model "forgot" the beginning of long sentences by the time it started generating the translation.
 
 ---
 
 ## Question 2: The Attention Mechanism Intuition
 
 **Topic:** Concept
 **Difficulty:** Basic
 
 ### Question
 Explain the intuition behind "Attention" in one sentence. How does it change the inputs to the Decoder?
 
 ### Answer
 
 **Intuition:** Instead of relying on a single summary vector, let the Decoder "look back" at **all** the Encoder's hidden states and focus (attend) only on the relevant parts for the current word it is generating.
 
 **Mechanism:**
 - At every step of decoding, the model computes a weighted sum of Source Hidden States.
 - If generating "Apple" (French: Pomme), it pays 99% attention to the source word "Apple" and ignores "The".
 
 ---
 
 ## Question 3: "Attention Is All You Need"
 
 **Topic:** History/Architecture
 **Difficulty:** Intermediate
 
 ### Question
 The 2017 Transformer paper was titled "Attention Is All You Need". What did they remove from previous architectures?
 
 ### Answer
 
 **They removed Recurrence (RNNs/LSTMs) entirely.**
 - Previous SOTA: RNN + Attention.
 - Transformer: Attention + Feed Forward Networks.
 - **Why?**
    - RNNs process sequentially ($t$ depends on $t-1$), preventing parallelization.
    - Transformers process the entire sequence at once (Parallelizable), utilizing GPUs efficiency.
 
 ---
 
 ## Question 4: Self-Attention (The Core)
 
 **Topic:** Algorithm
 **Difficulty:** Advanced
 
 ### Question
 Explain the Query (Q), Key (K), and Value (V) analogy in Self-Attention.
 
 ### Answer
 
 **Database Analogy:**
 - **Query:** What I am looking for? (My current token representation).
 - **Key:** What defines the information? (Labels of all other tokens).
 - **Value:** The actual content? (The richness of all other tokens).
 
 **Process:**
 1. **Match:** Calculate similarity between my Query ($Q$) and everyone's Keys ($K$). Score = Dot Product.
 2. **Scale:** Divide by $\sqrt{d_k}$ (for stability).
 3. **Softmax:** Convert scores to probabilities (Weights).
 4. **Aggregate:** Multiply weights by Values ($V$) to get the final representation.
 
 $$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$
 
 ---
 
 ## Question 5: Multi-Head Attention
 
 **Topic:** Algorithm
 **Difficulty:** Advanced
 
 ### Question
 Why do we need "Multi-Head" Attention instead of just one single attention block?
 
 ### Answer
 
 **Analogy:** Listening to a song.
 - Head 1 listens to the **Lyrics** (Semantic meaning).
 - Head 2 listens to the **Bass** (Rhythm/Structure).
 - Head 3 listens to the **Vocals** (Tone).
 
 **Technical:**
 - One attention head focuses on one type of relationship (e.g., subject-verb agreement).
 - Multi-heads allow the model to capture **multiple distinct relationships** simultaneously in different subspaces.
 - Outputs of all heads are concatenated and projected.
 
 ---
 
 ## Question 6: Positional Encoding
 
 **Topic:** Architecture
 **Difficulty:** Intermediate
 
 ### Question
 Since Transformers process all terms in parallel (no recurrence), how do they know the order of words?
 
 ### Answer
 
 **Positional Encoding:**
 - Because the Transformer has no built-in sense of order/time (unlike RNNs), we must inject position information.
 - We add a deterministic vector (based on Sine/Cosine functions of different frequencies) to the input embeddings.
 - $PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})$
 - This acts as a "timestamp" signature added to the word vector.
 
 ---
 
 ## Question 7: Encoder-Decoder in Transformer
 
 **Topic:** Architecture
 **Difficulty:** Intermediate
 
 ### Question
 The original Transformer had both an Encoder and a Decoder. What was the role of each?
 
 ### Answer
 
 - **Encoder Stack (6 layers):**
    - Takes raw input (English).
    - Uses Self-Attention to understand the context of every word relative to every other word.
    - Produces rich contextualized embeddings.
 
 - **Decoder Stack (6 layers):**
    - Takes the previous output tokens (French generated so far).
    - **Masked Self-Attention:** Only looks at past tokens (no cheating).
    - **Cross-Attention:** Looks at the Encoder's output (Focuses on English source).
    - Generates the next token.
 
 ---
 
 ## Question 8: Masked Self-Attention
 
 **Topic:** Architecture
 **Difficulty:** Intermediate
 
 ### Question
 In the Decoder, why must the Self-Attention be "Masked"?
 
 ### Answer
 
 - **Causality:** When predicting the word at position $t$, the model cannot be allowed to see words at $t+1, t+2$ (the future).
 - **Implementation:** We set the attention scores for future positions to $-\infty$ before the Softmax.
 - This turns the probability to 0, effectively hiding future words.
 
 ---
 
 ## Question 9: Scaled Dot-Product
 
 **Topic:** Math
 **Difficulty:** Advanced
 
 ### Question
 Why do we divide by $\sqrt{d_k}$ in the attention formula?
 
 ### Answer
 
 - If $d_k$ (dimension of key vectors) is large (e.g., 512), the dot product $Q \cdot K$ can become extremely large in magnitude.
 - **Effect:** Large inputs to Softmax push the function into regions where gradients are extremely small (vanishing gradients).
 - **Solution:** Scaling down by $\sqrt{d_k}$ keeps the variance of the product at 1, ensuring stable gradients.
 
 ---
 
 ## Question 10: Transformer Impact
 
 **Topic:** General Knowledge
 **Difficulty:** Basic
 
 ### Question
 Name 3 modern models that are based on the Transformer architecture.
 
 ### Answer
 
 1. **BERT (Encoder-only):** Google. Best for Understanding (Classification, QA, NER).
 2. **GPT (Decoder-only):** OpenAI. Best for Generation (Chatbots, Story writing).
 3. **T5/BART (Encoder-Decoder):** Google/Facebook. Best for Translation, Summarization.
 
 ---
 
 ## Key Takeaways
 
 - **Attention** solved the bottleneck problem by allowing focus.
 - **Transformers** replaced RNNs by enabling parallel training.
 - **Self-Attention** ($Q, K, V$) is the mechanism to relate words to each other.
 - **Positional Encodings** are necessary because the model is order-agnostic.
 - This architecture paved the way for the Large Language Model (LLM) era.
 
 **Next:** [Day 19 - BERT](../Day-19/README.md)
