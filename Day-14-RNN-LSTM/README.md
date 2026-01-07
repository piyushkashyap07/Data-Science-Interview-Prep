# Day 14 - Recurrent Neural Networks (RNNs) & LSTMs
 
 **Topics Covered:** Sequential Data, Standard RNN, BPTT, Vanishing Gradients, LSTM, GRU, Bidirectional RNNs
 
 ---
 
 ## Question 1: Processing Sequential Data
 
 **Topic:** Concepts
 **Difficulty:** Basic
 
 ### Question
 Why can't we use standard Feedforward Neural Networks (MLPs) effectively for sequential data like text or time series?
 
 ### Answer
 
 1. **Fixed Input Size:** MLPs require a fixed-size input vector. Sequences (e.g., sentences) have variable lengths (5 words vs 50 words).
 2. **No Temporal Context:** MLPs process inputs independently. In sequences, the order matters ("not good" vs "good not"). An MLP doesn't share features learned at position $t$ with position $t+1$.
 3. **Parameter Explosion:** To handle long sequences, an MLP would need a massive input layer, leading to parameter explosion.
 
 ---
 
 ## Question 2: RNN Architecture
 
 **Topic:** Architecture
 **Difficulty:** Intermediate
 
 ### Question
 Explain the core concept of a Recurrent Neural Network. What is the "Hidden State"?
 
 ### Answer
 
 **Concept:**
 - RNNs process sequences one step at a time.
 - At each time step $t$, it takes two inputs:
    1. Current input ($x_t$).
    2. Previous hidden state ($h_{t-1}$).
 - It produces an output ($y_t$) and a new hidden state ($h_t$).
 
 **Hidden State ($h_t$):**
 - Acts as the **memory** of the network.
 - It captures information about what has been calculated in all previous steps.
 - Equation: $h_t = \tanh(W_h h_{t-1} + W_x x_t + b)$
 
 ---
 
 ## Question 3: Backpropagation Through Time (BPTT)
 
 **Topic:** Training
 **Difficulty:** Advanced
 
 ### Question
 How does Backpropagation differ for RNNs?
 
 ### Answer
 
 - Since an RNN is essentially a deep network unrolled over time ($T$ steps), we use **BPTT**.
 - To update the weights (which are shared across all time steps), we sum the gradients calculated at each time step.
 - **Process:**
    1. Unroll the network for the sequence length.
    2. Compute Loss $L = \sum L_t$.
    3. Backpropagate errors from $t=T$ down to $t=0$.
    4. Gradient accumulation: $dW = \sum_{t=1}^T \frac{\partial L}{\partial W}_t$.
 
 ---
 
 ## Question 4: RNN Issues (Short vs Long Term)
 
 **Topic:** Challenges
 **Difficulty:** Intermediate
 
 ### Question
 Why do standard RNNs fail to capture long-term dependencies (e.g., remembering a subject from the start of a paragraph)?
 
 ### Answer
 
 **The Vanishing Gradient Problem:**
 - During BPTT, gradients flow backward through time.
 - Because the activation is usually Tanh/Sigmoid, gradients are $< 1$.
 - Over many time steps (e.g., 100 words), repeated multiplication makes gradients exponentially small ($0.5^{100} \approx 0$).
 - **Result:** The weight updates rely only on the immediate past (Short-term memory), forgetting early inputs.
 
 ---
 
 ## Question 5: LSTM (Long Short-Term Memory)
 
 **Topic:** Architecture
 **Difficulty:** Advanced
 
 ### Question
 How does an LSTM solve the Vanishing Gradient problem? Describe the role of the "Cell State".
 
 ### Answer
 
 **Solution:**
 - LSTM separates "Cell State" ($C_t$) from "Hidden State" ($h_t$).
 - **Cell State ($C_t$):** The "fast lane" or conveyor belt for information. It runs straight down the entire chain with only minor linear interactions (addition/multiplication), allowing gradients to flow unchanged (avoiding vanishing).
 - **Gates:** Neural layers that regulate information flow:
    1. **Forget Gate:** What to throw away from old state.
    2. **Input Gate:** What new info to store.
    3. **Output Gate:** What to output as hidden state.
 
 ---
 
 ## Question 6: GRU vs LSTM
 
 **Topic:** Architecture
 **Difficulty:** Intermediate
 
 ### Question
 What is a Gated Recurrent Unit (GRU)? How does it compare to an LSTM?
 
 ### Answer
 
 **GRU:** A simplified version of LSTM.
 - **Design:**
    - Merges Cell State and Hidden State into one ($h_t$).
    - Combines Forget and Input gates into a single "Update Gate".
    - Uses a "Reset Gate".
 
 **Comparison:**
 - **Performance:** Often comparable to LSTM.
 - **Efficiency:** GRU has fewer parameters (2 gates vs 3 gates), so it trains faster and needs less data.
 - **Usage:** GRU is a good default for smaller datasets; LSTM is more powerful for very long/complex sequences.
 
 ---
 
 ## Question 7: Bidirectional RNNs
 
 **Topic:** Architecture
 **Difficulty:** Intermediate
 
 ### Question
 Why would you use a Bidirectional RNN? In what tasks is it not applicable?
 
 ### Answer
 
 **Concept:**
 - Uses two separate RNN layers: one processing forward ($x_1 \to x_T$) and one backward ($x_T \to x_1$).
 - The output at time $t$ combines information from both past and future.
 
 **Use Case:**
 - **NLP (Translation, Named Entity Recognition):** To understand "Teddy" in "Teddy Roosevelt", you need to know "Roosevelt" follows it.
 
 **Not Applicable:**
 - **Real-time processing / Forecasting:** You cannot know the future inputs (e.g., predicting tomorrow's stock price or live speech-to-text with no delay).
 
 ---
 
 ## Question 8: Sequence-to-Sequence (Seq2Seq)
 
 **Topic:** Architecture
 **Difficulty:** Advanced
 
 ### Question
 Explain the Encoder-Decoder architecture used in machine translation.
 
 ### Answer
 
 **Structure:**
 1. **Encoder (RNN/LSTM):** Process the input sentence (sequence) step-by-step and compress it into a final "Context Vector" (the final hidden state).
 2. **Context Vector:** Contains the semantic summary of the input.
 3. **Decoder (RNN/LSTM):** Takes the context vector and generates the output sentence one word at a time.
 
 **Constraint:** Basic Seq2Seq struggles with long sentences because the fixed-size context vector becomes a bottleneck (information loss). **Attention Mechanisms** (Day 15+) solve this.
 
 ---
 
 ## Question 9: LSTM Parameter Count
 
 **Topic:** Implementation
 **Difficulty:** Advanced
 
 ### Question
 Why does an LSTM have 4x the parameters of a simple RNN?
 
 ### Answer
 
 - **Simple RNN:** One neural layer (tanh) inside the cell.
    - $W \cdot [h_{t-1}, x_t] + b$
 - **LSTM:** Four neural layers interacting inside the cell.
    1. Forget Gate (sigmoid)
    2. Input Gate (sigmoid)
    3. Candidate Value (tanh)
    4. Output Gate (sigmoid)
 - Therefore, `Parameters_LSTM` $\approx 4 \times$ `Parameters_RNN` for the same hidden size.
 
 ---
 
 ## Question 10: Sentiment Analysis with Keras
 
 **Topic:** Implementation
 **Difficulty:** Intermediate
 
 ### Question
 Write a Keras model snippet using an LSTM layer for binary sentiment analysis on text data.
 
 ### Answer
 
 ```python
 from tensorflow.keras import layers, models
 
 # Hyperparameters
 vocab_size = 10000  # Number of unique words
 embedding_dim = 32  # Vector size for each word
 max_length = 50     # Sequence length
 
 model = models.Sequential()
 
 # 1. Embedding Layer
 # Converts integer indices to dense vectors
 model.add(layers.Embedding(input_dim=vocab_size, 
                            output_dim=embedding_dim, 
                            input_length=max_length))
 
 # 2. LSTM Layer
 # 64 units, returns only the final hidden state
 model.add(layers.LSTM(64))
 
 # 3. Output Layer
 # Binary classification (Positive/Negative)
 model.add(layers.Dense(1, activation='sigmoid'))
 
 model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
 
 model.summary()
 ```
 
 ---
 
 ## Key Takeaways
 
 - **RNNs** are designed for sequences but suffer from short-term memory key.
 - **LSTMs/GRUs** use gates to control information flow, solving vanishing gradients.
 - **BPTT** is the training algorithm, unrolling the network over time.
 - **Embedding Layers** are essential for feeding text into RNNs.
 - **Bidirectional** models see the future and past (great for NLP).
 
 **Next:** [Day 15 - NLP Transformers](../Day-15/README.md)
