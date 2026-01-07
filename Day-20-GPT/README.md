# Day 20 - GPT & Decoder Models
 
 **Topics Covered:** Decoder-only Architecture, Causal Language Modeling (CLM), Generative AI, Zero-shot vs Few-shot Learning, Temperature, Top-k/Top-p Sampling
 
 ---
 
 ## Question 1: GPT Architecture
 
 **Topic:** Architecture
 **Difficulty:** Basic
 
 ### Question
 What does GPT stand for? How does its architecture differ from BERT?
 
 ### Answer
 
 **GPT:** **G**enerative **P**re-trained **T**ransformer.
 
 **Difference:**
 - **BERT:** Encoder-only. Bidirectional (Sees future). Excellent at understanding.
 - **GPT:** Decoder-only. Auto-regressive (Can only look back). Excellent at generation.
 
 ---
 
 ## Question 2: Causal Language Modeling (CLM)
 
 **Topic:** Pre-training
 **Difficulty:** Intermediate
 
 ### Question
 Explain the training objective of GPT. Why is it called "Causal"?
 
 ### Answer
 
 **Objective:** Next Token Prediction.
 - Given tokens $x_1, ..., x_{t-1}$, maximize the probability of $x_t$.
 - $P(x) = \prod P(x_i | x_1...x_{i-1})$.
 
 **Causal:**
 - The prediction for time $t$ depends *only* on the past (cause), never on the future.
 - This mirrors how humans generate speech/text (linearly).
 
 ---
 
 ## Question 3: The Scaling Hypothesis
 
 **Topic:** Concept
 **Difficulty:** Basic
 
 ### Question
 What was the main discovery from GPT-1 to GPT-2 to GPT-3?
 
 ### Answer
 
 **Scale is all you need.**
 - The architecture remained largely the same (Transformer Decoder).
 - **GPT-1:** 117M parameters.
 - **GPT-2:** 1.5B parameters. (Coherent paragraphs).
 - **GPT-3:** 175B parameters. (Human-level reasoning, coding, translation).
 - **Observation:** Performance improves properly with model size, data size, and compute (Scaling Laws).
 
 ---
 
 ## Question 4: Zero-Shot vs Few-Shot Learning
 
 **Topic:** Paradigm Shift
 **Difficulty:** Intermediate
 
 ### Question
 Explain the concept of "In-Context Learning" introduced by GPT-3.
 
 ### Answer
 
 Tradition ML required "Fine-tuning" (updating weights) for a new task. GPT introduced **In-Context Learning**:
 
 1. **Zero-Shot:** Give the task description without examples.
    - Prompt: "Translate to French: Hello" -> Output: "Bonjour".
 
 2. **One-Shot:** Give one example.
    - Prompt: "Sea -> Mer. Hello ->"
 
 3. **Few-Shot:** Give multiple examples to set the pattern.
   - Prompt: "Sea -> Mer. Dog -> Chien. Hello ->"
 
 **Key:** The model weights are NOT updated. It learns "on the fly" from the prompt context.
 
 ---
 
 ## Question 5: Temperature
 
 **Topic:** Inference
 **Difficulty:** Intermediate
 
 ### Question
 When generating text, what does the "Temperature" parameter control? What happens if T=0 vs T=1?
 
 ### Answer
 
 **Controls Randomness/Creativity.**
 - Softmax: $\frac{e^{z_i/T}}{\sum e^{z_j/T}}$
 
 - **T < 1 (Low):** Sharpens the probability distribution. The model becomes confident, repetitive, and deterministic. **T=0** is greedy decoding (always picks the #1 most likely word).
 - **T > 1 (High):** Flattens the distribution. Low probability words get boosted. Use for creative writing/poetry.
 
 ---
 
 ## Question 6: Sampling Strategies (Top-k vs Top-p)
 
 **Topic:** Inference
 **Difficulty:** Advanced
 
 ### Question
 Why simple random sampling is bad, and how Top-k or Nucleus (Top-p) Sampling fixes it.
 
 ### Answer
 
 **Bad Random Sampling:** Might pick a very low probability word (0.0001%) that makes no sense, derailing the sentence.
 
 **Top-k Sampling:**
 - Only sample from the top $k$ (e.g., 50) most likely words.
 - **Issue:** In some contexts, there are only 2 good words; in others, 100. K is fixed/rigid.
 
 **Nucleus (Top-p) Sampling:**
 - Sample from the smallest set of words whose cumulative probability exceeds $p$ (e.g., 0.90).
 - **Dynamic:** If confident, the set is small (maybe 2 words). If unsure, the set grows. **Standard for modern Chatbots.**
 
 ---
 
 ## Question 7: Hallucinations
 
 **Topic:** Challenges
 **Difficulty:** Basic
 
 ### Question
 What is "Hallucination" in LLMs? Why does it happen?
 
 ### Answer
 
 **Definition:** The model generating text that sounds plausible and authoritative but is factually incorrect or nonsensical.
 
 **Why?**
 - LLMs are **probabilistic token predictors**, not knowledge bases.
 - They prioritize coherence and grammar over truth.
 - If the most probable next word completes a sentence beautifully but incorrectly, the model will pick it.
 
 ---
 
 ## Question 8: RLHF (Reinforcement Learning from Human Feedback)
 
 **Topic:** Training
 **Difficulty:** Advanced
 
 ### Question
 How did ChatGPT transition from a raw completion engine (GPT-3) to a helpful assistant?
 
 ### Answer
 
 **RLHF Pipeline:**
 1. **SFT (Supervised Fine-Tuning):** Human contractors write "Instruction -> Response" pairs. Train GPT to imitate this conversation style.
 2. **Reward Model:** Humans rank different model outputs from best to worst. Train a generic "Reward Model" to predict human preference.
 3. **PPO (Reinforcement Learning):** Use the Reward Model to optimize the GPT policy to generate responses that humans prefer (Helpful, Honest, Harmless).
 
 ---
 
 ## Question 9: Context Window
 
 **Topic:** Architecture Constraint
 **Difficulty:** Basic
 
 ### Question
 What is the "Context Window"? What happens if you exceed it?
 
 ### Answer
 
 **Definition:** The maximum number of tokens the model can "keep in mind" (Input + Output).
 - GPT-3: 2048 tokens.
 - GPT-4: 128k tokens.
 
 **Exceeding it:**
 - The model physically cannot process it.
 - You must truncate the input (forgetting earlier conversation) or use RAG (Retrieval augmented generation) to feed only relevant snippets.
 
 ---
 
 ## Question 10: Generating Text with Transformers library
 
 **Topic:** Implementation
 **Difficulty:** Basic
 
 ### Question
 Write a Python script to load `gpt2` and generate text.
 
 ### Answer
 
 ```python
 from transformers import pipeline, set_seed
 
 generator = pipeline('text-generation', model='gpt2')
 set_seed(42)
 
 prompt = "The future of AI is"
 
 result = generator(prompt, max_length=30, num_return_sequences=1)
 
 print(result[0]['generated_text'])
 # Output: "The future of AI is bright, but we must ensure it remains safe..."
 ```
 
 ---
 
 ## Key Takeaways
 
 - **GPT** is a decoder-only model that predicts the next token.
 - **In-Context Learning** allows usage without fine-tuning.
 - **Sampling** (Top-p, Temperature) controls the flavor of the text.
 - **RLHF** aligns the raw model with human intent.
 - **Scale** (Parameters + Data) drives emergent abilities.
 
 **Next:** [Day 21 - Practical NLP](../Day-21/README.md)
