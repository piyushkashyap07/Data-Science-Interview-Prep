# Day 16 - Feature Extraction & Text Representation
 
 **Topics Covered:** Bag of Words (BoW), TF-IDF, N-Grams, Sparse vs Dense Vectors, Cosine Similarity Basics
 
 ---
 
 ## Question 1: Feature Extraction intuition
 
 **Topic:** Concepts
 **Difficulty:** Basic
 
 ### Question
 Why can't machine learning models understand raw text strings? What is the goal of Feature Extraction?
 
 ### Answer
 
 **Why not Strings:**
 - ML algorithms are mathematical functions (matrix multiplication, Dot products). They operate on numerical vectors.
 - "Apple" $\times$ "Orange" is undefined.
 
 **Goal of Feature Extraction:**
 - To convert variable-length text into fixed-length numerical vectors.
 - To preserve the **semantic meaning** of the text in the vector space (i.e., similar texts should have similar vectors).
 
 ---
 
 ## Question 2: CountVectorizer (Bag of Words)
 
 **Topic:** Technique
 **Difficulty:** Basic
 
 ### Question
 How does `CountVectorizer` work? What counts does it store?
 
 ### Answer
 
 - **Mechanism:**
    1. **Learn:** Scan all documents to build a dictionary of unique words (Vocabulary). Size = $V$.
    2. **Transform:** For each document, create a vector of length $V$.
    3. **Count:** If word $w$ appears $n$ times in the doc, set index $i$ to $n$.
 
 - **Example:**
    - Doc 1: "I love AI"
    - Doc 2: "I love ML"
    - Vocab: [AI, I, love, ML]
    - Vec 1: [1, 1, 1, 0]
    - Vec 2: [0, 1, 1, 1]
 
 ---
 
 ## Question 3: TF-IDF (Term Frequency - Inverse Document Frequency)
 
 **Topic:** Technique
 **Difficulty:** Intermediate
 
 ### Question
 What is the flaw in counting simple word frequencies (BoW)? How does TF-IDF solve it? Write the formula.
 
 ### Answer
 
 **Flaw of Counts:**
 - Common words (like "the", "good", "movie" in reviews) appear frequently in *all* documents.
 - They dominate the count vector but carry little specific information to distinguish one document from another.
 
 **TF-IDF Solution:**
 - Penalize words that appear in many documents. Boost words that are rare across the corpus but frequent in the specific document.
 
 **Formula:**
 $$ \text{TF-IDF}(t, d) = \text{TF}(t, d) \times \log\left(\frac{N}{\text{DF}(t)}\right) $$
 - $t$: term, $d$: document, $N$: Total docs.
 - $\text{DF}(t)$: Number of docs containing term $t$.
 - If everyone uses the word ($DF \approx N$), $\log(1) = 0$. The weight becomes 0.
 
 ---
 
 ## Question 4: Sparse vs Dense Matrices
 
 **Topic:** Implementation
 **Difficulty:** Intermediate
 
 ### Question
 Why are BoW and TF-IDF vectors called "Sparse"? What are the computational implications?
 
 ### Answer
 
 **Sparsity:**
 - Vocab size ($V$) might be 50,000 words.
 - A single email might contain only 100 unique words.
 - The vector has 100 non-zero values and 49,900 zeros.
 - **Sparse Matrix:** format stores only (index, value) pairs, not the zeros.
 
 **Implications:**
 - **Memory:** Efficient (Linearly scales with *words in doc*, not *total vocab*).
 - **Compute:** Specialized sparse matrix operations are faster than dense matmul.
 
 ---
 
 ## Question 5: TF-IDF Calculation Example
 
 **Topic:** Math
 **Difficulty:** Intermediate
 
 ### Question
 Calculate TF-IDF for the word "cat".
 - Doc A: "The cat sat" (Total words: 3).
 - Corpus: 100 documents.
 - "cat" appears in 10 documents.
 
 ### Answer
 
 1. **TF (Term Frequency):**
    - Raw count: 1
    - Normalized (freq / total): $1 / 3 \approx 0.33$
 
 2. **IDF (Inverse Document Frequency):**
    - $\log(\frac{\text{Total Docs}}{\text{Docs with 'cat'}})$
    - $\log(\frac{100}{10}) = \log(10) \approx 2.3$ (using base $e$) or $1$ (base 10).
 
 3. **TF-IDF:**
    - $0.33 \times 2.3 = 0.759$
 
 ---
 
 ## Question 6: N-Grams in TF-IDF
 
 **Topic:** Concept
 **Difficulty:** Intermediate
 
 ### Question
 How does using Bigrams (2-grams) in TF-IDF change the Feature Space?
 
 ### Answer
 
 - **Feature Space Exploration:**
    - Unigrams: "New", "York" -> Features [New, York].
    - Bigrams: "New York" -> Feature [New York].
 - **Meaning:** "New York" is a specific entity. "New" is just an adjective. Treating "New York" as a single token captures the location semantic.
 - **Curse of Dimensionality:**
    - Vocab size grows exponentially.
    - $V_{bi} \approx V_{uni}^2$ (worst case).
    - Requires Feature Selection (e.g., maintain only top 5000 frequent bigrams) to be practical.
 
 ---
 
 ## Question 7: Cosine Similarity
 
 **Topic:** Metric
 **Difficulty:** Intermediate
 
 ### Question
 Why do we typically use Cosine Similarity instead of Euclidean Distance for Text Vectors?
 
 ### Answer
 
 - **Euclidean Distance:** Sensitive to magnitude (length calculation).
    - If Doc A ("I like AI") is repeated 10 times -> Doc B ("I like AI ... x10").
    - The content is identical, but Doc B's vector is 10x longer. Euclidean distance is huge.
 
 - **Cosine Similarity:** Measures the **angle** between vectors.
    - $\cos(\theta) = \frac{A \cdot B}{||A|| ||B||}$.
    - It ignores magnitude. Doc A and Doc B point in the exact same direction (Angle 0, Similarity 1). Use this for text.
 
 ---
 
 ## Question 8: Implementing TF-IDF with Scikit-Learn
 
 **Topic:** Implementation
 **Difficulty:** Basic
 
 ### Question
 Write code to transform a list of sentences into a TF-IDF matrix using Scikit-Learn.
 
 ### Answer
 
 ```python
 from sklearn.feature_extraction.text import TfidfVectorizer
 import pandas as pd
 
 corpus = [
     "The quick brown fox",
     "The quick red dog",
     "The lazy dog"
 ]
 
 # 1. Initialize
 vectorizer = TfidfVectorizer(stop_words='english')
 
 # 2. Fit and Transform
 tfidf_matrix = vectorizer.fit_transform(corpus)
 
 # 3. Visualize
 df = pd.DataFrame(tfidf_matrix.toarray(), 
                   columns=vectorizer.get_feature_names_out())
 print(df)
 ```
 
 ---
 
 ## Question 9: OOV (Out of Vocabulary) Problem
 
 **Topic:** Challenges
 **Difficulty:** Intermediate
 
 ### Question
 What happens if your TF-IDF model sees a word during testing that it never saw during training? How do count-based models handle this vs Embeddings?
 
 ### Answer
 
 **Count-Based (TF-IDF/BoW):**
 - It simply **ignores** the word.
 - The column for that word doesn't exist in the training matrix.
 - Use case: If the "important" keyword is new, the model fails.
 
 **Embeddings (Word2Vec/GloVe/FastText):**
 - **Word2Vec:** Also fails (OOV token).
 - **FastText:** Can handle it by looking at sub-word character n-grams (e.g., "Google" -> "Goo", "ogl", "le").
 - **Transformers:** Use sub-word tokenizers that can construct *any* word from base characters.
 
 ---
 
 ## Question 10: One-Hot Encoding
 
 **Topic:** Concept
 **Difficulty:** Basic
 
 ### Question
 Why is One-Hot Encoding generally a bad idea for text?
 
 ### Answer
 
 1. **High Dimensionality:** A vector of size 50,000 for every single word.
 2. **Orthogonality:** The dot product between any two one-hot vectors is 0.
    - "Hotel" $[0, 1, 0]$
    - "Motel" $[0, 0, 1]$
    - Dot product = 0.
    - The model has no way to know that "Hotel" and "Motel" are related concepts.
 
 ---
 
 ## Key Takeaways
 
 - **BoW** counts words but ignores order.
 - **TF-IDF** weighs words by uniqueness, penalizing common stopwords.
 - **Sparse Matrices** are essential for efficiency in traditional NLP.
 - **Cosine Similarity** is the go-to metric for text comparison.
 - **Sparsity & Orthogonality** are the main limitations that lead us to **Word Embeddings** (Day 17).
 
 **Next:** [Day 17 - Word Embeddings](../Day-17/README.md)
