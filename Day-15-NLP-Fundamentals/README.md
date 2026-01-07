# Day 15 - NLP Fundamentals
 
 **Topics Covered:** Text Preprocessing, Tokenization, Stemming, Lemmatization, Stopwords, Regular Expressions (Regex), Part-of-Speech Tagging
 
 ---
 
 ## Question 1: Text Preprocessing Pipeline
 
 **Topic:** NLP
 **Difficulty:** Basic
 
 ### Question
 Describe the standard text preprocessing pipeline. Why is it necessary before feeding text to a model?
 
 ### Answer
 
 **Pipeline:**
 1. **Lowercasing:** Standardize case (Hello == hello).
 2. **Noise Removal:** Remove HTML tags, punctuation, special characters.
 3. **Tokenization:** Split text into words/subwords.
 4. **Stopword Removal:** Remove common words (and, the, is) that carry little meaning.
 5. **Normalization:** Stemming or Lemmatization (converting words to root form).
 
 **Necessity:**
 - Raw text is unstructured and noisy.
 - Models work with numbers/vectors, not strings.
 - Reduces vocabulary size (dimensionality), improving model efficiency and generalization.
 
 ---
 
 ## Question 2: Tokenization
 
 **Topic:** NLP
 **Difficulty:** Basic
 
 ### Question
 What is tokenization? Difference between word-level, character-level, and subword tokenization?
 
 ### Answer
 
 **Definition:** Breaking a string into smaller units calls "tokens".
 
 | Type | Example ("smart AI") | Pros | Cons |
 |------|----------------------|------|------|
 | **Word-level** | `['smart', 'AI']` | Intuitive | Huge vocabulary, "Out of Vocabulary" (OOV) issues |
 | **Character-level** | `['s', 'm', 'a', 'r', 't', ' ', 'A', 'I']` | Tiny vocab, no OOV | Loss of meaning, very long sequences |
 | **Subword (BPE/WordPiece)** | `['smart', 'A', '##I']` | **Best balance:** Handles rare words by breaking them down, keeps common words whole | Complex implementation |
 
 *Modern LLMs (BERT, GPT) use Subword tokenization.*
 
 ---
 
 ## Question 3: Stemming vs Lemmatization
 
 **Topic:** NLP
 **Difficulty:** Intermediate
 
 ### Question
 Explain the difference between Stemming and Lemmatization. Provide examples.
 
 ### Answer
 
 **Stemming:**
 - **Method:** Heuristic rule-based chopping of suffixes.
 - **Speed:** Fast.
 - **Accuracy:** Lower (can produce non-words).
 - **Example:** "Caring" -> "Car", "Better" -> "Better"
 - **Tool:** PorterStemmer.
 
 **Lemmatization:**
 - **Method:** Vocabulary and morphological analysis (uses a dictionary).
 - **Speed:** Slower.
 - **Accuracy:** High (returns the "Lemma" or dictionary root).
 - **Example:** "Caring" -> "Care", "Better" -> "Good"
 - **Tool:** WordNet Lemmatizer, Spacy.
 
 ---
 
 ## Question 4: Stopwords
 
 **Topic:** NLP
 **Difficulty:** Basic
 
 ### Question
 Why do we remove stopwords? When might you strictly preserve them?
 
 ### Answer
 
 **Why remove:**
 - Words like "the", "is", "at" appear very frequently but carry low semantic information.
 - Removing them reduces dataset size and noise for Bag-of-Words models.
 
 **When to KEEP them:**
 - **Sequence Models (RNN/BERT/GPT):** The grammar and structure matter. "To be or not to be" consists entirely of stopwords but has profound meaning.
 - **Sentiment Analysis:** "Not" is a stopword, but "Not good" is opposite to "Good". Removing "Not" flips the meaning.
 
 ---
 
 ## Question 5: Regular Expressions (Regex)
 
 **Topic:** Implementation
 **Difficulty:** Intermediate
 
 ### Question
 Write a Python Regex pattern to extract all email addresses from a given text.
 
 ### Answer
 
 ```python
 import re
 
 text = "Contact us at support@example.com or sales@test.co.uk for details."
 
 # Pattern Explanation:
 # [a-zA-Z0-9._%+-]+ : Username (alphanumeric + chars)
 # @                 : At symbol
 # [a-zA-Z0-9.-]+    : Domain name
 # \.                : Dot
 # [a-zA-Z]{2,}      : Top level domain (at least 2 chars)
 pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
 
 emails = re.findall(pattern, text)
 print(emails)
 # Output: ['support@example.com', 'sales@test.co.uk']
 ```
 
 ---
 
 ## Question 6: Bag of Words (BoW)
 
 **Topic:** Feature Extraction
 **Difficulty:** Basic
 
 ### Question
 Explain the Bag of Words model. What is its major limitation?
 
 ### Answer
 
 **Concept:**
 - Represents text as a fixed-length vector counting word frequency.
 - **Steps:**
    1. Build vocabulary from all documents.
    2. Convert valid document into a vector size of vocabulary.
    3. Value at index $i$ = count of word $i$.
 
 **Limitation:**
 - **No meaningful order:** "Dog bites man" and "Man bites dog" have the exact same BoW vector.
 - **Sparsity:** Vectors are mostly zeros.
 - **No Semantic meaning:** "Happy" and "Joyful" are orthogonal vectors (no relationship captured).
 
 ---
 
 ## Question 7: Part-of-Speech (POS) Tagging
 
 **Topic:** NLP Concepts
 **Difficulty:** Intermediate
 
 ### Question
 What is POS tagging? How is it useful in Named Entity Recognition (NER)?
 
 ### Answer
 
 **Definition:**
 - Assigning a grammatical category (Noun, Verb, Adjective, etc.) to each token in a sentence.
 - Example: "Apple (Noun) looks (Verb) great (Adj)".
 
 **Utility in NER:**
 - Named Entities (People, Organizations, Locations) are almost exclusively **Nouns** or **Noun Phrases**.
 - POS tagging acts as a critical first step feature for NER systems to narrow down candidates.
 
 ---
 
 ## Question 8: N-Grams
 
 **Topic:** Feature Extraction
 **Difficulty:** Basic
 
 ### Question
 What are N-Grams? How do they help with the "context" problem of Bag of Words?
 
 ### Answer
 
 **Definition:** Contiguous sequence of $N$ items from text.
 - **Unigram:** "New", "York"
 - **Bigram:** "New York"
 - **Trigram:** "New York City"
 
 **Context:**
 - By capturing pairs or triplets of words, simple "Bag of N-Grams" can capture local context like "New York" (Location) vs "New" (Adjective) and "York" (Noun).
 - **Cost:** Exponential increase in vocabulary size ($V^N$).
 
 ---
 
 ## Question 9: Preprocessing with NLTK
 
 **Topic:** Implementation
 **Difficulty:** Intermediate
 
 ### Question
 Write a Python function using NLTK to tokenize, remove stopwords, and stem a sentence.
 
 ### Answer
 
 ```python
 import nltk
 from nltk.corpus import stopwords
 from nltk.tokenize import word_tokenize
 from nltk.stem import PorterStemmer
 
 # Ensure resources are downloaded
 # nltk.download('punkt')
 # nltk.download('stopwords')
 
 def preprocess(text):
     # 1. Tokenize
     tokens = word_tokenize(text.lower())
     
     # 2. Stopwords
     stop_words = set(stopwords.words('english'))
     filtered_tokens = [w for w in tokens if w not in stop_words]
     
     # 3. Stemming
     stemmer = PorterStemmer()
     stemmed = [stemmer.stem(w) for w in filtered_tokens]
     
     return stemmed
 
 sentence = "The runners are running quickly to the finish line."
 print(preprocess(sentence))
 # Output: ['runner', 'run', 'quickli', 'finish', 'line']
 ```
 
 ---
 
 ## Question 10: Levenshtein Distance
 
 **Topic:** Algorithm
 **Difficulty:** Advanced
 
 ### Question
 What is Levenshtein Distance? How is it used in spell checking?
 
 ### Answer
 
 **Definition:**
 - The minimum number of single-character edits (insertions, deletions, substitutions) required to change one word into the other.
 
 **Example:**
 - "kitten" -> "sitting"
    1. **s**itting (substitute k->s)
    2. sittin**g** (insert g)
    3. sittin (substitute e->i)
 - Distance = 3.
 
 **Application:**
 - If a user types "aple", calculate distance to dictionary words.
 - "apple" (dist 1) is closer than "apply" (dist 2).
 - Suggest the word with minimum edit distance.
 
 ---
 
 ## Key Takeaways
 
 - **Preprocessing** is 80% of NLP work. Garbage in, Garbage out.
 - **Tokenization** choice (word/subword) impacts model architecture.
 - **Stemming** is aggressive; **Lemmatization** is linguistic.
 - **Stopwords** are noise for simple models but context for deep models.
 - **Regex** is the "Swiss Army Knife" for string manipulation.
 
 **Next:** [Day 16 - Feature Extraction](../Day-16/README.md)
