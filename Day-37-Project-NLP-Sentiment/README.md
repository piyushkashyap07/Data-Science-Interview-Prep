# Day 37 - End-to-End NLP Project: Sentiment Analysis
 
 **Project:** Detecting Sentiment (Positive/Negative) in Movie Reviews (IMDb).
 **Focus:** Text Cleaning, Word Embeddings (TF-IDF vs Word2Vec vs BERT), Model Training, and Model Serving (Gradio).
 
 ---
 
 ## 1. Problem Statement
 
 **Goal:** Classify text as Positive (1) or Negative (0).
 **Dataset:** IMDb Large Movie Review Dataset (50k reviews).
 **Challenge:** Handling sarcasm, negation ("not good"), and variable length text.
 
 ---
 
 ## 2. Text Preprocessing
 
 **Steps:**
 1. **Lowercasing:** "Good" == "good".
 2. **Remove HTML tags:** `<br />` is common in IMDb.
 3. **Remove Stopwords?** CAREFUL. "not" is a stopword in libraries like NLTK. Removing "not" changes "not good" to "good". **Keep negation words.**
 4. **Lemmatization:** "Running" -> "Run".
 
 ```python
 import re
 import nltk
 from nltk.corpus import stopwords
 from nltk.stem import WordNetLemmatizer
 
 nltk.download('stopwords')
 nltk.download('wordnet')
 
 def clean_text(text):
     # 1. Remove HTML
     text = re.sub(r'<.*?>', '', text)
     # 2. Lowercase & Regex
     text = re.sub(r'[^a-zA-Z]', ' ', text).lower()
     # 3. Tokenize
     words = text.split()
     # 4. Remove Stopwords (Custom list)
     stop_words = set(stopwords.words('english')) - {'not', 'no', 'nor'}
     words = [w for w in words if w not in stop_words]
     return ' '.join(words)
 ```
 
 ---
 
 ## 3. Feature Engineering: TF-IDF
 
 **Why TF-IDF?** Simple, fast baseline.
 
 ```python
 from sklearn.feature_extraction.text import TfidfVectorizer
 
 vectorizer = TfidfVectorizer(max_features=5000)
 X = vectorizer.fit_transform(df['cleaned_text'])
 y = df['sentiment']
 ```
 
 ---
 
 ## 4. Model 1: Logistic Regression (Baseline)
 
 Always start simple.
 
 ```python
 from sklearn.linear_model import LogisticRegression
 from sklearn.model_selection import train_test_split
 from sklearn.metrics import accuracy_score
 
 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
 
 model = LogisticRegression()
 model.fit(X_train, y_train)
 print("Accuracy:", accuracy_score(y_test, model.predict(X_test)))
 ```
 
 ---
 
 ## 5. Model 2: BERT (State of the Art)
 
 Use Hugging Face `transformers` for superior performance.
 
 ```python
 from transformers import pipeline
 
 # Load pre-trained pipeline (Zero-code approach)
 sentiment_pipeline = pipeline("sentiment-analysis")
 
 data = ["This movie was absolute trash.", "I loved the ending!"]
 results = sentiment_pipeline(data)
 print(results)
 # [{'label': 'NEGATIVE', 'score': 0.99}, {'label': 'POSITIVE', 'score': 0.99}]
 ```
 
 **Fine-Tuning (Snippet):**
 
 ```python
 from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
 
 model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
 tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
 
 # Tokenize dataset...
 # Trainer(model=model, args=training_args, train_dataset=train_dataset...).train()
 ```
 
 ---
 
 ## 6. Serving with Gradio
 
 Build a UI in 3 lines of code.
 
 ```python
 import gradio as gr
 
 def classify(text):
     # Preprocess text
     clean = clean_text(text)
     # Vectorize and Predict using Logistic Regression
     vec = vectorizer.transform([clean])
     pred = model.predict(vec)[0]
     return "Positive" if pred == 1 else "Negative"
 
 interface = gr.Interface(fn=classify, inputs="text", outputs="text")
 interface.launch(share=True)
 ```
 
 ---
 
 ## 7. Interview Questions
 
 **Q: How does TF-IDF handle word context?**
 A: It doesn't. "Dog bites man" and "Man bites dog" have the exact same TF-IDF vector (Bag of Words). This is why BERT (Contextual Embeddings) is better.
 
 **Q: Why did you keep the top 5000 features only?**
 A: Curse of Dimensionality. Too many unique words (vocab size 100k+) leads to overfitting and huge memory usage.
 
 **Q: How would you handle a sentence like "I don't think it was bad"?**
 A: Bag of Words struggles here (sees "bad").
 - N-grams (bi-grams): Captures "not bad".
 - RNN/LSTM/BERT: Captures the full sequence dependency.
 
 ---
 
 ## Key Takeaways
 
 - **Preprocessing** is 80% of NLP. Careful with stopwords!
 - **Baselines:** Always beat Logistic Regression before trying BERT.
 - **Deployment:** Gradio/Streamlit are great for MVPs.
 
 **Next:** [Day 38 - System Design: Recommenders](../Day-38/README.md)
