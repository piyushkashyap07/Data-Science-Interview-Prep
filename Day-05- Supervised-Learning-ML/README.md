# 🤖 Machine Learning Interview Questions - Day 5/50
---

## 1️⃣ Supervised vs. Unsupervised Learning

**Supervised Learning:** You teach the model with labeled examples (input + correct answer).
- **Example:** Showing a child pictures of cats and dogs with labels. Later, they can identify new animals.
- **Use cases:** Email spam detection, house price prediction, disease diagnosis

**Unsupervised Learning:** The model finds patterns on its own without labels.
- **Example:** Giving a child mixed toys and they naturally group them by type (cars together, dolls together).
- **Use cases:** Customer segmentation, anomaly detection, recommendation systems

---

## 2️⃣ Handling Imbalanced Datasets

**Problem:** When one class has way more examples (e.g., 95% normal transactions, 5% fraud).

**Solutions:**
- **Resampling:** 
  - Oversample minority class (duplicate fraud cases)
  - Undersample majority class (reduce normal transactions)
  - SMOTE: Create synthetic minority examples
- **Change metrics:** Use precision/recall/F1 instead of accuracy
- **Class weights:** Tell the model to pay more attention to minority class
- **Ensemble methods:** Use algorithms like Random Forest that handle imbalance better

**Example:** In fraud detection with 1 fraud per 100 transactions, accuracy of 99% means nothing if you just predict "not fraud" every time!

---

## 3️⃣ Overfitting vs. Underfitting

**Underfitting:** Model is too simple, doesn't learn enough.
- Like studying only chapter 1 for an exam covering 10 chapters
- **Detect:** Poor performance on both training AND test data
- **Fix:** Use more complex model, add features, train longer

**Overfitting:** Model memorizes training data instead of learning patterns.
- Like memorizing exam answers without understanding concepts
- **Detect:** Great training performance, poor test performance (big gap)
- **Fix:** Get more data, use regularization, reduce model complexity, dropout, early stopping

**Example:** Predicting house prices:
- **Underfitting:** Using only "number of rooms" → too simple
- **Just right:** Rooms + location + size + age
- **Overfitting:** Memorizing exact addresses of training houses

---

## 4️⃣ Bias-Variance Tradeoff

**Bias:** Error from wrong assumptions (underfitting).
- Like using a straight line to fit a curve - systematically wrong

**Variance:** Error from being too sensitive to training data (overfitting).
- Like drawing a wiggly line through every training point - changes wildly with new data

**The Tradeoff:** 
- Simple models: High bias, low variance
- Complex models: Low bias, high variance
- **Goal:** Find the sweet spot in the middle

**Example:** Throwing darts:
- High bias: Always missing left (systematic error)
- High variance: Scattered all over (inconsistent)
- Low bias + low variance: Clustered around bullseye

---

## 5️⃣ Cross-Validation

**What it does:** Tests model performance more reliably by using data multiple times.

**K-Fold Cross-Validation:**
1. Split data into K parts (e.g., 5 parts)
2. Train on 4 parts, test on 1 part
3. Repeat 5 times, each part gets to be test set once
4. Average the 5 scores

**Why it matters:**
- Prevents lucky/unlucky single train-test split
- Uses all data for both training and testing
- Gives better estimate of real-world performance

**Example:** With 100 samples and 5-fold CV:
- Fold 1: Train on 80, test on 20
- Fold 2: Train on different 80, test on different 20
- ... repeat 5 times, average results

---

## 6️⃣ Accuracy, Precision, Recall, and F1

**Confusion Matrix First:**
- True Positive (TP): Correctly predicted positive
- True Negative (TN): Correctly predicted negative
- False Positive (FP): Wrongly predicted positive
- False Negative (FN): Wrongly predicted negative

**Metrics:**

**Accuracy** = (TP + TN) / Total
- Overall correctness
- **Use when:** Classes are balanced

**Precision** = TP / (TP + FP)
- Of all positive predictions, how many were correct?
- **Use when:** False positives are costly (spam filter - don't want real emails marked spam)

**Recall** = TP / (TP + FN)
- Of all actual positives, how many did we catch?
- **Use when:** False negatives are costly (cancer detection - don't want to miss sick patients)

**F1 Score** = 2 × (Precision × Recall) / (Precision + Recall)
- Balance between precision and recall
- **Use when:** You need both, or classes are imbalanced

**Example - Airport Security:**
- High recall: Catch all threats (some innocent flagged)
- High precision: Only flag real threats (might miss some)
- F1: Balance both concerns

---

## 7️⃣ Parametric vs. Non-Parametric Models

**Parametric:** Fixed number of parameters, makes assumptions about data.
- Learns parameters, then discards training data
- **Examples:** Linear Regression, Logistic Regression, Naive Bayes
- **Pros:** Fast, simple, less data needed
- **Cons:** Rigid assumptions may be wrong

**Non-Parametric:** Flexible, grows with data, fewer assumptions.
- Keeps training data around
- **Examples:** K-Nearest Neighbors, Decision Trees, SVMs with RBF kernel
- **Pros:** More flexible, adapts to data
- **Cons:** Needs more data, slower, can overfit

**Example:**
- **Parametric:** Assuming height follows a bell curve (normal distribution) - just need mean and standard deviation
- **Non-Parametric:** Making no assumptions about height distribution - keep all measurements

---

## 8️⃣ How Decision Trees Split

**Goal:** Find splits that create the purest groups (all same class).

**Methods:**

**Gini Impurity:** Probability of incorrect classification
- Gini = 1 - Σ(probability of each class)²
- Lower is better (0 = pure)

**Information Gain (Entropy):**
- Entropy = -Σ(p × log₂(p))
- Choose split that reduces entropy most

**Process:**
1. Try every feature and every possible split point
2. Calculate impurity for each potential split
3. Choose the split that reduces impurity most
4. Repeat recursively for each branch

**Example - Predicting "Play Tennis":**
- Split by Weather: 
  - Sunny: 2 yes, 3 no (impure)
  - Rainy: 3 yes, 2 no (impure)
- Split by Temperature:
  - Hot: 0 yes, 4 no (pure!)
  - Cool: 5 yes, 1 no (mostly pure)
- Temperature wins - better split!

---

## 9️⃣ Regularization (L1 vs L2)

**Why regularize:** Prevent overfitting by penalizing complex models.

**L1 Regularization (Lasso):**
- Adds penalty: α × Σ|weights|
- Forces some weights to exactly zero
- **Result:** Feature selection (removes unimportant features)
- **Use when:** You have many features and want automatic selection

**L2 Regularization (Ridge):**
- Adds penalty: α × Σ(weights²)
- Shrinks all weights but keeps them non-zero
- **Result:** Weights become small and distributed
- **Use when:** All features are potentially useful

**Example - House Price Prediction:**
Without regularization:
- Weight for "yard gnome color" = 50,000 (overfitting noise)

With L1:
- Forces irrelevant features (gnome color) to zero

With L2:
- Makes gnome color weight tiny but keeps it

---

## 🔟 Encoding Categorical Variables

**Problem:** ML models need numbers, not categories.

**Methods:**

**1. Label Encoding:** Assign numbers (0, 1, 2...)
- **Example:** Red=0, Blue=1, Green=2
- **Use when:** Categories have natural order (Small, Medium, Large)
- **Problem:** Implies ordering when there isn't any

**2. One-Hot Encoding:** Create binary columns for each category
- **Example:** 
  - Red → [1, 0, 0]
  - Blue → [0, 1, 0]
  - Green → [0, 0, 1]
- **Use when:** No natural order (colors, countries)
- **Problem:** Many columns if many categories

**3. Target Encoding:** Replace category with target mean
- **Example:** Average house price for each neighborhood
- **Use when:** High cardinality (many unique values)
- **Problem:** Can leak information, needs careful validation

**4. Frequency Encoding:** Replace with how often it appears
- **Example:** "Honda" appears 200 times → encode as 200
- **Use when:** Frequency is informative

**Practical Example - Customer Data:**
```
City (high cardinality) → Target or frequency encoding
Size (S/M/L) → Label encoding (0,1,2)
Color preference → One-hot encoding
```

---

## 📝 **Key Takeaways**

### **Learning Types:**
- **Supervised learning** requires labeled data; best for prediction tasks with known outcomes
- **Unsupervised learning** discovers hidden patterns; ideal for exploration and segmentation
- Choose based on whether you have labels and what insights you need

### **Model Performance:**
- **Overfitting** = memorizing; **Underfitting** = oversimplifying
- Use **cross-validation** (not just train-test split) for reliable performance estimates
- The **bias-variance tradeoff** is about finding the sweet spot between too simple and too complex

### **Evaluation Metrics:**
- **Accuracy** alone is misleading with imbalanced data
- **Precision** matters when false positives are costly
- **Recall** matters when false negatives are costly
- **F1 Score** balances both and works well for imbalanced datasets
- Always consider the business context when choosing metrics

### **Model Selection:**
- **Parametric models** (Linear Regression, Logistic Regression) are fast and interpretable but rigid
- **Non-parametric models** (KNN, Decision Trees) are flexible but need more data and can overfit
- Match model complexity to problem complexity and data size

### **Handling Data Challenges:**
- **Imbalanced datasets:** Resample (SMOTE), adjust class weights, change metrics, or use ensemble methods
- **Categorical variables:** Choose encoding based on cardinality and whether order matters
- **Regularization (L1/L2):** Prevents overfitting; L1 for feature selection, L2 for shrinking coefficients

### **Decision Trees:**
- Split using **Gini impurity** or **Information Gain** to maximize class purity
- Greedy algorithm: locally optimal splits at each node
- Prone to overfitting without pruning or ensemble methods (Random Forests, Gradient Boosting)

### **Interview Success Tips:**
- Connect theory to real-world scenarios (don't just define concepts)
- Explain trade-offs (no algorithm is perfect for everything)
- Mention when you'd use each technique
- Be ready with concrete examples from past projects
- Understand *why* techniques work, not just *what* they do

---

**Pro Tip:** In interviews, always connect theory to practical scenarios. Show you understand both *what* these concepts are and *when* to use them! 🎯

---

**Previous:** [Day 04 - SQL](../Day-04-SQL/README.md) | **Next:** [Day 06 - Supervised Learning (Intermediate)](../Day-06-Supervised-Learning-Intermediate/README.md)
