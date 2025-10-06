# 📊 Model Evaluation & Feature Engineering - Day 7/40

**Topics Covered:** Train-Test-Validation Split, Cross-validation, Feature Scaling, Feature Selection, Multicollinearity, Model Evaluation Metrics

---

## Question 1: Training, Validation, and Test Sets

**Topic:** Model Evaluation  
**Difficulty:** Intermediate

### Question
What's the difference between training, validation, and test sets, and why are all three needed?

### Answer

These three sets serve distinct purposes in the machine learning pipeline. Using all three prevents data leakage and provides unbiased performance estimates.

#### Understanding Each Set:

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

print("="*60)
print("TRAIN-VALIDATION-TEST SPLIT EXPLAINED")
print("="*60)

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                           n_redundant=5, random_state=42)

# Split into train+val (80%) and test (20%)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Split train+val into train (60% of total) and validation (20% of total)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
)  # 0.25 of 80% = 20% of total

print("\n📊 Dataset Splits:")
print(f"   Total samples: {len(X)}")
print(f"   Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.0f}%)")
print(f"   Validation set: {len(X_val)} samples ({len(X_val)/len(X)*100:.0f}%)")
print(f"   Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.0f}%)")

print("\n" + "="*60)
print("PURPOSE OF EACH SET")
print("="*60)

purposes = {
    '1️⃣ TRAINING SET (60%)': {
        'purpose': 'Train model parameters',
        'what_happens': 'Model learns patterns from this data',
        'examples': [
            'Neural network learns weights',
            'Decision tree finds splits',
            'Linear model finds coefficients'
        ],
        'sees_data': 'YES - model trains on this'
    },
    '2️⃣ VALIDATION SET (20%)': {
        'purpose': 'Tune hyperparameters & select models',
        'what_happens': 'Evaluate different model configurations',
        'examples': [
            'Choose best learning rate',
            'Select optimal max_depth',
            'Compare different algorithms',
            'Early stopping in neural networks'
        ],
        'sees_data': 'NO training, but used for decisions'
    },
    '3️⃣ TEST SET (20%)': {
        'purpose': 'Final, unbiased performance estimate',
        'what_happens': 'Evaluate final model (once only!)',
        'examples': [
            'Report final accuracy to stakeholders',
            'Estimate real-world performance',
            'Compare with baseline/competitors'
        ],
        'sees_data': 'NO - completely held out'
    }
}

for set_name, details in purposes.items():
    print(f"\n{set_name}")
    print(f"   Purpose: {details['purpose']}")
    print(f"   What happens: {details['what_happens']}")
    print(f"   Model sees data: {details['sees_data']}")
    print(f"   Examples:")
    for example in details['examples']:
        print(f"      - {example}")
```

#### Practical Example - Model Selection:

```python
print("\n\n" + "="*60)
print("PRACTICAL EXAMPLE: HYPERPARAMETER TUNING")
print("="*60)

# Train multiple models with different hyperparameters
from sklearn.metrics import accuracy_score

hyperparameters = [
    {'max_depth': 3, 'n_estimators': 50},
    {'max_depth': 5, 'n_estimators': 100},
    {'max_depth': 10, 'n_estimators': 150},
    {'max_depth': None, 'n_estimators': 200}
]

results = []

for i, params in enumerate(hyperparameters, 1):
    # Train on training set
    model = RandomForestClassifier(**params, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate on training set (will be overly optimistic)
    train_score = accuracy_score(y_train, model.predict(X_train))
    
    # Evaluate on validation set (used for selection)
    val_score = accuracy_score(y_val, model.predict(X_val))
    
    results.append({
        'config': i,
        'params': params,
        'train_score': train_score,
        'val_score': val_score
    })
    
    print(f"\nConfiguration {i}: {params}")
    print(f"   Training accuracy: {train_score:.3f}")
    print(f"   Validation accuracy: {val_score:.3f}")
    print(f"   Gap: {abs(train_score - val_score):.3f}")

# Select best model based on validation performance
best_config = max(results, key=lambda x: x['val_score'])
print(f"\n{'='*60}")
print(f"BEST MODEL SELECTED (based on validation set):")
print(f"   Configuration: {best_config['config']}")
print(f"   Parameters: {best_config['params']}")
print(f"   Validation accuracy: {best_config['val_score']:.3f}")

# Train final model with best hyperparameters
final_model = RandomForestClassifier(**best_config['params'], random_state=42)
final_model.fit(X_train, y_train)

# NOW evaluate on test set (only once!)
test_score = accuracy_score(y_test, final_model.predict(X_test))

print(f"\n{'='*60}")
print(f"FINAL EVALUATION (on held-out test set):")
print(f"   Test accuracy: {test_score:.3f}")
print(f"   ✅ This is our unbiased performance estimate!")
```

#### Common Mistakes:

```python
print("\n\n" + "="*60)
print("COMMON MISTAKES TO AVOID")
print("="*60)

mistakes = {
    '❌ Mistake 1: No Validation Set': {
        'problem': 'Using test set for hyperparameter tuning',
        'consequence': 'Test set becomes "seen" data, overoptimistic results',
        'example': 'Trying 100 models on test set, picking best one',
        'solution': 'Always use separate validation set for tuning'
    },
    '❌ Mistake 2: Data Leakage': {
        'problem': 'Fitting preprocessing on all data before split',
        'consequence': 'Information leaks from test to train',
        'example': 'Scaling data before train-test split',
        'solution': 'Fit scaler on train set only, transform test set'
    },
    '❌ Mistake 3: Multiple Test Evaluations': {
        'problem': 'Evaluating on test set multiple times',
        'consequence': 'Test set no longer unbiased',
        'example': 'Tweaking model after seeing test performance',
        'solution': 'Evaluate test set only once, after all decisions made'
    },
    '❌ Mistake 4: Ignoring Class Balance': {
        'problem': 'Not using stratified splits',
        'consequence': 'Unequal class distribution across sets',
        'example': 'All positive samples end up in training set',
        'solution': 'Use stratify parameter in train_test_split'
    },
    '❌ Mistake 5: Wrong Split Proportions': {
        'problem': 'Too small test/validation sets',
        'consequence': 'Unreliable performance estimates',
        'example': 'Using 98% train, 1% val, 1% test',
        'solution': 'Typical: 60/20/20 or 70/15/15 split'
    }
}

for mistake, details in mistakes.items():
    print(f"\n{mistake}: {details['problem']}")
    print(f"   Consequence: {details['consequence']}")
    print(f"   Example: {details['example']}")
    print(f"   Solution: {details['solution']}")
```

#### Alternative: K-Fold Cross-Validation:

```python
print("\n\n" + "="*60)
print("ALTERNATIVE: CROSS-VALIDATION (Small Datasets)")
print("="*60)

from sklearn.model_selection import cross_val_score

# When dataset is small, use cross-validation instead of validation set
# Still keep test set separate!

X_train_full = np.vstack([X_train, X_val])  # Combine train and validation
y_train_full = np.hstack([y_train, y_val])

print("\n📊 For small datasets:")
print(f"   Training + Validation: {len(X_train_full)} samples (80%)")
print(f"   Test (still held out): {len(X_test)} samples (20%)")
print("\n   Use cross-validation on training+validation")
print("   Test set remains untouched until final evaluation")

# Hyperparameter tuning with cross-validation
for i, params in enumerate(hyperparameters[:2], 1):  # Just show 2 for brevity
    model = RandomForestClassifier(**params, random_state=42)
    cv_scores = cross_val_score(model, X_train_full, y_train_full, cv=5)
    
    print(f"\nConfig {i}: {params}")
    print(f"   CV scores: {[f'{s:.3f}' for s in cv_scores]}")
    print(f"   Mean CV: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
```

#### Decision Framework:

```python
print("\n\n" + "="*60)
print("WHEN TO USE WHICH APPROACH")
print("="*60)

decision_guide = {
    'Large Dataset (>10,000 samples)': {
        'approach': 'Train-Validation-Test Split',
        'split': '60% Train / 20% Validation / 20% Test',
        'reason': 'Enough data for all three sets',
        'example': 'Image classification with 100,000 images'
    },
    'Medium Dataset (1,000-10,000 samples)': {
        'approach': 'Train-Validation-Test or Cross-Validation',
        'split': '70% Train / 15% Val / 15% Test OR 80% Train / 20% Test + CV',
        'reason': 'Balance between having enough data and reliable estimates',
        'example': 'Customer churn prediction with 5,000 customers'
    },
    'Small Dataset (<1,000 samples)': {
        'approach': 'Cross-Validation + Hold-out Test',
        'split': '80% for CV / 20% Test',
        'reason': 'Maximize training data while maintaining test set',
        'example': 'Medical diagnosis with 500 patient records'
    },
    'Very Small Dataset (<100 samples)': {
        'approach': 'Leave-One-Out CV or Nested CV',
        'split': 'LOOCV or Nested 5-Fold CV',
        'reason': 'Need to use data as efficiently as possible',
        'example': 'Rare disease study with 50 patients'
    }
}

for scenario, details in decision_guide.items():
    print(f"\n📊 {scenario}:")
    print(f"   Approach: {details['approach']}")
    print(f"   Split: {details['split']}")
    print(f"   Reason: {details['reason']}")
    print(f"   Example: {details['example']}")
```

**Key Visualization:**

```
┌─────────────────────────────────────────────────────────┐
│                    Full Dataset (100%)                  │
└─────────────────────────────────────────────────────────┘
         │
         ├──────────────────────────────────────┐
         │                                      │
    ┌────────────────┐                   ┌───────────┐
    │  Train+Val     │                   │   Test    │
    │     (80%)      │                   │   (20%)   │
    └────────────────┘                   └───────────┘
         │                                      │
         ├────────────────┐                    │
         │                │                    │
    ┌─────────┐     ┌──────────┐         Held out
    │  Train  │     │   Val    │         until final
    │  (60%)  │     │  (20%)   │         evaluation
    └─────────┘     └──────────┘              │
         │                │                    │
    Learn model    Tune hyper-           Unbiased
    parameters     parameters            performance
```

**Key Takeaways:**
- Training set: Learn model parameters
- Validation set: Select hyperparameters and models
- Test set: Final unbiased evaluation (use only once!)
- Never tune on test set (causes overfitting to test data)
- Use stratified splits for imbalanced datasets
- For small datasets, use cross-validation instead of separate validation set

---

## Question 2: Cross-Validation vs. Train-Test Split

**Topic:** Model Evaluation  
**Difficulty:** Intermediate

### Question
When should you use cross-validation versus a simple train-test split?

### Answer

The choice depends on dataset size, computational resources, and the reliability needed in performance estimates.

#### Direct Comparison:

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
import time

print("="*60)
print("CROSS-VALIDATION VS TRAIN-TEST SPLIT")
print("="*60)

# Generate datasets of different sizes
dataset_sizes = [100, 500, 5000]

for n_samples in dataset_sizes:
    print(f"\n\n{'='*60}")
    print(f"DATASET SIZE: {n_samples} samples")
    print('='*60)
    
    X, y = make_classification(
        n_samples=n_samples, n_features=20, n_informative=15,
        random_state=42
    )
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # METHOD 1: Train-Test Split
    print("\n1️⃣ TRAIN-TEST SPLIT (80-20)")
    print("-" * 50)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    start = time.time()
    model.fit(X_train, y_train)
    score_single = model.score(X_test, y_test)
    time_single = time.time() - start
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    print(f"   Accuracy: {score_single:.3f}")
    print(f"   Time: {time_single:.3f}s")
    
    # Test variance by trying different random states
    scores_multiple = []
    for rs in range(10):
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=rs
        )
        model.fit(X_tr, y_tr)
        scores_multiple.append(model.score(X_te, y_te))
    
    print(f"   Variance across splits: {np.std(scores_multiple):.3f}")
    print(f"   ⚠️ Single split can be misleading!")
    
    # METHOD 2: Cross-Validation
    print("\n2️⃣ CROSS-VALIDATION (5-Fold)")
    print("-" * 50)
    
    start = time.time()
    cv_scores = cross_val_score(model, X, y, cv=5)
    time_cv = time.time() - start
    
    print(f"   Fold scores: {[f'{s:.3f}' for s in cv_scores]}")
    print(f"   Mean accuracy: {cv_scores.mean():.3f}")
    print(f"   Std deviation: {cv_scores.std():.3f}")
    print(f"   Time: {time_cv:.3f}s")
    print(f"   ✅ More reliable estimate!")
    
    # Comparison
    print("\n📊 COMPARISON:")
    print(f"   Time ratio (CV/Single): {time_cv/time_single:.1f}x")
    print(f"   Data used for training: Single={len(X_train)}, CV={len(X)*0.8:.0f} avg")
    print(f"   Reliability: {'CV is more reliable' if n_samples < 10000 else 'Both reasonable'}")
```

#### When to Use Each Approach:

```python
print("\n\n" + "="*60)
print("DECISION GUIDE: WHICH APPROACH TO USE?")
print("="*60)

scenarios = {
    '✅ Use Train-Test Split When:': [
        {
            'scenario': 'Very Large Dataset (>100,000 samples)',
            'reason': 'Computational cost of CV too high',
            'example': 'Training deep learning model on ImageNet',
            'recommendation': '80-20 split is sufficient'
        },
        {
            'scenario': 'Time-Series Data',
            'reason': 'Cannot randomly shuffle data',
            'example': 'Stock price prediction, sensor data',
            'recommendation': 'Use temporal split (train on past, test on future)'
        },
        {
            'scenario': 'Production-Like Evaluation',
            'reason': 'Simulate real deployment scenario',
            'example': 'Testing model before deployment',
            'recommendation': 'Single train-test split mirrors production'
        },
        {
            'scenario': 'Computational Resources Limited',
            'reason': 'CV trains model k times (k-fold)',
            'example': 'Training expensive models (large neural nets)',
            'recommendation': 'Single split much faster'
        }
    ],
    '✅ Use Cross-Validation When:': [
        {
            'scenario': 'Small to Medium Dataset (<10,000 samples)',
            'reason': 'Need robust performance estimate',
            'example': 'Medical study with 500 patients',
            'recommendation': '5 or 10-fold CV'
        },
        {
            'scenario': 'Hyperparameter Tuning',
            'reason': 'Need reliable comparison of configurations',
            'example': 'GridSearchCV for model selection',
            'recommendation': 'Nested CV or CV + holdout test'
        },
        {
            'scenario': 'Imbalanced Dataset',
            'reason': 'Ensure all folds have minority class',
            'example': 'Fraud detection (99% legitimate)',
            'recommendation': 'Stratified K-Fold CV'
        },
        {
            'scenario': 'Model Comparison',
            'reason': 'Fair comparison across different algorithms',
            'example': 'Choosing between RF, XGBoost, SVM',
            'recommendation': 'Use same CV folds for all models'
        }
    ]
}

for category, cases in scenarios.items():
    print(f"\n{category}")
    print("=" * 60)
    for i, case in enumerate(cases, 1):
        print(f"\n{i}. {case['scenario']}")
        print(f"   Reason: {case['reason']}")
        print(f"   Example: {case['example']}")
        print(f"   Recommendation: {case['recommendation']}")
```

#### Hybrid Approach:

```python
print("\n\n" + "="*60)
print("HYBRID APPROACH: BEST OF BOTH WORLDS")
print("="*60)

# For medium datasets: Use both approaches
X, y = make_classification(n_samples=5000, n_features=20, random_state=42)

# Step 1: Hold out test set
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n📊 Hybrid Strategy:")
print(f"   1. Hold out 20% as final test set: {len(X_test)} samples")
print(f"   2. Use remaining 80% for cross-validation: {len(X_train_val)} samples")

# Step 2: Use CV for model selection and hyperparameter tuning
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [5, 10, 15],
    'n_estimators': [50, 100, 150]
}

print(f"\n   3. Hyperparameter tuning with GridSearchCV (5-fold)")

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train_val, y_train_val)

print(f"      Best parameters: {grid_search.best_params_}")
print(f"      Best CV score: {grid_search.best_score_:.3f}")

# Step 3: Final evaluation on held-out test set
test_score = grid_search.score(X_test, y_test)
print(f"\n   4. Final test set evaluation: {test_score:.3f}")
print(f"      ✅ This is our unbiased performance estimate!")

print("\n💡 Benefits of Hybrid Approach:")
print("   ✅ Reliable model selection (CV on 80%)")
print("   ✅ Unbiased final estimate (holdout 20%)")
print("   ✅ Best of both worlds!")
```

#### Practical Workflow:

```python
print("\n\n" + "="*60)
print("COMPLETE WORKFLOW EXAMPLE")
print("="*60)

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Complete ML workflow
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

print("\n🔄 Step-by-step workflow:\n")

# Step 1: Initial train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"1️⃣ Split data: {len(X_train)} train, {len(X_test)} test")

# Step 2: Create pipeline (preprocessing + model)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])
print(f"2️⃣ Create pipeline with preprocessing")

# Step 3: Hyperparameter tuning with cross-validation
param_grid = {
    'classifier__max_depth': [5, 10, None],
    'classifier__n_estimators': [50, 100]
}

grid_search = GridSearchCV(
    pipeline, param_grid, cv=5, n_jobs=-1
)

print(f"3️⃣ Hyperparameter tuning with 5-fold CV on training set")
grid_search.fit(X_train, y_train)
print(f"   Best params: {grid_search.best_params_}")

# Step 4: Evaluate on test set
test_score = grid_search.score(X_test, y_test)
print(f"4️⃣ Final evaluation on test set: {test_score:.3f}")

# Step 5: Analyze CV results
cv_results = grid_search.cv_results_
print(f"\n5️⃣ Cross-validation insights:")
print(f"   Number of configurations tested: {len(cv_results['params'])}")
print(f"   Best CV score: {grid_search.best_score_:.3f}")
print(f"   Test score: {test_score:.3f}")
print(f"   Difference: {abs(grid_search.best_score_ - test_score):.3f}")

if abs(grid_search.best_score_ - test_score) > 0.05:
    print(f"   ⚠️ Large difference suggests overfitting to CV folds")
else:
    print(f"   ✅ Good agreement between CV and test scores")
```

#### Time-Series Special Case:

```python
print("\n\n" + "="*60)
print("SPECIAL CASE: TIME-SERIES DATA")
print("="*60)

from sklearn.model_selection import TimeSeriesSplit

# Generate time-series data
n_samples = 1000
X_ts = np.random.randn(n_samples, 10)
y_ts = np.random.randint(0, 2, n_samples)

print("\n⏰ Time-series requires special handling:")
print("   ❌ Cannot use regular K-Fold (would leak future into past)")
print("   ✅ Use TimeSeriesSplit or simple temporal split\n")

# Option 1: TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
print("Option 1: TimeSeriesSplit (for cross-validation)")
for i, (train_idx, test_idx) in enumerate(tscv.split(X_ts), 1):
    print(f"   Fold {i}: Train={len(train_idx)}, Test={len(test_idx)}")
    print(f"      Train indices: 0 to {train_idx[-1]}")
    print(f"      Test indices: {test_idx[0]} to {test_idx[-1]}")

# Option 2: Simple temporal split
split_point = int(0.8 * n_samples)
X_train_ts, X_test_ts = X_ts[:split_point], X_ts[split_point:]
y_train_ts, y_test_ts = y_ts[:split_point], y_ts[split_point:]

print(f"\nOption 2: Simple temporal split")
print(f"   Train: samples 0 to {split_point-1}")
print(f"   Test: samples {split_point} to {n_samples-1}")
print(f"   ✅ Simulates real-world scenario (predict future from past)")
```

**Summary Table:**

| Aspect | Train-Test Split | Cross-Validation |
|--------|------------------|------------------|
| **Speed** | Fast (1x model training) | Slow (k× model training) |
| **Data Usage** | 80% for training | ~80-90% for training (avg) |
| **Reliability** | Lower (single split) | Higher (averaged over k splits) |
| **Variance** | Can be high | Lower (more stable) |
| **Best For** | Large datasets, production testing | Small-medium datasets, model selection |
| **Computational Cost** | Low | k times higher |
| **Typical Use** | Final evaluation | Hyperparameter tuning |

**Key Takeaways:**
- Use cross-validation for small-medium datasets (<10,000 samples)
- Use train-test split for very large datasets or time-series
- Hybrid approach: CV for tuning + holdout test for final evaluation
- Always use stratified splits for imbalanced data
- Time-series data requires temporal splits (no shuffling)

---

*Due to length constraints, I'll create a separate file continuation. Shall I continue with the remaining questions 3-10 for Day 7?*

---

**Previous:** [Day 06 - Supervised Learning (Intermediate)](../Day-06-Supervised-Learning-Intermediate/README.md) | **Next:** [Day 08 - Ensemble Methods](../Day-08-Ensemble-Methods/README.md)
