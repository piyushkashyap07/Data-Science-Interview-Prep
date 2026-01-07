# Day 35 - MLOps Fundamentals
 
 **Topics Covered:** Model Serialization (Pickle/Joblib), API Serving (Flask/FastAPI), Docker Basics, Model Drift (Data/Concept), CI/CD for ML.
 
 ---
 
 ## Question 1: What is MLOps?
 
 **Topic:** Concept
 **Difficulty:** Basic
 
 ### Question
 How does MLOps differ from distinct DevOps?
 
 ### Answer
 
 - **DevOps:** Focuses on Code. (Is the code bug-free? Does it build? Is the server up?)
 - **MLOps:** Focuses on **Code + Data + Model**.
    - Code might be perfect, but if the Data changes (Drift), the system breaks.
    - Testing involves "Accuracy" checks, not just "Integration" tests.
    - Requires tracking Experiments (Weights & Biases, MLflow) which DevOps doesn't do.
 
 ---
 
 ## Question 2: Serialization (Pickle vs ONNX)
 
 **Topic:** Deployment
 **Difficulty:** Intermediate
 
 ### Question
 You trained a model in PyTorch. How do you save it for production? Compare Pickle vs ONNX.
 
 ### Answer
 
 1. **Pickle/Joblib (Python native):**
    - `joblib.dump(model, 'model.pkl')`.
    - **Pros:** Easy. **Cons:** Python-only. Slow. Security risk (can execute arbitrary code).
 2. **ONNX (Open Neural Network Exchange):**
    - Universal standard (Works in C++, C#, Java, Python).
    - **Pros:** Extremely fast runtime (ONNX Runtime). Cross-platform.
    - **Best Practice:** Convert to ONNX for production deployment.
 
 ---
 
 ## Question 3: Flask vs FastAPI
 
 **Topic:** Serving
 **Difficulty:** Intermediate
 
 ### Question
 Why is FastAPI preferred over Flask for ML Inference APIs?
 
 ### Answer
 
 1. **Speed:** FastAPI is asynchronous (ASGI) and extremely fast (built on Starlette).
 2. **Type Validation:** Uses Pydantic to validate input JSON automatically.
    - If user sends "Age": "Ten", Flask crashes inside your code. FastAPI rejects it with a clear 422 error automatically.
 3. **Documentation:** Auto-generates Swagger (OpenAPI) UI (`/docs`).
 
 ---
 
 ## Question 4: Docker
 
 **Topic:** Infrastructure
 **Difficulty:** Basic
 
 ### Question
 Why simple "It works on my machine" is not enough? What problem does Docker solve for ML?
 
 ### Answer
 
 **Problem:** Dependency Hell.
 - You have Python 3.8, PyTorch 1.7, and CUDA 10.1.
 - Server has Python 3.6, PyTorch 1.9, and CUDA 11.
 - **Result:** Code crashes.
 
 **Docker Solution:**
 - Packages Code + Dependencies + OS + Drivers into a single lightweight **Container**.
 - A container runs exactly the same everywhere (Laptop, Cloud, On-prem).
 
 ---
 
 ## Question 5: Model Drift (Data Drift)
 
 **Topic:** Monitoring
 **Difficulty:** Intermediate
 
 ### Question
 You deployed a Credit Card Fraud model in 2019. In 2020, accuracy dropped 20%. Why?
 
 ### Answer
 
 **Data Drift (Covariate Shift):**
 - The input distribution $P(X)$ changed.
 - **Example:** Covid happened. consumer spending behavior changed drastically (More online shopping).
 - The model trained on 2019 patterns doesn't recognize 2020 patterns.
 - **Fix:** Re-train model on recent data.
 
 ---
 
 ## Question 6: Concept Drift
 
 **Topic:** Monitoring
 **Difficulty:** Intermediate
 
 ### Question
 Difference between Data Drift and Concept Drift?
 
 ### Answer
 
 - **Data Drift:** Input features change. (Users are younger now). The *relationship* between Age and Fraud might still be same.
 - **Concept Drift:** The **relationship** changes ($P(Y|X)$ changes).
    - Example: "Spam" definition changes. "Crypto" emails were safe 5 years ago, now they are mostly spam. The meaning of the feature "Crypto" has flipped.
 
 ---
 
 ## Question 7: CI/CD for ML (CT)
 
 **Topic:** Pipeline
 **Difficulty:** Advanced
 
 ### Question
 Google calls it "CT" (Continuous Training). What triggers a CT pipeline?
 
 ### Answer
 
 In standard DevOps, a "Git Push" triggers CD. In ML, **two things** trigger pipeline:
 1. **Code Change:** New algorithm pushed to Git.
 2. **Performance Drop:** Monitoring system detects accuracy < 90%.
    - **Action:** Automatically trigger the "Training Pipeline".
    - Fetch new data -> Preprocess -> Train -> Evaluate -> If better, Deploy.
 
 ---
 
 ## Question 8: A/B Testing vs Canary
 
 **Topic:** Deployment
 **Difficulty:** Intermediate
 
 ### Question
 How do you safely roll out a new model Model B to replace Model A?
 
 ### Answer
 
 1. **Canary Deployment:**
    - Give Model B to 1% of users. Monitor errors.
    - If safe, increase to 10%, 50%, 100%. (Safety focus).
 2. **A/B Testing:**
    - Give Model A to 50% and Model B to 50%.
    - Measure distinct business metric (e.g., Conversion Rate).
    - If B > A statistically, replace A. (Profit focus).
 
 ---
 
 ## Question 9: Feature Store
 
 **Topic:** Infrastructure
 **Difficulty:** Advanced
 
 ### Question
 What is a Feature Store (like Feast)? Why do big companies need it?
 
 ### Answer
 
 **Problem:** Training-Serving Skew.
 - Data Scientist writes SQL: `avg(clicks) over last 30 days`.
 - Engineer writes Java for app: `avg(clicks)`.
 - The implementations differ slightly -> Bug.
 
 **Feature Store:** Centralized database of features.
 - **Training:** `get_historical_features(entity, timestamp)`.
 - **Serving:** `get_online_features(entity)`.
 - Guarantees the exact same feature logic for training and inference.
 
 ---
 
 ## Question 10: Dockerfile Example
 
 **Topic:** Implementation
 **Difficulty:** Basic
 
 ### Question
 Write a simple Dockerfile for a generic Python ML app.
 
 ### Answer
 
 ```dockerfile
 # 1. Base Image
 FROM python:3.9-slim
 
 # 2. Set Working Dir
 WORKDIR /app
 
 # 3. Copy Requirements
 COPY requirements.txt .
 
 # 4. Install Dependencies
 RUN pip install --no-cache-dir -r requirements.txt
 
 # 5. Copy Code
 COPY . .
 
 # 6. Command
 CMD ["python", "app.py"]
 ```
 
 ---
 
 ## Key Takeaways
 
 - **MLOps** bridges the gap between Jupyter Notebook and Production.
 - **Docker** is non-negotiable for reproducibility.
 - **FastAPI** is the modern standard for Python APIs.
 - **Drift** is inevitable; Monitoring is essential.
 - **ONNX** allows you to break free from Python for inference.
 
 **Next:** [Week 6 Coming Soon](../README.md)
