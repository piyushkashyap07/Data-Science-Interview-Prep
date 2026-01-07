# Day 29 - Time Series Basics
 
 **Topics Covered:** Trend, Seasonality, Stationarity, ADF Test, White Noise, Autocorrelation (ACF/PACF), Moving Averages.
 
 ---
 
 ## Question 1: What makes Time Series unique?
 
 **Topic:** Concept
 **Difficulty:** Basic
 
 ### Question
 How does Time Series data differ from standard Tabular data used in Regression?
 
 ### Answer
 
 **Independence Assumption Violated:**
 - In standard regression, we assume rows are independent ($y_i$ does not depend on $y_{i-1}$).
 - In Time Series, **Time Dependence** is crucial. Today's stock price depends heavily on yesterday's price.
 - Order matters. You cannot shuffle the data during training.
 
 ---
 
 ## Question 2: Components of Time Series
 
 **Topic:** Concept
 **Difficulty:** Basic
 
 ### Question
 Decompose a time series into its four main components.
 
 ### Answer
 
 $$ Y_t = T_t + S_t + C_t + \epsilon_t $$
 
 1. **Trend ($T_t$):** Long-term movement (Increasing/Decreasing/Flat).
 2. **Seasonality ($S_t$):** Repeating pattern with fixed period (Weekly, Yearly). e.g., Ice cream sales in summer.
 3. **Cyclical ($C_t$):** Oscillations without fixed period (Economic cycles, Boom/Bust).
 4. **Noise/Residual ($\epsilon_t$):** Random, unpredictable variation.
 
 ---
 
 ## Question 3: Stationarity
 
 **Topic:** Concept
 **Difficulty:** Intermediate
 
 ### Question
 What is "Stationarity"? Why do ARIMA models require it?
 
 ### Answer
 
 **Definition:** A time series is stationary if its statistical properties do not change over time.
 1. Constant **Mean** (No Trend).
 2. Constant **Variance** (Homoscedasticity).
 3. Constant **Autocovariance** (Relationship between $t$ and $t-k$ depends only on $k$, not time $t$).
 
 **Why needed:** Models like ARIMA assume that the "rules" of the series are constant. If the mean is increasing (Trend), the model trained on the past will fail on the future.
 
 ---
 
 ## Question 4: Augmented Dickey-Fuller (ADF) Test
 
 **Topic:** Statistics
 **Difficulty:** Intermediate
 
 ### Question
 How do we mathematically test for stationarity? Interpret the p-value.
 
 ### Answer
 
 **ADF Test:** A hypothesis test for the presence of a "Unit Root" (Non-stationarity).
 - **Null Hypothesis ($H_0$):** Series is Non-Stationary.
 - **Alternate Hypothesis ($H_1$):** Series is Stationary.
 
 **Interpretation:**
 - **p-value < 0.05:** Reject Null. **Series is Stationary.**
 - **p-value > 0.05:** Fail to reject. Series is Non-Stationary (Needs differencing).
 
 ---
 
 ## Question 5: Differencing ($d$)
 
 **Topic:** Technique
 **Difficulty:** Basic
 
 ### Question
 Your series has a strong upward trend. How do you make it stationary?
 
 ### Answer
 
 **Differencing:** Subtract the previous value from the current value.
 $$ Y'_t = Y_t - Y_{t-1} $$
 - This removes the trend (linear growth becomes a constant mean).
 - If trend is exponential, apply **Log Transform** first, then Difference.
 
 ---
 
 ## Question 6: ACF vs PACF
 
 **Topic:** Analysis
 **Difficulty:** Advanced
 
 ### Question
 Differentiate between Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF). Which helps determine $p$ in AR(p)?
 
 ### Answer
 
 - **ACF:** Correlation between $Y_t$ and $Y_{t-k}$ (Direct + Indirect effects).
    - Includes influence carried through intermediate lags ($t-1 \to t-2 \to ... \to t-k$).
 - **PACF:** Correlation between $Y_t$ and $Y_{t-k}$ **after removing** the effects of intermediate lags.
    - Captures only the "direct" impact.
 
 **Determining $p$ (AutoRegressive term):**
 - Look at **PACF**. If it cuts off after lag $k$, use AR($k$).
 
 ---
 
 ## Question 7: Moving Averages
 
 **Topic:** Concept
 **Difficulty:** Basic
 
 ### Question
 What is a Simple Moving Average (SMA)? How does "Window Size" affect it?
 
 ### Answer
 
 **SMA:** The average of the last $N$ data points.
 $$ SMA_t = \frac{1}{N} \sum_{i=0}^{N-1} Y_{t-i} $$
 
 **Window Size ($N$):**
 - **Small $N$ (e.g., 5):** Reacts fast to changes. Noisy.
 - **Large $N$ (e.g., 200):** Very smooth. Reacts slowly (Lag). Used to identify long-term trends (Golden Cross).
 
 ---
 
 ## Question 8: White Noise
 
 **Topic:** Concept
 **Difficulty:** Intermediate
 
 ### Question
 If your model's residuals (errors) look like "White Noise", is that good or bad?
 
 ### Answer
 
 **Good.**
 - White Noise = Sequence of random numbers with mean 0, constant variance, and **no autocorrelation**.
 - It implies that your model has extracted **all** meaningful patterns (Trend, Seasonality, Correlation) from the data.
 - Only pure randomness is left.
 
 ---
 
 ## Question 9: ARIMA Model
 
 **Topic:** Modeling
 **Difficulty:** Intermediate
 
 ### Question
 Explain the three parameters of ARIMA(p, d, q).
 
 ### Answer
 
 - **AR (p): AutoRegression.** The current value depends on previous $p$ values. ($Y_t$ regressed on $Y_{t-1}...Y_{t-p}$).
 - **I (d): Integrated.** The number of times we differenced the raw data to make it stationary.
 - **MA (q): Moving Average.** The current value depends on previous $q$ forecast errors (shocks). ($Y_t$ regressed on $\epsilon_{t-1}...\epsilon_{t-q}$).
 
 ---
 
 ## Question 10: Python Implementation
 
 **Topic:** Implementation
 **Difficulty:** Basic
 
 ### Question
 How do you perform the ADF test in Python?
 
 ### Answer
 
 ```python
 from statsmodels.tsa.stattools import adfuller
 import pandas as pd
 
 data = [1, 2, 3, 4, 5, 6, 7] # Non-stationary
 
 result = adfuller(data)
 print(f'ADF Metric: {result[0]}')
 print(f'p-value: {result[1]}')
 
 if result[1] < 0.05:
     print("Stationary")
 else:
     print("Non-Stationary")
 ```
 
 ---
 
 ## Key Takeaways
 
 - **Stationarity** is the prerequisite for statistical forecasting.
 - **Trend & Seasonality** must be removed/accounted for.
 - **ADF Test:** The litmus test for stationarity.
 - **ARIMA:** The classic linear forecasting workhorse.
 - **White Noise Residuals:** The sign of a good model.
 
 **Next:** [Day 30 - Advanced Time Series](../Day-30/README.md)
