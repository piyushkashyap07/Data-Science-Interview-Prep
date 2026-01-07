# Day 30 - Advanced Time Series Forecasting
 
 **Topics Covered:** SARIMA, Facebook Prophet, LSTM for Time Series, Lag Features, Windowing, Backtesting (TimeSeriesSplit), Evaluation Metrics (MAPE, RMSE).
 
 ---
 
 ## Question 1: SARIMA vs ARIMA
 
 **Topic:** Modeling
 **Difficulty:** Intermediate
 
 ### Question
 What is the "S" in SARIMA, and when do you need it?
 
 ### Answer
 
 **SARIMA:** **S**easonal ARIMA.
 - ARIMA works well for trends but fails with seasonality (e.g., sales spiking every December).
 - SARIMA adds 4 new parameters $(P, D, Q, s)$.
    - $P, D, Q$: AR, I, MA parts for the seasonal component.
    - $s$: The length of the season (e.g., 12 for monthly data, 7 for daily).
 
 ---
 
 ## Question 2: Facebook Prophet
 
 **Topic:** Algorithm
 **Difficulty:** Intermediate
 
 ### Question
 Why was Facebook Prophet designed? How does it differ from ARIMA?
 
 ### Answer
 
 **Design Philosophy:**
 - Traditional ARIMA is hard to tune and handles missing data/outliers poorly.
 - Prophet is an **Additive Regressor model** ($y(t) = g(t) + s(t) + h(t) + \epsilon_t$).
    - $g(t)$: Trend (Piecewise linear/logistic).
    - $s(t)$: Seasonality (Fourier series).
    - $h(t)$: Holidays.
 
 **Advantages:**
 - Handles missing data and outliers robustly.
 - Fast.
 - Intuitive parameters ("add holiday").
 - Requires less statistical expertise than ARIMA.
 
 ---
 
 ## Question 3: Preparing Data for LSTM (Sliding Window)
 
 **Topic:** Data Processing
 **Difficulty:** Advanced
 
 ### Question
 LSTMs require 3D input $(Samples, Timesteps, Features)$. How do you convert a 1D stock price array into this format?
 
 ### Answer
 
 **Sliding Window Technique:**
 - **Concept:** Use the last $T$ days to predict day $T+1$.
 - **Raw:** `[10, 20, 30, 40, 50]`
 - **Window ($T=3$):**
    - $X_1$: `[10, 20, 30]`, $y_1$: `40`
    - $X_2$: `[20, 30, 40]`, $y_2$: `50`
 - **Final Shape:** `(2, 3, 1)` -> (2 samples, 3 lookback steps, 1 feature).
 
 ---
 
 ## Question 4: Lag Features
 
 **Topic:** Feature Engineering
 **Difficulty:** Basic
 
 ### Question
 If you want to use XGBoost for time series, you can't feed it "Date". What features do you create?
 
 ### Answer
 
 XGBoost is a tabular model. We must convert time context into columns:
 1. **Lag Features:** $y_{t-1}, y_{t-7}, y_{t-30}$ (Yesterday's value, Last week's value).
 2. **Rolling Statistics:** Rolling Mean (7-day), Rolling Std Dev.
 3. **Time Components:** DayOfWeek, Month, IsHoliday.
 
 ---
 
 ## Question 5: Evaluation Metrics (RMSE vs MAPE)
 
 **Topic:** Evaluation
 **Difficulty:** Intermediate
 
 ### Question
 When should you use MAPE (Mean Absolute Percentage Error) over RMSE (Root Mean Squared Error)?
 
 ### Answer
 
 - **RMSE:** scale-dependent (e.g., "Error is $50").
    - Good for optimization (differentiable).
    - Hard to compare across datasets (Is $50 good?).
 
 - **MAPE:** scale-independent (e.g., "Error is 5%").
    - $$ \frac{1}{n} \sum \left| \frac{Actual - Forecast}{Actual} \right| $$
    - **Use case:** Business reporting. "Our forecast is 95% accurate" is easier to understand than "RMSE is 415".
    - **Flaw:** Explodes if Actual is close to 0.
 
 ---
 
 ## Question 6: Time Series Cross-Validation
 
 **Topic:** Evaluation
 **Difficulty:** Intermediate
 
 ### Question
 Why can't you use standard K-Fold Cross-Validation on Time Series?
 
 ### Answer
 
 **Data Leakage:** Standard K-Fold shuffles data.
 - Training on "Future" features (e.g., Dec 2023) to predict "Past" targets (e.g., Jan 2023) is cheating.
 
 **Solution: TimeSeriesSplit (Walk-Forward Validation)**
 - Fold 1: Train [Jan], Test [Feb]
 - Fold 2: Train [Jan, Feb], Test [Mar]
 - Fold 3: Train [Jan, Feb, Mar], Test [Apr]
 - Always respects temporal order.
 
 ---
 
 ## Question 7: Direct vs Recursive Forecasting
 
 **Topic:** Strategy
 **Difficulty:** Advanced
 
 ### Question
 You need to predict the next 7 days. Explain the Recursive vs Direct strategy.
 
 ### Answer
 
 1. **Recursive (Autoregressive):**
    - Train model to predict $t+1$.
    - Predict Day 1.
    - **Feed prediction back as input** to predict Day 2.
    - **Pros:** One model. **Cons:** Errors accumulate (Day 7 is garbage).
 
 2. **Direct:**
    - Train 7 separate models.
    - Model 1 predicts $t+1$. Model 7 predicts $t+7$.
    - **Pros:** No error compounding. **Cons:** Computationally expensive, ignores dependency between output steps.
 
 ---
 
 ## Question 8: Univariate vs Multivariate
 
 **Topic:** Concept
 **Difficulty:** Basic
 
 ### Question
 Difference between Univariate and Multivariate Time Series?
 
 ### Answer
 
 - **Univariate:** Only the target variable history is available.
    - *Predict "Sales" using only past "Sales".* (ARIMA).
 - **Multivariate:** External features (Exogenous variables) are available.
    - *Predict "Sales" using past "Sales" + "Advertising Budget" + "Temperature".* (VAR, LSTM, Prophet).
 
 ---
 
 ## Question 9: LSTM Implementation (Keras)
 
 **Topic:** Implementation
 **Difficulty:** Advanced
 
 ### Question
 Write a Keras LSTM model structure for forecasting.
 
 ### Answer
 
 ```python
 from tensorflow.keras.models import Sequential
 from tensorflow.keras.layers import LSTM, Dense
 
 # Input Shape: (Samples, Timesteps, Features)
 model = Sequential()
 
 # LSTM Layer
 model.add(LSTM(50, activation='relu', input_shape=(30, 1)))
 
 # Output Layer (Predicting 1 value)
 model.add(Dense(1))
 
 model.compile(optimizer='adam', loss='mse')
 # model.fit(X_train, y_train...)
 ```
 
 ---
 
 ## Question 10: Prophet Implementation
 
 **Topic:** Implementation
 **Difficulty:** Basic
 
 ### Question
 Use `prophet` to forecast 365 days.
 
 ### Answer
 
 ```python
 from prophet import Prophet
 import pandas as pd
 
 # 1. Prepare Data (Must be columns 'ds' and 'y')
 df = pd.DataFrame({'ds': dates, 'y': values})
 
 # 2. Train
 m = Prophet()
 m.fit(df)
 
 # 3. Make Future Dataframe
 future = m.make_future_dataframe(periods=365)
 
 # 4. Predict
 forecast = m.predict(future)
 m.plot(forecast)
 ```
 
 ---
 
 ## Key Takeaways
 
 - **SARIMA:** ARIMA + Seasonality.
 - **Prophet:** Great baseline, handles messy data/holidays well.
 - **Cross-Validation:** Must use sliding/expanding window (No shuffling!).
 - **LSTM:** Good for complex non-linear patterns.
 - **XGBoost:** Powerful if you engineer the right Lag features.
 
 **Next:** [Day 31 - Recommender Systems Basics](../Day-31/README.md)
