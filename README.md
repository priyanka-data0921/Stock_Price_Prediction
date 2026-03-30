# Stock_Price_Prediction
Stock_Price_Prediction Apple Inc. (AAPL) Time period: January 1, 2020 — January 1, 2023
# 📈 Stock Price Prediction with Large Language Models (LLM)
### Columbia University — Prompt Engineering & Programming with OpenAI

---

## 🧠 Project Overview

This project explores how **Large Language Models (LLMs)** can perform complex data science and machine learning tasks — without requiring traditional coding knowledge. Instead of writing code manually, I used **Google's Gemini AI** inside Google Colab to load data, engineer features, build models, and discuss trading strategies.

The key skill demonstrated here is **Prompt Engineering** — the ability to give clear, precise instructions to an AI to produce accurate and useful results.

> **Tools used:** Google Colab, Gemini AI, Python (yfinance, pandas, scikit-learn, ta)

---

## 🎯 What This Project Does

Given historical Apple stock data, this project:
1. **Explores and summarizes** the data using key statistics
2. **Engineers new features** to help predict future price movements
3. **Builds two machine learning models** to predict stock price direction
4. **Evaluates model performance** on unseen 2023–2024 data
5. **Discusses real-world trading strategies** based on model results

---

## 📋 Step-by-Step Breakdown

---

### Step 1 — Data Exploration 🔍

**What I did:**
Loaded 3 years of Apple stock price history from Yahoo Finance and summarized key statistics.

**Key Findings:**

| Statistic | Value | Plain English Meaning |
|---|---|---|
| Trading Days | 756 | Stock market is closed on weekends & holidays |
| Average Closing Price | $127.80 | Average price Apple stock closed at over 3 years |
| Minimum Price | $54.45 | Lowest price — happened during COVID crash (March 2020) |
| Maximum Price | $178.88 | Highest price — peak of Apple's growth (late 2021) |
| Standard Deviation | $30.46 | Price bounced around a lot — very volatile stock |
| Average Daily Volume | 112 million shares | 112 million Apple shares bought/sold every day on average |

**What I learned:**
Apple stock went on a massive rollercoaster from 2020–2023. It crashed during COVID, then soared to record highs, making it a volatile but ultimately upward-trending stock over this period. The standard deviation of $30.46 confirms significant price swings throughout the period.

**Prompt used:**
```
Load historical stock price data for Apple Inc. (AAPL) from 2020-01-01 to 2023-01-01
using Yahoo Finance and summarize its key statistics.
```

---

### Step 2 — Feature Engineering 🔧

**What I did:**
Created new data columns (features) from the raw stock data to give the model more useful information to learn from. Think of these as extra "clues" to help predict whether the price will go up or down.

**Features Created:**

| Feature | What It Means |
|---|---|
| SMA 5, 10, 20, 50, 200 | Average closing price over last 5/10/20/50/200 days — shows trends |
| RSI (Relative Strength Index) | Measures if a stock is overbought or oversold — popular trading signal |
| Williams %R | Another momentum indicator showing price relative to recent highs/lows |
| Bollinger Bands (Upper, Middle, Lower, Width) | Shows price volatility range — like a price "envelope" |
| Daily Return | How much the price changed from yesterday in percentage |
| Lagged Returns (5-day, 20-day) | Returns from 5 and 20 days ago — captures momentum |
| Quarter, Day of Year, Week of Year | Time-based features — captures seasonal patterns |

**Why this matters:**
Raw stock prices alone don't tell the full story. By adding moving averages, momentum indicators, and time features, we give the model much richer information to find patterns from.

**Prompt used:**
```
Using the AAPL data already loaded, add these features:
Moving averages (SMA 5, 10, 20, 50, 200), RSI, Williams %R,
Bollinger Bands, lagged returns, and time-based features.
Show the last 5 rows of the updated dataframe.
```

---

### Step 3 — Model Building 🤖

Two different machine learning models were built and compared.

---

#### Model 3a — Linear Regression (Predicting Exact Price)

**What it does:**
Tries to predict the **exact closing price** for tomorrow by finding a straight-line pattern in past data.

**Results:**

| Metric | Value | What It Means |
|---|---|---|
| MSE (Mean Squared Error) | 859 | Predictions were off by ~$29 per day on average |
| R² Score | -0.009 | Worse than just guessing the average price every day! |

**Conclusion:**
Linear Regression **failed** at predicting exact stock prices. This makes sense — stock prices are far too complex and influenced by too many unpredictable factors (news, world events, emotions) to follow a simple straight line. This is a valuable finding!

---

#### Model 3b — Random Forest Classification (Predicting UP or DOWN)

**What it does:**
Instead of predicting the exact price, this model asks a simpler question: **Will Apple stock go UP or DOWN tomorrow?**

It uses 100 different "decision trees" that each vote on the answer, and the majority vote wins — like asking 100 friends for advice instead of just one.

**Results:**

| Metric | Value | What It Means |
|---|---|---|
| Accuracy | 71.82% | Got the direction right 72 out of every 100 days |

**Most Important Features (what the model relied on most):**

| Feature | Importance | Why It Matters |
|---|---|---|
| Closing Price | 30.87% | Most recent price is the strongest signal |
| 5-day Moving Average | 12.42% | Short-term trend is very predictive |
| Volume | 9.82% | How many shares traded matters |
| Low Price | 9.80% | Daily low captures downward pressure |
| Open Price | 9.00% | Morning price sets the tone for the day |

**Conclusion:**
Random Forest significantly outperformed Linear Regression. Predicting direction (up/down) is more achievable than predicting exact prices, and the 71.82% accuracy is a solid result for stock market prediction.

---

### Step 4 — Model Evaluation on New Data 📊

**What I did:**
Tested the Random Forest model on completely new data it had never seen — Apple stock from **2023 to 2024**.

**Results:**

| Metric | Training Period (2020–2023) | Test Period (2023–2024) |
|---|---|---|
| Accuracy | 71.82% | 60.85% |

**What this tells us:**
The accuracy dropped from 71% to 61% on the new data. This is called **overfitting** — the model learned the patterns of 2020–2023 very well, but struggled to generalize to a new time period. This is a common and important challenge in machine learning.

The model was much better at predicting days when the price would NOT increase (class 0) than days when it would increase (class 1).

---

### Step 5 — Trading Strategy Discussion 💼

**Based on the model's performance, three potential strategies were identified:**

#### 🟢 Conservative Strategy
- **When to act:** Only buy when the model predicts a price increase with high confidence
- **Logic:** Be selective — only trade on the model's strongest signals
- **Risk:** Miss some opportunities but reduce false positives

#### 🔴 Contrarian Strategy
- **When to act:** Stay out of the market when the model predicts no price increase
- **Logic:** The model is better at identifying "no increase" days, so trust those predictions
- **Risk:** May miss upward movements during uncertain periods

#### 🟡 Hybrid Strategy
- **When to act:** Set confidence thresholds — only buy when probability is very high, only sell when model is confident about a decline
- **Logic:** Get the best of both strategies by being flexible
- **Risk:** Requires careful monitoring and threshold adjustment

**Important Disclaimer:**
No model should be solely relied upon for real trading decisions. Stock markets are influenced by countless unpredictable factors. These strategies are for educational purposes only.

---

## 📌 Key Takeaways

| Concept | What I Learned |
|---|---|
| **Prompt Engineering** | Clear, specific prompts produce better AI-generated code |
| **Data Exploration** | Always understand your data before modeling |
| **Feature Engineering** | Better features lead to better models |
| **Model Selection** | Different problems need different models — classification beat regression here |
| **Model Evaluation** | Always test on new unseen data to check real-world performance |
| **Critical Thinking** | A 60% accurate model is interesting, but not reliable enough for real trading |

---

## 🛠️ Technologies Used

| Tool | Purpose |
|---|---|
| **Google Colab** | Cloud-based Python notebook environment |
| **Gemini AI** | LLM used to generate all code via prompts |
| **yfinance** | Downloaded historical Apple stock data from Yahoo Finance |
| **pandas** | Data manipulation and analysis |
| **ta (Technical Analysis)** | Calculated RSI, Williams %R, Bollinger Bands |
| **scikit-learn** | Built Linear Regression and Random Forest models |

---


