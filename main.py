import nest_asyncio
nest_asyncio.apply()

import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pandas.tseries.offsets import BDay

# For machine learning:
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error

import math
from math import log, sqrt, exp
from scipy.stats import norm

import openai
import os
import json
import time
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
openai.api_key = os.environ.get("OPENAI_API_KEY")

st.set_page_config(page_title="Gap Fill Backtesting Dashboard", layout="wide")
st.title("Gap Fill Backtesting Dashboard with Consolidated Metrics, ML, and Automated 0DTE Strategy (Using Uploaded Greeks)")

st.markdown("""
This dashboard allows you to select a lookback period and an asset (Apple, Amazon, SPY, Google, Tesla, or Bitcoin). 
It downloads historical data, calculates daily gaps and other metrics, and builds an ML model to predict next day metrics.  
Finally, you can upload a CSV of your 0DTE option chain (with pre-computed Greeks) and your custom assistant will return an optimal 
option order combination as a JSON object (which is then displayed as a table). If no viable opportunity exists, a fallback JSON object with all values as "n/a" is returned.
""")

# -------------------------------
# Asset Selection
# -------------------------------
asset_options = {
    "Apple": "AAPL",
    "Amazon": "AMZN",
    "SPY": "SPY",
    "Google": "GOOG",
    "Tesla": "TSLA",
    "Bitcoin": "BTC-USD"
}
asset_choice = st.selectbox("Choose Asset", list(asset_options.keys()), index=2)
ticker_symbol = asset_options[asset_choice]
st.write(f"**Selected Asset:** {asset_choice} ({ticker_symbol})")

# -------------------------------
# 1. User Input for Lookback Period
# -------------------------------
period_choice = st.selectbox(
    "Choose Lookback Period",
    ["7d", "14d", "1mo", "3mo", "6mo", "1y"],
    index=0
)

def choose_interval(period):
    if period == "7d":
        return "1m"
    elif period == "14d":
        return "5m"
    elif period in ["1mo", "3mo"]:
        return "15m"
    else:
        return "30m"

interval_choice = choose_interval(period_choice)
st.write(f"**Selected Period:** {period_choice}, **Interval:** {interval_choice}")

# -------------------------------
# 2. Data Download and Preparation
# -------------------------------
@st.cache_data(show_spinner=True)
def load_data(ticker, period, interval):
    data = yf.download(ticker, period=period, interval=interval, auto_adjust=True)
    data.columns = data.columns.get_level_values(0)
    return data

data = load_data(ticker_symbol, period_choice, interval_choice)
if data.empty:
    st.error("No data downloaded. Check your network connection or ticker symbol.")
    st.stop()

data.index = pd.to_datetime(data.index)
data['Date'] = data.index.date

# -------------------------------
# 3. Create Daily Summary (including Volume)
# -------------------------------
daily_summary = data.groupby('Date').agg({
    'Open': 'first',
    'Close': 'last',
    'High': 'max',
    'Low': 'min',
    'Volume': 'sum'
}).dropna()

daily_summary['Prev_Close'] = daily_summary['Close'].shift(1)
daily_summary['Gap'] = daily_summary['Open'] - daily_summary['Prev_Close']
daily_summary['Gap_Pct'] = (daily_summary['Gap'] / daily_summary['Prev_Close']) * 100
daily_summary['Volume_Pct_Change'] = daily_summary['Volume'].pct_change() * 100
daily_summary.dropna(inplace=True)
daily_summary['Day_of_Week'] = pd.to_datetime(daily_summary.index).day_name()
daily_summary['Daily_Volatility'] = (daily_summary['High'] - daily_summary['Open']) / daily_summary['Open'] * 100

# -------------------------------
# 4. Download VIX Daily Data for the same date range
# -------------------------------
start_date = daily_summary.index.min().strftime("%Y-%m-%d")
end_date = daily_summary.index.max().strftime("%Y-%m-%d")
vix = yf.download("^VIX", start=start_date, end=end_date, interval="1d", auto_adjust=True)
vix.columns = vix.columns.get_level_values(0)
vix_daily = vix[['Close']].rename(columns={'Close': 'VIX_Close'})
vix_daily.index = vix_daily.index.date
daily_summary = daily_summary.merge(vix_daily, left_index=True, right_index=True, how='left')

# -------------------------------
# 5. Additional Column: Gap Type
# -------------------------------
def classify_gap(gap_pct):
    return "Small Gap" if abs(gap_pct) <= 1 else "Large Gap"
daily_summary['Gap_Type'] = daily_summary['Gap_Pct'].apply(classify_gap)

# -------------------------------
# 6. Intraday Gap Fill Analysis
# -------------------------------
gap_fill_data = []
for date in daily_summary.index:
    day_data = data[data['Date'] == date]
    if day_data.empty:
        continue
    prev_close = daily_summary.loc[date, 'Prev_Close']
    open_price = daily_summary.loc[date, 'Open']
    market_open = day_data.index[0]
    if open_price > prev_close:
        fill_condition = day_data['Low'] <= prev_close
    else:
        fill_condition = day_data['High'] >= prev_close
    fill_data = day_data[fill_condition]
    if not fill_data.empty:
        fill_time = fill_data.index[0]
        minutes_to_fill = (fill_time - market_open).seconds / 60
        filled = True
    else:
        minutes_to_fill = None
        filled = False
    gap_fill_data.append({
        'Date': date,
        'Minutes_to_Fill': minutes_to_fill,
        'Filled': filled
    })
gap_fill_df = pd.DataFrame(gap_fill_data).set_index('Date')
daily_summary = daily_summary.merge(gap_fill_df, left_index=True, right_index=True, how='left')
daily_summary['Filled'] = daily_summary['Filled'].fillna(False)

# -------------------------------
# 7. Historical Fill Probability (by Day of Week)
# -------------------------------
fill_prob = daily_summary.groupby('Day_of_Week')['Filled'].mean()
days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
fill_prob = fill_prob.reindex(days_order)
fill_prob_mapping = fill_prob.to_dict()
daily_summary['Historical_Fill_Probability'] = daily_summary['Day_of_Week'].map(fill_prob_mapping)

# -------------------------------
# 8. Historical Volume Metrics (by Day of Week)
# -------------------------------
grouped_volume = daily_summary.groupby('Day_of_Week')['Volume'].mean()
grouped_volume_pct = daily_summary.groupby('Day_of_Week')['Volume_Pct_Change'].mean()
daily_summary['Historical_Avg_Volume'] = daily_summary['Day_of_Week'].map(grouped_volume.to_dict())
daily_summary['Historical_Avg_Volume_Pct_Change'] = daily_summary['Day_of_Week'].map(grouped_volume_pct.to_dict())

# -------------------------------
# 9. Market Sentiment Column (dummy Economic_Event data)
# -------------------------------
dummy_events = {
    daily_summary.index[0]: "Positive earnings surprise",
    daily_summary.index[1]: "Slightly negative macro report",
    daily_summary.index[2]: None,
    daily_summary.index[3]: "Negative economic outlook",
    daily_summary.index[4]: "Slightly positive consumer sentiment"
}
daily_summary['Economic_Event'] = daily_summary.index.map(lambda d: dummy_events.get(d, None))
def sentiment_score(event):
    if event is None:
        return 0
    event_lower = event.lower()
    if "positive" in event_lower or "good" in event_lower or "better" in event_lower:
        return 1
    elif "negative" in event_lower or "bad" in event_lower or "worse" in event_lower:
        return -1
    elif "slightly positive" in event_lower:
        return 0.5
    elif "slightly negative" in event_lower:
        return -0.5
    else:
        return 0
daily_summary['Market_Sentiment'] = daily_summary['Economic_Event'].apply(sentiment_score)

# -------------------------------
# 10. Consolidated Daily Summary Display
# -------------------------------
final_columns = [
    'Open', 'Close', 'High', 'Low', 'Prev_Close', 'Gap', 'Gap_Pct', 
    'Day_of_Week', 'Daily_Volatility', 'Gap_Type', 'Economic_Event', 
    'Minutes_to_Fill', 'Filled', 'Historical_Fill_Probability', 'Market_Sentiment',
    'Volume', 'Volume_Pct_Change', 'Historical_Avg_Volume', 'Historical_Avg_Volume_Pct_Change',
    'VIX_Close'
]
daily_summary = daily_summary[final_columns]
st.subheader("Consolidated Daily Summary with Metrics")
st.dataframe(daily_summary)

# -------------------------------
# 11. Correlation: Day of Week vs. Gap Fill Probability Heatmap
# -------------------------------
st.subheader("Gap Fill Probability by Day of Week (Historical)")
day_prob_df = pd.DataFrame({'Fill_Probability': fill_prob})
fig2, ax2 = plt.subplots(figsize=(6, 2))
sns.heatmap(day_prob_df.T, annot=True, cmap='YlGnBu', vmin=0, vmax=1, ax=ax2)
ax2.set_title("Gap Fill Probability by Day of Week")
st.pyplot(fig2)

# -------------------------------
# 12. Next Day Prediction Section (including Volume Metrics)
# -------------------------------
last_date = pd.to_datetime(daily_summary.index.max())
next_trading_day = (last_date + BDay(1)).date()
next_day_name = pd.to_datetime(next_trading_day).day_name()
predicted_probability = fill_prob_mapping.get(next_day_name, None)
predicted_volatility = daily_summary[daily_summary['Day_of_Week'] == next_day_name]['Daily_Volatility'].mean()
predicted_vix = daily_summary[daily_summary['Day_of_Week'] == next_day_name]['VIX_Close'].mean()
predicted_volume = daily_summary[daily_summary['Day_of_Week'] == next_day_name]['Volume'].mean()
predicted_volume_pct_change = daily_summary[daily_summary['Day_of_Week'] == next_day_name]['Volume_Pct_Change'].mean()
st.subheader("Next Trading Day Prediction (Historical Averages)")
if predicted_probability is not None:
    st.write(f"The next trading day is **{next_trading_day} ({next_day_name})**.")
    st.write("Based on historical data over the lookback period:")
    st.write(f"- **Gap Fill Probability:** **{predicted_probability:.0%}**")
    st.write(f"- **Average Daily Volatility:** **{predicted_volatility:.2f}%**")
    st.write(f"- **Average VIX:** **{predicted_vix:.2f}**")
    st.write(f"- **Predicted Trading Volume:** **{predicted_volume:,.0f}**")
    st.write(f"- **Predicted Volume % Change:** **{predicted_volume_pct_change:.2f}%**")
else:
    st.write("Not enough historical data to predict next day's metrics.")

# -------------------------------
# 13. Machine Learning Model for Next Day Metrics Prediction
# -------------------------------
st.subheader("ML Next Day Predictions")
ml_data = daily_summary.sort_index().copy()
ml_data['Filled'] = ml_data['Filled'].fillna(False)
ml_data['next_open'] = ml_data['Open'].shift(-1)
ml_data['next_gap_filled'] = ml_data['Filled'].shift(-1)
ml_data['next_volume_pct_change'] = ml_data['Volume_Pct_Change'].shift(-1)
ml_data['next_minutes_to_fill'] = ml_data['Minutes_to_Fill'].shift(-1)
ml_data['next_price_change'] = ((ml_data['next_open'] - ml_data['Close']) / ml_data['Close']) * 100
ml_data = ml_data.dropna(subset=['next_open', 'next_gap_filled', 'next_volume_pct_change', 'next_minutes_to_fill', 'next_price_change'])
target_cols = ['next_price_change', 'next_gap_filled', 'next_volume_pct_change', 'next_minutes_to_fill']
feature_cols = ['Open', 'Close', 'High', 'Low', 'Gap', 'Gap_Pct', 'Daily_Volatility', 'Volume', 'Volume_Pct_Change', 'VIX_Close']
X_numeric = ml_data[feature_cols].copy()
X_day = pd.get_dummies(ml_data['Day_of_Week'], prefix='DOW')
X = pd.concat([X_numeric, X_day], axis=1)
y = ml_data[target_cols].copy()
st.write("Shape of Feature Matrix (X):", X.shape)
st.write("Shape of Target Matrix (y):", y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, y_pred, multioutput='raw_values')
st.write("Mean Absolute Error for each target on the test set:")
st.write(pd.Series(mae, index=target_cols))

test_results = X_test.reset_index(drop=True)
actual_vs_pred = pd.DataFrame(y_test.reset_index(drop=True))
actual_vs_pred.columns = [col + " (Actual)" for col in actual_vs_pred.columns]
pred_vs = pd.DataFrame(y_pred, columns=[col + " (Predicted)" for col in y.columns])
comparison = pd.concat([actual_vs_pred, pred_vs], axis=1)
st.write("Comparison of Actual vs. Predicted (first 5 rows):")
st.dataframe(comparison.head())

latest_features = X.iloc[[-1]]
next_day_prediction = model.predict(latest_features)[0]
predicted_metrics = {
    "Next Day Price Change (%)": next_day_prediction[0],
    "Next Day Gap Fill Probability": next_day_prediction[1],
    "Next Day Volume % Change": next_day_prediction[2],
    "Next Day Minutes to Fill": next_day_prediction[3]
}
st.subheader("Predicted Next Day Metrics (ML Model)")
for metric, value in predicted_metrics.items():
    st.write(f"- **{metric}:** {value:.2f}")

# -------------------------------
# 14. Option Chain CSV Upload (with Greeks)
# -------------------------------
st.subheader("Option Chain Upload")
st.markdown("""
Upload a CSV file containing your 0DTE option chain data, including columns for:
- **Strike**
- **Call Delta**, **Call Gamma**, **Call Theta**, **Call Vega**,
- **Put Delta**, **Put Gamma**, **Put Theta**, **Put Vega**,
- **(Optionally)** Call Bid, Call Ask, Put Bid, Put Ask, etc.
""")
uploaded_file = st.file_uploader("Upload Option Chain CSV", type=["csv"])
option_chain_df = None
if uploaded_file is not None:
    option_chain_df = pd.read_csv(uploaded_file)
    st.write("Uploaded Option Chain Data (first 10 rows):")
    st.dataframe(option_chain_df.head(10))

# -------------------------------
# 15. AI-Generated Option Order Combination (Using Uploaded Greeks)
# -------------------------------
st.subheader("AI-Generated Option Order Combination")

def generate_option_order_combination_with_uploaded_greeks(
    predicted_metrics,
    predicted_probability,
    option_chain_df,
    underlying_price,
    risk_parameters
):
    """
    Use the Assistants API to generate a JSON object for an optimal option order combination,
    referencing the pre-computed Greeks in the uploaded CSV.
    """
    if option_chain_df is not None and 'Strike' in option_chain_df.columns:
        available_strikes = option_chain_df['Strike'].dropna().unique().tolist()
        available_strikes = [round(float(s), 2) for s in available_strikes]
    else:
        available_strikes = []
    
    preview_df = option_chain_df.head(5) if option_chain_df is not None else pd.DataFrame()
    preview_json = preview_df.to_dict(orient="records")
    
    prompt = f"""
You are an expert in selecting option combinations to maximize profit while limiting risk, based on predictions of price movement for the next day and an available 0DTE options chain.

Analyze the provided data to determine if there exists a viable option trade opportunity for tomorrow. The uploaded option chain represents the current available 0DTE options and includes pre-computed Greeks (e.g. 'Call Delta', 'Call Gamma', 'Call Theta', 'Call Vega', 'Put Delta', 'Put Gamma', 'Put Theta', 'Put Vega').

Input Data:
- Current Underlying Price: {underlying_price:.2f}
- ML-Predicted Next Day Metrics: {predicted_metrics}
- Predicted Gap Fill Probability: {predicted_probability:.0%}
- Risk Criteria: {risk_parameters}
- Available Strikes: {available_strikes}
- Option Chain Sample (first 5 rows): {json.dumps(preview_json, indent=2)}

Determine if there is a profitable, risk-managed opportunity. If so, return only a valid JSON object exactly conforming to this schema:

{{
  "strike_price": number,
  "delta": number,
  "stop_loss": number,
  "take_profit": number,
  "limit_price": number,
  "type": "Put" or "Call",
  "estimated_profit": number,
  "possible_loss": number,
  "probability_of_success": number
}}

If no viable opportunity exists, return a JSON object with all values as "n/a". Do not output any extra text.
"""
    payload = {
        "assistant_id": "asst_umHbxf54jQ5W5up0oRFTlibH",
        "thread": {
            "messages": [
                {"role": "user", "content": prompt}
            ]
        },
        "response_format": "auto",
        "temperature": 0.2,
        "max_completion_tokens": 300
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai.api_key}",
        "OpenAI-Beta": "assistants=v2"
    }
    
    run_url = "https://api.openai.com/v1/threads/runs"
    run_response = requests.post(run_url, headers=headers, json=payload)
    if run_response.status_code != 200:
        return f"Error creating run: {run_response.text}"
    run_data = run_response.json()
    run_id = run_data["id"]
    thread_id = run_data["thread_id"]
    
    polling_status = st.empty()
    run_status = run_data.get("status")
    timeout = 90
    start_time = time.time()
    while run_status not in ["completed", "failed"]:
        polling_status.info(f"Waiting for assistant run to complete... (Status: {run_status})")
        time.sleep(2)
        poll_url = f"https://api.openai.com/v1/threads/{thread_id}/runs/{run_id}"
        poll_response = requests.get(poll_url, headers=headers)
        if poll_response.status_code != 200:
            return f"Error polling run: {poll_response.text}"
        run_data = poll_response.json()
        run_status = run_data.get("status")
        if time.time() - start_time > timeout:
            polling_status.info("Polling timed out; retrieving messages anyway.")
            break
    
    polling_status.info("Assistant run completed. Retrieving messages...")
    messages_url = f"https://api.openai.com/v1/threads/{thread_id}/messages"
    messages_response = requests.get(messages_url, headers=headers)
    if messages_response.status_code != 200:
        return f"Error retrieving messages: {messages_response.text}"
    messages_data = messages_response.json().get("data", [])
    # Check for messages with role "assistant" or "function"
    assistant_messages = [m for m in messages_data if m.get("role") in ["assistant", "function"]]
    if not assistant_messages:
        return {"strike_price": "n/a", "delta": "n/a", "stop_loss": "n/a", "take_profit": "n/a", "limit_price": "n/a", "type": "n/a", "estimated_profit": "n/a", "possible_loss": "n/a", "probability_of_success": "n/a"}
    last_message = assistant_messages[-1]
    try:
        output_text = last_message["content"][0]["text"]["value"].strip()
        if not output_text:
            fallback = {"strike_price": "n/a", "delta": "n/a", "stop_loss": "n/a", "take_profit": "n/a", "limit_price": "n/a", "type": "n/a", "estimated_profit": "n/a", "possible_loss": "n/a", "probability_of_success": "n/a"}
            return fallback
        result = json.loads(output_text)
        if not result:
            fallback = {"strike_price": "n/a", "delta": "n/a", "stop_loss": "n/a", "take_profit": "n/a", "limit_price": "n/a", "type": "n/a", "estimated_profit": "n/a", "possible_loss": "n/a", "probability_of_success": "n/a"}
            return fallback
        return result
    except Exception:
        fallback = {"strike_price": "n/a", "delta": "n/a", "stop_loss": "n/a", "take_profit": "n/a", "limit_price": "n/a", "type": "n/a", "estimated_profit": "n/a", "possible_loss": "n/a", "probability_of_success": "n/a"}
        return fallback

if option_chain_df is not None:
    risk_parameters = {
        "Desired Net Delta Range": "(-0.2, 0.2)",
        "Max Acceptable Theta": "-10 per day"
    }
    underlying_price = float(data['Close'].iloc[-1])
    ai_output = generate_option_order_combination_with_uploaded_greeks(
        predicted_metrics=predicted_metrics,
        predicted_probability=predicted_probability if predicted_probability else 0.0,
        option_chain_df=option_chain_df,
        underlying_price=underlying_price,
        risk_parameters=risk_parameters
    )
    st.subheader("Option Order Combination (AI Output)")
    if isinstance(ai_output, dict):
        df_orders = pd.DataFrame([ai_output])
        st.dataframe(df_orders)
    else:
        st.write(ai_output)
else:
    st.info("Please upload an option chain CSV (with Greeks) to generate an AI strategy recommendation.")


