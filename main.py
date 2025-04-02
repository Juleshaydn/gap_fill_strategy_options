import nest_asyncio
nest_asyncio.apply()

import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, time
from pandas.tseries.offsets import BDay

# For machine learning:
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error

import math
from math import log, sqrt, exp
from scipy.stats import norm

from openai_strategy import generate_option_strategy


st.set_page_config(page_title="SPY Gap Fill Backtesting Dashboard", layout="wide")
st.title("SPY Gap Fill Backtesting Dashboard with Consolidated Metrics, ML, and Automated 0DTE Strategy")

st.markdown("""
This dashboard allows you to select a lookback period for SPY historical data. 
It calculates daily gaps, performs intraday gap fill analysis, and computes various metrics 
(including market volatility, VIX, trading volume, and market sentiment). All these metrics 
are consolidated into one daily summary table. A heatmap shows the day-of-week correlation 
of the historical gap fill probability. An ML model is built to predict next day's metrics 
(price change, gap fill probability, volume % change, and minutes to fill) based on historical data.

Finally, you can **upload a CSV of the option chain** for a specific 0DTE date. 
The system will parse the chain, pick a nearest ATM strike, and create an **automated vertical spread** 
(bullish or bearish) based on the predicted next day metrics. 
If no valid opportunity is found, it returns **"No Trading Opportunities."**
""")

# -----------------------------------
# 1. User Input for Lookback Period
# -----------------------------------
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

# -----------------------------------
# 2. Data Download and Preparation
# -----------------------------------
@st.cache_data(show_spinner=True)
def load_data(period, interval):
    data = yf.download("SPY", period=period, interval=interval, auto_adjust=True)
    data.columns = data.columns.get_level_values(0)
    return data

spy = load_data(period_choice, interval_choice)
if spy.empty:
    st.error("No data downloaded. Check your network connection or ticker symbol.")
    st.stop()

spy.index = pd.to_datetime(spy.index)
spy['Date'] = spy.index.date

# -----------------------------------
# 3. Create Daily Summary for SPY (including Volume)
# -----------------------------------
daily_summary = spy.groupby('Date').agg({
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

# -----------------------------------
# 4. Download VIX Daily Data for the same date range
# -----------------------------------
start_date = daily_summary.index.min().strftime("%Y-%m-%d")
end_date = daily_summary.index.max().strftime("%Y-%m-%d")
vix = yf.download("^VIX", start=start_date, end=end_date, interval="1d", auto_adjust=True)
vix.columns = vix.columns.get_level_values(0)
vix_daily = vix[['Close']].rename(columns={'Close': 'VIX_Close'})
vix_daily.index = vix_daily.index.date
daily_summary = daily_summary.merge(vix_daily, left_index=True, right_index=True, how='left')

# -----------------------------------
# 5. Additional Column: Gap Type
# -----------------------------------
def classify_gap(gap_pct):
    return "Small Gap" if abs(gap_pct) <= 1 else "Large Gap"

daily_summary['Gap_Type'] = daily_summary['Gap_Pct'].apply(classify_gap)

# -----------------------------------
# 6. Intraday Gap Fill Analysis
# -----------------------------------
gap_fill_data = []
for date in daily_summary.index:
    day_data = spy[spy['Date'] == date]
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

# -----------------------------------
# 7. Historical Fill Probability (by Day of Week)
# -----------------------------------
fill_prob = daily_summary.groupby('Day_of_Week')['Filled'].mean()
days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
fill_prob = fill_prob.reindex(days_order)
fill_prob_mapping = fill_prob.to_dict()
daily_summary['Historical_Fill_Probability'] = daily_summary['Day_of_Week'].map(fill_prob_mapping)

# -----------------------------------
# 8. Historical Volume Metrics (by Day of Week)
# -----------------------------------
grouped_volume = daily_summary.groupby('Day_of_Week')['Volume'].mean()
grouped_volume_pct = daily_summary.groupby('Day_of_Week')['Volume_Pct_Change'].mean()
daily_summary['Historical_Avg_Volume'] = daily_summary['Day_of_Week'].map(grouped_volume.to_dict())
daily_summary['Historical_Avg_Volume_Pct_Change'] = daily_summary['Day_of_Week'].map(grouped_volume_pct.to_dict())

# -----------------------------------
# 9. Market Sentiment Column (using dummy Economic_Event data)
# -----------------------------------
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

# -----------------------------------
# 10. Consolidated Daily Summary Display
# -----------------------------------
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

# -----------------------------------
# 11. Correlation: Day of Week vs. Gap Fill Probability Heatmap
# -----------------------------------
st.subheader("Gap Fill Probability by Day of Week (Historical)")
day_prob_df = pd.DataFrame({'Fill_Probability': fill_prob})
fig2, ax2 = plt.subplots(figsize=(6, 2))
sns.heatmap(day_prob_df.T, annot=True, cmap='YlGnBu', vmin=0, vmax=1, ax=ax2)
ax2.set_title("Gap Fill Probability by Day of Week")
st.pyplot(fig2)

# -----------------------------------
# 12. Next Day Prediction Section (including Volume Metrics)
# -----------------------------------
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

# -----------------------------------
# 13. Machine Learning Model for Next Day Metrics Prediction
# -----------------------------------
st.subheader("Machine Learning: Next Day Metrics Prediction")
st.markdown("""
We now build an ML model using historical daily data to predict next day metrics:
- **Next Day Price Change (%)**: ((next_open - today's close) / today's close * 100)
- **Next Day Gap Fill Probability** (0 if gap not filled, 1 if filled)
- **Next Day Volume % Change**
- **Next Day Minutes to Fill**
""")

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

# -----------------------------------
# 14. Option Chain CSV Upload
# -----------------------------------
st.subheader("Option Chain CSV Upload")
st.markdown("""
Upload a CSV file containing your 0DTE option chain data (e.g. columns like Strike, Calls Bid, Calls Ask, Puts Bid, Puts Ask, etc.).
""")

uploaded_file = st.file_uploader("Upload your Option Chain CSV", type=["csv"])
option_chain_df = None
if uploaded_file is not None:
    option_chain_df = pd.read_csv(uploaded_file)
    st.write("Uploaded Option Chain Data (first 10 rows):")
    st.dataframe(option_chain_df.head(10))

# -----------------------------------
# 15. Automated 0DTE Option Strategy Recommendation (Using Option Chain)
# -----------------------------------
st.subheader("Automated 0DTE Option Strategy Recommendation")

st.markdown("""
Based on the ML-predicted gap fill probability and next day price change, 
the system will automatically generate an optimal 0DTE vertical spread by looking up 
strikes in your uploaded option chain. If no valid strikes are found, it returns 
"No Trading Opportunities."
""")

min_gap_fill_prob = 0.80
default_time_to_expiry_minutes = 30
strike_offset = st.number_input("Strike Offset for Spread (USD)", value=5.0, step=0.5)
contracts = st.number_input("Number of Contracts per Leg", value=1, min_value=1)

underlying_price = float(spy['Close'].iloc[-1])
predicted_price_change = predicted_metrics["Next Day Price Change (%)"]

def black_scholes_greeks(S, K, T, r, sigma, option='call'):
    d1 = (log(S / K) + (r + sigma**2 / 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    if option == 'call':
        price = S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
        theta = - (S * sigma * norm.pdf(d1)) / (2 * sqrt(T)) - r * K * exp(-r * T) * norm.cdf(d2)
    else:
        price = K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1
        theta = - (S * sigma * norm.pdf(d1)) / (2 * sqrt(T)) + r * K * exp(-r * T) * norm.cdf(-d2)
    gamma = norm.pdf(d1) / (S * sigma * sqrt(T))
    vega = S * sqrt(T) * norm.pdf(d1)
    return price, delta, gamma, theta, vega

T_auto = default_time_to_expiry_minutes / (60 * 24 * 365)
r_auto = 0.01
sigma_auto = 0.20

# Let user override if they want
r_auto = st.number_input("Risk-Free Rate (Strategy)", value=0.01, step=0.001)
sigma_auto = st.number_input("Implied Volatility (Strategy)", value=0.20, step=0.01)

if predicted_probability is None or predicted_probability < min_gap_fill_prob:
    st.error("No Trading Opportunities: Predicted gap fill probability is too low.")
else:
    # Determine direction
    if predicted_price_change > 0:
        strategy = "bullish"
    elif predicted_price_change < 0:
        strategy = "bearish"
    else:
        strategy = "neutral"
    
    if strategy == "neutral":
        st.error("No Trading Opportunities: Predicted price change is neutral.")
    else:
        if option_chain_df is None:
            st.error("Please upload an option chain CSV first.")
        else:
            # Attempt to find an ATM strike
            # e.g. we assume the chain has a 'Strike' column
            option_chain_df['StrikeDiff'] = (option_chain_df['Strike'] - underlying_price).abs()
            atm_row = option_chain_df.loc[option_chain_df['StrikeDiff'].idxmin()]
            atm_strike = atm_row['Strike']
            
            if strategy == "bullish":
                # Look for short strike = atm_strike + offset
                short_strike_target = atm_strike + strike_offset
            else:  # "bearish"
                short_strike_target = atm_strike - strike_offset
            
            # find the row in the chain that's closest to short_strike_target
            option_chain_df['OffsetDiff'] = (option_chain_df['Strike'] - short_strike_target).abs()
            short_row = option_chain_df.loc[option_chain_df['OffsetDiff'].idxmin()]
            short_strike = short_row['Strike']
            
            st.write(f"ATM Strike found: {atm_strike}, Short Strike found: {short_strike}")
            
            # If the short_strike is the same as atm_strike, might not be a valid offset
            if abs(atm_strike - short_strike) < 0.5:
                st.error("No Trading Opportunities: Could not find a valid offset strike in the chain.")
            else:
                # For demonstration, let's just do a quick black-scholes calc
                # ignoring the actual bid/ask from the chain
                if strategy == "bullish":
                    long_option_type = "call"
                    short_option_type = "call"
                else:
                    long_option_type = "put"
                    short_option_type = "put"
                
                long_price, long_delta, long_gamma, long_theta, long_vega = black_scholes_greeks(
                    underlying_price, atm_strike, T_auto, r_auto, sigma_auto, option=long_option_type)
                short_price, short_delta, short_gamma, short_theta, short_vega = black_scholes_greeks(
                    underlying_price, short_strike, T_auto, r_auto, sigma_auto, option=short_option_type)
                
                # Weighted by position
                pos_factor_long = 1
                pos_factor_short = -1
                agg_delta = (long_delta * pos_factor_long * contracts) + (short_delta * pos_factor_short * contracts)
                agg_gamma = (long_gamma * pos_factor_long * contracts) + (short_gamma * pos_factor_short * contracts)
                agg_theta = (long_theta * pos_factor_long * contracts) + (short_theta * pos_factor_short * contracts)
                agg_vega = (long_vega * pos_factor_long * contracts) + (short_vega * pos_factor_short * contracts)
                net_premium = (long_price * contracts) - (short_price * contracts)
                
                st.write("### Generated Spread from Option Chain")
                st.write(f"**Long Leg:** {long_option_type.capitalize()} at {atm_strike}, {contracts} contracts")
                st.write(f"**Short Leg:** {short_option_type.capitalize()} at {short_strike}, {contracts} contracts")
                st.write(f"**Net Premium (Cost):** {net_premium:.2f}")
                st.write(f"**Aggregated Delta:** {agg_delta:.2f}")
                st.write(f"**Aggregated Gamma:** {agg_gamma:.4f}")
                st.write(f"**Aggregated Theta:** {agg_theta:.2f} per day")
                st.write(f"**Aggregated Vega:** {agg_vega:.2f}")
                
                # Check viability
                desired_net_delta_range = (-0.2, 0.2)
                max_acceptable_net_theta = -10
                if desired_net_delta_range[0] <= agg_delta <= desired_net_delta_range[1] and agg_theta > max_acceptable_net_theta:
                    st.success("Recommended Trade: The automated 0DTE vertical spread meets risk-adjusted criteria.")
                else:
                    st.error("No Trading Opportunities: The aggregated Greeks do not meet the desired risk-adjusted criteria.")

st.markdown("""
**Note:** 
- This approach uses the uploaded CSV to find an ATM strike and an offset strike. 
- We still compute the Greeks with Black-Scholes for demonstration (ignoring the chain’s actual bid/ask). 
- You can further refine this to use the chain’s bid/ask or actual IV for each strike to compute more accurate prices.
""")

# -----------------------------------
# 16. AI-Generated Option Strategy Recommendation
# -----------------------------------
st.subheader("AI-Generated Option Strategy Recommendation")

# Define risk parameters (you can adjust these)
risk_parameters = {
    "Desired Net Delta Range": "(-0.2, 0.2)",
    "Max Acceptable Theta": "-10 per day"
}

# Call the OpenAI API to generate the strategy
if option_chain_df is not None:
    ai_strategy = generate_option_strategy(predicted_metrics, predicted_probability, option_chain_df, float(spy['Close'].iloc[-1]), risk_parameters)
    st.markdown("### Recommended Strategy:")
    st.write(ai_strategy)
else:
    st.info("Please upload an option chain CSV to generate an AI strategy recommendation.")

