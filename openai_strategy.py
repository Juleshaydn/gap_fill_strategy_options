import openai
import pandas as pd

openai.api_key = "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"

def generate_option_strategy(predicted_metrics, predicted_probability, option_chain_df, underlying_price, risk_parameters):
    """
    Generate an optimal 0DTE option strategy using OpenAI's API.
    
    Parameters:
        predicted_metrics (dict): ML-predicted metrics (e.g., next day price change, gap fill probability, etc.).
        predicted_probability (float): Predicted next day gap fill probability (as a decimal, e.g. 0.85 for 85%).
        option_chain_df (pd.DataFrame): Uploaded option chain DataFrame (must include a 'Strike' column).
        underlying_price (float): Current underlying price.
        risk_parameters (dict): Risk criteria thresholds (e.g., desired net delta range, max theta decay).
    
    Returns:
        str: Recommended option strategy or "No Trading Opportunities".
    """
    # Create a list of available strikes from the option chain
    if option_chain_df is not None and 'Strike' in option_chain_df.columns:
        available_strikes = option_chain_df['Strike'].dropna().unique().tolist()
        available_strikes = [round(float(s), 2) for s in available_strikes]
    else:
        available_strikes = []

    # Compose the prompt
    prompt = f"""
    You are a quantitative options trading expert. I have the following data:
    
    - Current underlying price: {underlying_price:.2f}
    - Available strikes from the option chain: {available_strikes}
    - Predicted next day metrics: {predicted_metrics}
    - Predicted gap fill probability: {predicted_probability:.0%}
    - Risk management criteria: {risk_parameters}
    
    Based on these inputs, please propose the optimal 0DTE options trading strategy. 
    Specify the exact strategy (for example, a call vertical spread or put vertical spread), including:
      - Which option leg to buy and which to sell
      - The strike prices to use
      - The number of contracts for each leg
      - Your reasoning behind this recommendation.
    
    If no viable trading opportunity exists, simply respond with "No Trading Opportunities".
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a quantitative options trading expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=300
        )
        strategy = response.choices[0].message["content"]
        return strategy
    except Exception as e:
        return f"Error generating strategy: {e}"

# For testing this module standalone:
if __name__ == "__main__":
    # Dummy values for testing
    predicted_metrics = {
        "Next Day Price Change (%)": 1.2,
        "Next Day Gap Fill Probability": 0.85,
        "Next Day Volume % Change": 2.5,
        "Next Day Minutes to Fill": 15
    }
    predicted_probability = 0.85
    # Create a dummy option chain DataFrame
    dummy_chain = pd.DataFrame({
        "Strike": [415, 420, 425, 430]
    })
    underlying_price = 420.0
    risk_parameters = {
        "Desired Net Delta Range": "(-0.2, 0.2)",
        "Max Acceptable Theta": "-10 per day"
    }
    
    strategy = generate_option_strategy(predicted_metrics, predicted_probability, dummy_chain, underlying_price, risk_parameters)
    print("Generated Strategy:")
    print(strategy)
