import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.stats import norm

# Streamlit App Title
st.title("📈 Monte Carlo Portfolio Simulation")
st.subheader("Before running the Stimulation, Click on Fetch Prices!")

# Sidebar - Stock Selection
st.sidebar.header("Simulation Parameters")

# User inputs stock tickers (comma-separated)
tickers_input = st.sidebar.text_input("Enter Stock Tickers (comma-separated, e.g., AAPL, TSLA, GOOG)", "AAPL, TSLA")
tickers = [t.strip().upper() for t in tickers_input.split(",")]

# Fetch real-time stock prices
if "initial_prices" not in st.session_state:
    st.session_state["initial_prices"] = {}

if st.sidebar.button("Fetch Prices"):
    st.session_state["initial_prices"] = {}
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")  # Fetch past 1 year data
        if hist.empty:
            st.sidebar.error(f"Invalid Ticker: {ticker}")
        else:
            last_price = hist["Close"].iloc[-1]
            st.session_state["initial_prices"][ticker] = last_price
            st.sidebar.success(f"Latest price for {ticker}: ${last_price:.2f}")

# Default initial prices
initial_prices = st.session_state.get("initial_prices", {t: 100 for t in tickers})

# Portfolio Weights
st.sidebar.subheader("Portfolio Weights")
weights = {}
for ticker in tickers:
    weights[ticker] = st.sidebar.slider(f"Weight for {ticker} (%)", min_value=0, max_value=100, value=100 // len(tickers), step=1)
total_weight = sum(weights.values())
if total_weight != 100:
    st.sidebar.warning("⚠️ Portfolio weights should sum up to 100%!")

# User Input Parameters
mu = st.sidebar.number_input("Expected Annual Return (mu)", value=0.10, step=0.01)
sigma = st.sidebar.number_input("Portfolio Volatility (sigma)", value=0.25, step=0.01)
days = st.sidebar.number_input("Number of Days", value=252, step=1)
num_simulations = st.sidebar.number_input("Number of Simulations", value=1000, step=100)

# Run Simulation Button
if st.sidebar.button("Run Simulation"):
    # Monte Carlo Simulation
    np.random.seed(42)
    dt = 1 / days

    # Convert weights to fractions
    weights = {t: w / total_weight for t, w in weights.items()}  # Normalize

    # Store individual stock price paths
    stock_paths = {t: np.zeros((days, num_simulations)) for t in tickers}
    
    # Initialize with actual prices
    for t in tickers:
        stock_paths[t][0] = initial_prices[t]

    # Simulate stock prices
    for t in tickers:
        for day in range(1, days):
            rand = np.random.standard_normal(num_simulations)
            stock_paths[t][day] = stock_paths[t][day - 1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rand)

    # Calculate Portfolio Value
    portfolio_paths = np.zeros((days, num_simulations))
    for t in tickers:
        portfolio_paths += stock_paths[t] * weights[t]  # Weighted sum

    # Plot Simulated Portfolio Prices
    st.subheader("📊 Simulated Portfolio Price Paths")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(portfolio_paths, alpha=0.2)
    ax.set_xlabel("Days")
    ax.set_ylabel("Portfolio Value")
    ax.set_title(f"Monte Carlo Portfolio Simulation")
    st.pyplot(fig)

    # Calculate Portfolio Returns
    portfolio_returns = (portfolio_paths[-1] - sum(initial_prices[t] * weights[t] for t in tickers)) / sum(initial_prices[t] * weights[t] for t in tickers)

    # **Risk Analysis Metrics**
    st.subheader("📉 Risk Analysis")

    # **1. 95% Value at Risk (VaR)**
    VaR_95 = np.percentile(portfolio_returns, 5) * sum(initial_prices[t] * weights[t] for t in tickers)
    st.write(f"**95% Value at Risk (VaR):** ${VaR_95:.2f}")

    # **2. Expected Shortfall (Conditional VaR)**
    ES_95 = portfolio_returns[portfolio_returns <= (VaR_95 / sum(initial_prices[t] * weights[t] for t in tickers))].mean() * sum(initial_prices[t] * weights[t] for t in tickers)
    st.write(f"**Expected Shortfall (ES):** ${ES_95:.2f}")

    # **3. Sharpe Ratio**
    risk_free_rate = 0.02  # Assume 2% risk-free rate
    sharpe_ratio = (np.mean(portfolio_returns) - (risk_free_rate / 252)) / np.std(portfolio_returns)
    st.write(f"**Sharpe Ratio:** {sharpe_ratio:.2f}")

    # **4. Maximum Drawdown**
    rolling_max = np.maximum.accumulate(portfolio_paths, axis=0)
    drawdown = (portfolio_paths - rolling_max) / rolling_max
    max_drawdown = np.min(drawdown)
    st.write(f"**Max Drawdown:** {max_drawdown:.2%}")

# Footer
st.markdown("---")
st.markdown("📊 Monte Carlo Portfolio Simulation")
