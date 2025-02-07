"""
Stock Market Analysis and Prediction System
Author: Aidan Frost
Date: 2025-01-31
Description: Real-time stock tracking with AI-powered predictions and visualization
"""
# Import required libraries
import ollama
import yfinance as yf
import mplfinance as mpf
import time
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import re
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sys
import atexit
@atexit.register
def cleanup():
    plt.close('all')
    matplotlib.pyplot.close('all')

# Configure matplotlib backend
matplotlib.use('Qt5Agg')  # At the top of your file

# ========================
# DATA MANAGEMENT FUNCTIONS
# ========================

def save_to_csv(ticker, data):
    """Saves historical data to CSV file with deduplication"""
    filename = f"{ticker}_historical.csv"
    try:
        if not os.path.exists(filename):
            pd.DataFrame([data]).to_csv(filename, index=False)
        else:
            existing_df = pd.read_csv(filename)
            new_df = pd.concat([existing_df, pd.DataFrame([data])])
            new_df['Datetime'] = pd.to_datetime(new_df['Datetime'])
            new_df = new_df.drop_duplicates(subset=['Datetime'])
            new_df.to_csv(filename, index=False)
    except Exception as e:
        print(f"Error saving to CSV: {e}")




def analyze_historical_data(ticker):
    """Analyzes historical data for daily performance metrics"""
    filename = f"{ticker}_historical.csv"
    if not os.path.exists(filename):
        print("No historical data found.")
        return

    try:
        df = pd.read_csv(filename, parse_dates=['Datetime'])
        today = datetime.now().date()
        today_data = df[df['Datetime'].dt.date == today]
        
        if today_data.empty:
            print("No data for today yet.")
            return
            
        # Calculate daily metrics
        open_price = today_data.iloc[0]['Open']
        current_price = today_data.iloc[-1]['Close']
        high = today_data['High'].max()
        low = today_data['Low'].min()
        volume = today_data['Volume'].sum()
        movement = ((current_price - open_price) / open_price) * 100
        
        # Display formatted analysis
        print(f"\n📅 Today's Historical Analysis ({ticker}):")
        print(f"Open: {open_price:.2f}")
        print(f"Current Price: {current_price:.2f}")
        print(f"Day Range: {low:.2f} - {high:.2f}")
        print(f"Volume: {volume:,}")
        print(f"Daily Movement: {'▲' if movement >= 0 else '▼'} {abs(movement):.2f}%")

    except Exception as e:
        print(f"Error analyzing historical data: {e}")

# ======================
# MARKET ANALYSIS MODULE
# ======================

def get_market_sentiment():
    """Calculates market sentiment from major indices"""
    tickers = {
        "S&P 500": "^GSPC",
        "Nasdaq": "^IXIC",
        "Dow Jones": "^DJI"
    }
    
    sentiment_score = 5  # Neutral baseline
    print("\n🌐 Analyzing Market Sentiment...")
    
    for index, ticker in tickers.items():
        try:
            data = yf.Ticker(ticker).history(period="2d")
            if len(data) < 2:
                continue
                
            # Calculate index movements
            prev_close = data['Close'].iloc[-2]
            last_close = data['Close'].iloc[-1]
            change_pct = (last_close - prev_close) / prev_close * 100
            change_vol = data['Volume'].iloc[-1] / data['Volume'].iloc[-2] - 1
            
            # Update sentiment score
            sentiment_score += change_pct / 10 + change_vol / 10

        except Exception as e:
            print(f"Error processing {index}: {e}")

    # Normalize and classify sentiment
    sentiment_score = max(0, min(10, sentiment_score))
    sentiment_index = int(sentiment_score // 2)
    sentiment_labels = [
        "Very Bearish", "Bearish", "Neutral",
        "Bullish", "Very Bullish"
    ]
    

    print(f"Market Sentiment Score: {sentiment_score:.2f}/10 ({sentiment_labels[sentiment_index]})")
    return {
        "score": sentiment_score,
        "label": sentiment_labels[sentiment_index]
    }

# ===========================
# TECHNICAL ANALYSIS FUNCTIONS
# ===========================

def calculate_technical_indicators(data):
    """Calculates various technical indicators for price analysis"""

    print("\n📈💯 Completing Technical Analysis...")
    # Simple Moving Averages
    data["SMA_5"] = data["Close"].rolling(window=5).mean()
    data["SMA_10"] = data["Close"].rolling(window=10).mean()

    # Relative Strength Index (RSI)
    delta = data["Close"].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data["RSI"] = 100 - (100 / (1 + rs))

    # Moving Average Convergence Divergence (MACD)
    short_ema = data["Close"].ewm(span=12, adjust=False).mean()
    long_ema = data["Close"].ewm(span=26, adjust=False).mean()
    data["MACD"] = short_ema - long_ema
    data["Signal_Line"] = data["MACD"].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    data["BB_Mid"] = data["Close"].rolling(window=20).mean()
    data["BB_Upper"] = data["BB_Mid"] + (2 * data["Close"].rolling(window=20).std())
    data["BB_Lower"] = data["BB_Mid"] - (2 * data["Close"].rolling(window=20).std())

    #print("\n- Technicals: RSI {intraday_data['RSI'].iloc[-1]:.1f}, MACD {intraday_data['MACD'].iloc[-1]:.4f}")

    return data

def detect_support_resistance(df, window=20):
    """Identifies key support/resistance levels using rolling extremes"""
    highs = df['High'].rolling(window=window, center=True).max()
    lows = df['Low'].rolling(window=window, center=True).min()
    return lows.iloc[-1], highs.iloc[-1]

def detect_candlestick_patterns(df):
    """Detects common candlestick patterns for market sentiment"""
    patterns = []
    
    # Engulfing Patterns
    df['Bullish_Engulfing'] = (
        (df['Close'] > df['Open']) &
        (df['Close'].shift(1) < df['Open'].shift(1)) &
        (df['Close'] > df['Open'].shift(1)) &
        (df['Open'] < df['Close'].shift(1))
    )
    
    df['Bearish_Engulfing'] = (
        (df['Close'] < df['Open']) &
        (df['Close'].shift(1) > df['Open'].shift(1)) &
        (df['Close'] < df['Open'].shift(1)) &
        (df['Open'] > df['Close'].shift(1))
    )
    
    # Doji Detection
    df['Doji'] = abs(df['Close'] - df['Open']) / (df['High'] - df['Low']) < 0.1
    
    # Identify latest patterns
    latest = df.iloc[-1]
    if latest['Bullish_Engulfing']:
        patterns.append("Bullish Engulfing")
    if latest['Bearish_Engulfing']:
        patterns.append("Bearish Engulfing")
    if latest['Doji']:
        patterns.append("Doji")
    
    return patterns

# ======================
# VISUALIZATION FUNCTIONS
# ======================

def graphing_function(stock_data, actual_points, predicted_points, ticker_symbol):
    """Generates interactive candlestick chart with predictions"""
    plt.close('all')  # Clear previous plot
    
    # Create figure with proper timezone handling
    fig = plt.figure(figsize=(12, 7), facecolor='white')
    ax = fig.add_subplot(111)
    
    # Convert stock_data index to datetime if needed
    if not isinstance(stock_data.index, pd.DatetimeIndex):
        stock_data.index = pd.to_datetime(stock_data.index)
    
    # Plot candlestick chart
    mpf.plot(stock_data, type='candle', ax=ax, volume=False, style='charles')
    
    # Convert actual_points datetimes if needed
    if not actual_points.empty:
        actual_points['Datetime'] = pd.to_datetime(actual_points['Datetime'])
        # Filter to current session
        today = stock_data.index[-1].date()
        actual_points = actual_points[pd.to_datetime(actual_points['Datetime']).dt.date == today]
        
        # Plot actual prices with proper formatting
        ax.plot(actual_points['Datetime'], 
                actual_points['Price'], 
                'bo-', alpha=0.5, markersize=6, 
                linewidth=1, label='Actual Prices')
    
    # Handle predictions
    if not predicted_points.empty:
        # Convert and filter predictions
        predicted_points['Datetime'] = pd.to_datetime(predicted_points['Datetime'])
        today_preds = predicted_points[pd.to_datetime(predicted_points['Datetime']).dt.date == today]
        
        if not today_preds.empty:
            # Plot predictions with future markers
            ax.plot(today_preds['Datetime'], 
                    today_preds['Price'], 
                    'rx', markersize=10, 
                    markeredgewidth=2, 
                    label='AI Predictions')
            
            # Add prediction time labels
            for idx, row in today_preds.iterrows():
                ax.annotate(f"{row['Price']:.2f}\n({row['Datetime'].strftime('%H:%M')})",
                          (row['Datetime'], row['Price']),
                          textcoords="offset points",
                          xytext=(0,10),
                          ha='center',
                          color='darkred',
                          fontsize=8)

    # Configure plot aesthetics
    ax.set_title(f'{ticker_symbol} Price Analysis\n{datetime.now().strftime("%Y-%m-%d %H:%M")}', 
               fontsize=14, pad=20)
    ax.set_ylabel('Price (USD)', fontsize=12)
    ax.yaxis.tick_right()
    ax.grid(True, alpha=0.3)
    
    # Format x-axis dates
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    fig.autofmt_xdate(rotation=0, ha='center')
    
    # Add legend and layout tweaks
    ax.legend(loc='upper left', framealpha=0.9)
    fig.tight_layout(pad=2.0)
    
    # Update display
    plt.draw()
    plt.pause(0.1)

# ======================
# MAIN TRACKING FUNCTION
# ======================

def track_stock_price(ticker_symbol):
    """Main function for real-time stock tracking and analysis"""
    # Initial setup
    analyze_historical_data(ticker_symbol)
    market_sentiment = get_market_sentiment()
    ticker = yf.Ticker(ticker_symbol)
    
    # Initialize data structures
    session_high = session_low = session_open = None
    actual_points = pd.DataFrame(columns=['Datetime', 'Price'])
    predicted_points = pd.DataFrame(columns=['Datetime', 'Price'])
    previous_length = 0
    
    # Configure DataFrame data types
    actual_points['Datetime'] = pd.to_datetime(actual_points['Datetime'], utc=True)
    actual_points['Price'] = pd.to_numeric(actual_points['Price'])
    predicted_points['Datetime'] = pd.to_datetime(predicted_points['Datetime'], utc=True)
    predicted_points['Price'] = pd.to_numeric(predicted_points['Price'])
    
    # Enable interactive plotting
    plt.ion()

    try:
        while True:
            # Fetch intraday data
            intraday_data = ticker.history(period="1d", interval="1m")
            if intraday_data.empty:
                time.sleep(5)
                continue

            current_length = len(intraday_data)
            if current_length > previous_length:
                latest = intraday_data.iloc[-1]
                timestamp = latest.name
                current_price = latest['Close']

                # Initialize session values
                if session_open is None:
                    session_open = latest['Open']
                    session_high = latest['High']
                    session_low = latest['Low']
                
                # Update session extremes
                session_high = max(session_high, latest['High'])
                session_low = min(session_low, latest['Low'])
                
                # Store actual price data
                new_actual = pd.DataFrame({
                    'Datetime': [timestamp],
                    'Price': [current_price]
                })
                actual_points = pd.concat([actual_points, new_actual], ignore_index=True)
                previous_length = current_length

                # Save to historical data
                save_data = {
                    'Datetime': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    'Open': latest['Open'],
                    'High': latest['High'],
                    'Low': latest['Low'],
                    'Close': latest['Close'],
                    'Volume': latest['Volume']
                }
                save_to_csv(ticker_symbol, save_data)

                # Generate AI predictions when enough data exists
                if len(intraday_data) > 20:
                    intraday_data = calculate_technical_indicators(intraday_data)
                    patterns = detect_candlestick_patterns(intraday_data)
                    support, resistance = detect_support_resistance(intraday_data)
                    
                    # Construct AI prompt
                    # Add this near the top of your track_stock_price function
                    historical_df = pd.read_csv(f"{ticker_symbol}_historical.csv", parse_dates=['Datetime'])
                    today_history = historical_df[historical_df['Datetime'].dt.date == datetime.now().date()]

# Format historical data for the prompt
                    price_history = "\n".join(
                        [f"- {row['Datetime'].strftime('%H:%M:%S')}: ${row['Close']:.2f}" 
                        for _, row in today_history.iterrows()][-15:]  # Last 15 data points
                    )

                    message_content = f"""**Market Analysis Request**
Current {ticker_symbol} Status:
- Price: ${current_price:.2f} (Open: {save_data['Open']:.2f})
- Time Right Noe: {timestamp.strftime('%H:%M')}
- Session Range: {session_low:.2f}-{session_high:.2f}
- Today's High: {today_history['High'].max():.2f}
- Today's Low: {today_history['Low'].min():.2f}
- Technicals: 
  - RSI {intraday_data['RSI'].iloc[-1]:.1f}
  - MACD {intraday_data['MACD'].iloc[-1]:.4f}
  - SMA(5): {intraday_data['SMA_5'].iloc[-1]:.2f}
  - SMA(10): {intraday_data['SMA_10'].iloc[-1]:.2f}
- Key Levels: 
  - Support: {support:.2f}
  - Resistance: {resistance:.2f}
- Market Sentiment: {market_sentiment['score']}/10 ({market_sentiment['label']})
- Recent Patterns: {', '.join(patterns) if patterns else 'None'}

**Recent Price History (Last 15 Entries):**
{price_history}

**Prediction Requirements**
1. Momentum analysis considering price history (20 words max)
2. Recommendation: [Buy/Sell/Hold] with confidence level (High/Medium/Low)
3. Entry price range (e.g., $XXX.XX-$XXX.XX)
4. Exit price range (e.g., $XXX.XX-$XXX.XX)
5. Time-specific predictions with rationale:
   - {(timestamp + timedelta(minutes=1)).strftime('%H:%M')}: $XXX.XX (Reason: ...)
   - {(timestamp + timedelta(minutes=2)).strftime('%H:%M')}: $XXX.XX (Reason: ...)

Format strictly as:
1. [Analysis]
2. Recommendation: [Action] ([Confidence])
3. Entry Range: $XXX.XX-$XXX.XX
4. Exit Range: $XXX.XX-$XXX.XX
5. Predictions:
   - {(timestamp + timedelta(minutes=1)).strftime('%H:%M')}: $XXX.XX (Brief reason)
   - {(timestamp + timedelta(minutes=2)).strftime('%H:%M')}: $XXX.XX (Brief reason)"""

                    print("\n🤖 Generating AI Analysis...")
                    #print(f"Prompt:\n{message_content}")

                    try:
                        # Get AI response
                        response = ollama.chat(
                            model="deepseek-r1:1.5b",
                            messages=[{"role": "user", "content": message_content}]
                        )
                        print(f"\nAI Prediction:\n{response['message']['content']}")
                        
                        # Process predictions
                        predicted = re.findall(
                            r'\$\d+\.\d{2}',
                            response['message']['content']
                        )
                        if predicted:
                            predicted_prices = [float(price[1:]) for price in predicted]
                            predicted_timestamps = [
                                (timestamp + timedelta(minutes=i+1)).tz_localize(None)
                                for i in range(len(predicted_prices))
                            ]
                            new_predicted = pd.DataFrame({
                                'Datetime': predicted_timestamps,
                                'Price': predicted_prices
                            })
                            if not new_predicted.empty:
                                predicted_points = pd.concat(
                                    [predicted_points, new_predicted],
                                    ignore_index=True
                                ).astype({
                                    'Datetime': 'datetime64[ns, UTC]',
                                    'Price': 'float64'
                                })

                    except ConnectionError:
                        print("CONNECTION ERROR: Unable to reach the AI model.\nCONNECT TO OLLAMA 🦙")
                        sys.exit(1)

                # Update visualization
                print(predicted_points)
                graphing_function(intraday_data, actual_points, predicted_points, ticker_symbol)

            print("\n🕒 Waiting for next data point...")
            time.sleep(30)

    except KeyboardInterrupt:
        print("\n🛑 Tracking stopped.")
        plt.ioff()
        plt.close('all')

# ==============
# MAIN EXECUTION
# ==============
if __name__ == "__main__":
    track_stock_price("NVDA")