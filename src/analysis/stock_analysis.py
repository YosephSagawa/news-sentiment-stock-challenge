import pandas as pd
import numpy as np
import pynance as pn
import matplotlib.pyplot as plt
import os

def load_data(file_path):
    """Load and prepare stock data from a CSV file."""
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.dropna(inplace=True)
    return df

def calculate_rsi(data, periods=14):
    """Calculate RSI using Pandas."""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculate MACD using Pandas."""
    ema_fast = data.ewm(span=fast, adjust=False).mean()
    ema_slow = data.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - signal_line
    return macd, signal_line, hist

def calculate_indicators(df):
    """Calculate technical indicators and financial metrics using Pandas."""
    # Simple Moving Average (20-day)
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    
    # Relative Strength Index (14-day)
    df['RSI_14'] = calculate_rsi(df['Close'], periods=14)
    
    # MACD (12, 26, 9)
    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = calculate_macd(df['Close'], fast=12, slow=26, signal=9)
    
    # Financial metrics
    df['Daily_Returns'] = df['Close'].pct_change()
    df['Volatility'] = df['Daily_Returns'].rolling(window=21).std() * (252 ** 0.5)
    
    return df

def visualize_data(df, ticker, output_dir='output'):
    """Create and save visualizations for the stock data."""
    plt.figure(figsize=(14, 10))

    # Plot 1: Close Price and SMA
    plt.subplot(3, 1, 1)
    plt.plot(df.index, df['Close'], label='Close Price', color='blue')
    plt.plot(df.index, df['SMA_20'], label='20-Day SMA', color='orange')
    plt.title(f'{ticker} Close Price and 20-Day SMA')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)

    # Plot 2: RSI
    plt.subplot(3, 1, 2)
    plt.plot(df.index, df['RSI_14'], label=' RSI (14)', color='purple')
    plt.axhline(70, linestyle='--', color='red', alpha=0.5, label='Overbought (70)')
    plt.axhline(30, linestyle='--', color='green', alpha=0.5, label='Oversold (30)')
    plt.title('Relative Strength Index (RSI)')
    plt.xlabel('Date')
    plt.ylabel('RSI')
    plt.legend()
    plt.grid(True)

    # Plot 3: MACD
    plt.subplot(3, 1, 3)
    plt.plot(df.index, df['MACD'], label='MACD', color='blue')
    plt.plot(df.index, df['MACD_Signal'], label='Signal Line', color='orange')
    plt.bar(df.index, df['MACD_Hist'], label='MACD Histogram', color='gray', alpha=0.3)
    plt.title('MACD')
    plt.xlabel('Date')
    plt.ylabel('MACD')
    plt.legend()
    plt.grid(True)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot
    output_path = os.path.join(output_dir, f'{ticker}_analysis.png')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path

def analyze_stock(file_path, ticker, output_dir='output'):
    """Run full analysis for a single stock."""
    df = load_data(file_path)
    df = calculate_indicators(df)
    output_path = visualize_data(df, ticker, output_dir)
    return df, output_path