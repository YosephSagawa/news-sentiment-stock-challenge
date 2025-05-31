import os
from analysis.stock_analysis import analyze_stock

# Define stock files and tickers
stock_files = {
    'AAPL': 'data/AAPL_historical_data.csv',
    'AMZN': 'data/AMZN_historical_data.csv',
    'GOOG': 'data/GOOG_historical_data.csv',
    'META': 'data/META_historical_data.csv',
    'MSFT': 'data/MSFT_historical_data.csv',
    'NVDA': 'data/NVDA_historical_data.csv',
    'TSLA': 'data/TSLA_historical_data.csv'
}

# Run analysis for each stock
for ticker, file_path in stock_files.items():
    if os.path.exists(file_path):
        print(f"Analyzing {ticker}...")
        df, output_path = analyze_stock(file_path, ticker, output_dir='output')
        print(f"Saved plot for {ticker} at {output_path}")
    else:
        print(f"File not found: {file_path}")