import pandas as pd
import numpy as np
from textblob import TextBlob
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

# Define tickers and file paths
tickers = ['AAPL', 'AMZN', 'GOOG', 'META', 'MSFT', 'NVDA', 'TSLA']
main_path = 'D:/KAIM/week1/news-sentiment-stock-challenge/'  # Base project directory
data_path = os.path.join(main_path, 'data')  # Data folder
output_path = os.path.join(main_path, 'output')  # Output folder

# Create output folder
if not os.path.exists(output_path):
    os.makedirs(output_path)
    print(f"Created output folder: {output_path}")

# Load financial news dataset
try:
    df = pd.read_csv(os.path.join(data_path, 'raw_analyst_ratings.csv'), encoding='utf-8', engine='python')
    print("Financial news data loaded successfully.")
    print(df.head())
except FileNotFoundError:
    print(f"Error: 'raw_analyst_ratings.csv' not found in '{data_path}'")
    exit()
except Exception as e:
    print(f"Error loading financial news data: {e}")
    exit()

# Check and rename columns
if 'Date' in df.columns and 'date' not in df.columns:
    df.rename(columns={'Date': 'date'}, inplace=True)
if 'Symbol' in df.columns and 'stock' not in df.columns:
    df.rename(columns={'Symbol': 'stock'}, inplace=True)
if 'headline' not in df.columns or 'date' not in df.columns or 'stock' not in df.columns:
    print("Error: Required columns ('headline', 'date', 'stock') not found in raw_analyst_ratings.csv")
    exit()

# Filter news data by tickers
initial_row_count = len(df)
news_df = df[df['stock'].isin(tickers)].copy()  # Create a copy to avoid SettingWithCopyWarning
filtered_row_count = len(news_df)
print(f"Filtered news data to {filtered_row_count} rows for tickers {tickers} (removed {initial_row_count - filtered_row_count} rows).")

# Load stock data from CSVs
def load_stock_data(tickers, data_path):
    stock_data = {}
    for ticker in tickers:
        filename = os.path.join(data_path, f'{ticker}_historical_data.csv')
        try:
            stock = pd.read_csv(filename)
            stock_data[ticker] = stock
            print(f"{filename} loaded successfully.")
        except FileNotFoundError:
            print(f"Error: {filename} not found.")
    return stock_data

stock_data = load_stock_data(tickers, data_path)
if not stock_data:
    print("No stock data loaded. Exiting.")
    exit()

# Sentiment Analysis
def calculate_sentiment(text):
    return TextBlob(str(text)).sentiment.polarity

# Normalize dates and perform sentiment analysis
try:
    # Pre-clean invalid date entries
    pre_clean_count = len(news_df)
    # Convert date column to string, strip whitespace, and handle NaN
    news_df.loc[:, 'date'] = news_df['date'].astype(str).str.strip().replace(['nan', 'NaN', ''], np.nan)
    
    # Debug: Print sample of unique date values
    print("Sample of unique date values before filtering:")
    print(news_df['date'].unique()[:10])
    
    # Filter out ISO 8601 format dates (YYYY-MM-DD HH:MM:SS±HH:MM)
    iso8601_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}[-+]\d{2}:\d{2}$')
    iso8601_rows = news_df[news_df['date'].str.match(iso8601_pattern, na=False)]
    if not iso8601_rows.empty:
        print(f"Filtering out {len(iso8601_rows)} rows with ISO 8601 format (e.g., '2020-06-10 11:33:26-04:00').")
        iso8601_rows[['headline', 'date', 'stock']].to_csv(os.path.join(output_path, 'iso8601_date_rows.csv'), index=False)
        print(f"ISO 8601 date rows saved to '{os.path.join(output_path, 'iso8601_date_rows.csv')}'")
        news_df = news_df[~news_df['date'].str.match(iso8601_pattern, na=False)]
    
    # Filter out NaN dates
    news_df = news_df[news_df['date'].notna()]
    if len(news_df) < pre_clean_count:
        print(f"Removed {pre_clean_count - len(news_df)} rows with invalid date values (e.g., empty strings, NaN, ISO 8601).")
    
    # Debug: Print sample of unique date values after filtering
    print("Sample of unique date values after filtering ISO 8601:")
    print(news_df['date'].unique()[:10])
    
    # Normalize dates to handle 2020-05-28 00:00:00 and 6/2/2020 0:00
    def normalize_date(date_str):
        try:
            # Try parsing YYYY-MM-DD HH:MM:SS
            return pd.to_datetime(date_str, format='%Y-%m-%d %H:%M:%S', errors='coerce')
        except ValueError:
            # Replace '0:00' with '12:00:00 AM' for 6/2/2020 0:00
            if '0:00' in date_str:
                date_str = date_str.replace('0:00', '12:00:00 AM')
            try:
                # Try parsing MM/DD/YYYY HH:MM:SS AM/PM
                return pd.to_datetime(date_str, format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
            except ValueError:
                # Try parsing MM/DD/YYYY HH:MM
                return pd.to_datetime(date_str, format='%m/%d/%Y %H:%M', errors='coerce')

    news_df['date'] = news_df['date'].apply(normalize_date)
    
    # Log invalid dates with original date values
    invalid_dates = news_df['date'].isna()
    invalid_count = invalid_dates.sum()
    if invalid_count > 0:
        print(f"Warning: {invalid_count} rows have invalid dates and will be excluded.")
        print("Sample of rows with invalid dates:")
        invalid_rows = news_df[invalid_dates][['headline', 'date', 'stock']].copy()
        invalid_rows['original_date'] = df.loc[invalid_rows.index, 'date']  # Preserve original date
        print(invalid_rows.head())
        invalid_rows.to_csv(os.path.join(output_path, 'invalid_dates.csv'), index=False)
        print(f"Invalid date rows saved to '{os.path.join(output_path, 'invalid_dates.csv')}'")
        # Exclude invalid dates
        news_df = news_df[~invalid_dates]
    
    # Initialize date_only as datetime64[ns]
    news_df['date_only'] = pd.to_datetime(news_df['date'].dt.date)
    
    # Perform sentiment analysis
    news_df['Sentiment'] = news_df['headline'].apply(calculate_sentiment).astype(float)
    print("Sentiment analysis completed.")
except KeyError as e:
    print(f"Error: Required column not found. {e}")
    exit()
except Exception as e:
    print(f"Error during date normalization or sentiment analysis: {e}")
    exit()

# Normalize stock data
def normalize_stock_data(news_df, stock_data):
    stock_dfs = []
    for ticker, stock in stock_data.items():
        stock = stock.copy()
        stock['stock'] = ticker
        # Parse dates for historical_data.csv (format: 7/2/2010)
        stock['date_only'] = pd.to_datetime(stock['Date'], format='%m/%d/%Y', errors='coerce')
        stock_dfs.append(stock[['date_only', 'stock', 'Close']])
    
    stock_df = pd.concat(stock_dfs, ignore_index=True)
    return news_df, stock_df

# Calculate Daily Stock Returns
def calculate_daily_returns(stock_df):
    stock_df = stock_df.sort_values(['stock', 'date_only'])
    stock_df['daily_return'] = stock_df.groupby('stock')['Close'].pct_change() * 100
    return stock_df

# Correlation Analysis
def correlation_analysis(news_df, stock_df):
    daily_sentiment = news_df.groupby(['stock', 'date_only'])['Sentiment'].mean().reset_index()
    merged_df = pd.merge(daily_sentiment, stock_df, on=['stock', 'date_only'], how='inner')
    
    correlation_results = {}
    for ticker in stock_data.keys():
        stock_data_ticker = merged_df[merged_df['stock'] == ticker]
        stock_data_ticker.to_csv(os.path.join(output_path, f'{ticker}_sentiment_stock_correlation.csv'), index=False)
        if len(stock_data_ticker) > 1:
            correlation, p_value = pearsonr(stock_data_ticker['Sentiment'], stock_data_ticker['daily_return'].fillna(0))
            correlation_results[ticker] = {'correlation': correlation, 'p_value': p_value}
    
    return correlation_results, merged_df

# Main execution
def main():
    news_df_normalized, stock_df_normalized = normalize_stock_data(news_df, stock_data)
    stock_df_with_returns = calculate_daily_returns(stock_df_normalized)
    correlation_results, merged_data = correlation_analysis(news_df_normalized, stock_df_with_returns)
    
    # Print correlation results
    print("\nCorrelation Analysis Results:")
    for ticker, result in correlation_results.items():
        print(f"{ticker}: Correlation = {result['correlation']:.4f}, P-value = {result['p_value']:.4f}")
    
    # Save overall results
    merged_data.to_csv(os.path.join(output_path, 'sentiment_stock_correlation.csv'), index=False)
    print(f"Overall results saved to '{os.path.join(output_path, 'sentiment_stock_correlation.csv')}'")
    print(f"Individual stock results saved as '{os.path.join(output_path, '[ticker]_sentiment_stock_correlation.csv')}'")

    # Descriptive statistics and visualizations
    print("\nDescriptive statistics for 'headline' column:")
    print(news_df['headline'].describe())
    
    print("\nMissing values in 'headline' column:")
    print(news_df['headline'].isnull().sum())
    
    print("\nNumber of unique headlines:")
    print(news_df['headline'].nunique())
    
    print("\nMost frequent headlines:")
    print(news_df['headline'].value_counts().head())
    
    print("\nSentiment distribution:")
    print(news_df['Sentiment'].describe())
    
    # Save histogram
    plt.figure(figsize=(8, 4))
    if not news_df['Sentiment'].empty:
        sns.histplot(news_df['Sentiment'], bins=20, kde=True)
        plt.title('Distribution of Sentiment Scores for Headlines')
        plt.xlabel('Sentiment Polarity')
        plt.ylabel('Frequency')
        plt.gca().spines[['top', 'right']].set_visible(False)
        plt.savefig(os.path.join(output_path, 'sentiment_histogram.png'))
        plt.show()
    else:
        print("Warning: No data available for histogram.")
    
    # Handle empty dataframe for nlargest/nsmallest
    if not news_df.empty:
        most_positive = news_df.nlargest(5, 'Sentiment')
        print("\nMost Positive Headlines:")
        print(most_positive[['headline', 'Sentiment', 'stock']])
        
        most_negative = news_df.nsmallest(5, 'Sentiment')
        print("\nMost Negative Headlines:")
        print(most_negative[['headline', 'Sentiment', 'stock']])
    else:
        print("\nWarning: No data available for sentiment analysis.")

if __name__ == "__main__":
    main()