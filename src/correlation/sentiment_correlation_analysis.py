import pandas as pd
import numpy as np
from textblob import TextBlob
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define tickers and file paths
tickers = ['AAPL', 'AMZN', 'GOOG', 'META', 'MSFT', 'NVDA', 'TSLA']
main_path = 'D:/KAIM/week1/news-sentiment-stock-challenge/'  # Base project directory
data_path = os.path.join(main_path, 'data')  # Data folder
output_path = os.path.join(main_path, 'output')  # Output folder

# Create output folder
if not os.path.exists(output_path):
    os.makedirs(output_path)
    print(f"Created output folder: {output_path}")

# Verify output folder is writable
try:
    test_file = os.path.join(output_path, 'test_write.txt')
    with open(test_file, 'w') as f:
        f.write('test')
    os.remove(test_file)
    print(f"Output folder '{output_path}' is writable.")
except Exception as e:
    print(f"Error: Cannot write to output folder '{output_path}': {e}")
    exit()

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
    print("Sample of unique date values before parsing:")
    print(news_df['date'].unique()[:10])
    
    # Normalize dates to handle all formats
    def normalize_date(date_str):
        try:
            # Try parsing ISO 8601 with timezone
            return pd.to_datetime(date_str, utc=True).tz_convert(None)
        except:
            try:
                # Try parsing YYYY-MM-DD HH:MM:SS
                return pd.to_datetime(date_str, format='%Y-%m-%d %H:%M:%S')
            except:
                # Replace '0:00' with '12:00:00 AM' for 6/2/2020 0:00
                if '0:00' in date_str:
                    date_str = date_str.replace('0:00', '12:00:00 AM')
                try:
                    # Try parsing MM/DD/YYYY HH:MM:SS AM/PM
                    return pd.to_datetime(date_str, format='%m/%d/%Y %I:%M:%S %p')
                except:
                    # Try parsing MM/DD/YYYY HH:MM
                    return pd.to_datetime(date_str, format='%m/%d/%Y %H:%M')

    news_df['date'] = news_df['date'].apply(normalize_date)
    
    # Log invalid dates
    invalid_dates = news_df['date'].isna()
    invalid_count = invalid_dates.sum()
    if invalid_count > 0:
        print(f"Warning: {invalid_count} rows have invalid dates and will be excluded.")
        print("Sample of rows with invalid dates:")
        invalid_rows = news_df[invalid_dates][['headline', 'date', 'stock']].copy()
        invalid_rows['original_date'] = df.loc[invalid_rows.index, 'date']
        print(invalid_rows.head())
        invalid_rows.to_csv(os.path.join(output_path, 'invalid_dates.csv'), index=False)
        print(f"Invalid date rows saved to '{os.path.join(output_path, 'invalid_dates.csv')}'")
        news_df = news_df[~invalid_dates]
    
    # Debug: Print number of valid rows and parsed dates
    print(f"Number of valid news rows after date parsing: {len(news_df)}")
    print("Sample of parsed news dates:", news_df['date'].dt.date.unique()[:10])
    
    # Extract date-only as datetime64[ns]
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
        # Parse Date with multiple formats
        def parse_stock_date(date_str):
            try:
                return pd.to_datetime(date_str, format='%m/%d/%Y')
            except:
                try:
                    return pd.to_datetime(date_str, format='%Y-%m-%d')
                except:
                    return pd.to_datetime(date_str, errors='coerce')
        
        stock['Date'] = stock['Date'].apply(parse_stock_date)
        
        # Log invalid stock dates
        invalid_stock_dates = stock['Date'].isna()
        if invalid_stock_dates.sum() > 0:
            print(f"Warning: {invalid_stock_dates.sum()} invalid dates in {ticker}_historical_data.csv")
            invalid_stock_rows = stock[invalid_stock_dates][['Date']].copy()
            invalid_stock_rows['original_date'] = stock.loc[invalid_stock_rows.index, 'Date']
            print(f"Sample of invalid stock dates for {ticker}:")
            print(invalid_stock_rows.head())
            stock = stock[~invalid_stock_dates]
        
        # Extract date-only as datetime64[ns]
        stock['date_only'] = pd.to_datetime(stock['Date'].dt.date)
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
    # Aggregate sentiments: Compute average daily sentiment scores
    daily_sentiment = news_df.groupby(['stock', 'date_only'])['Sentiment'].mean().reset_index()
    
    # Debug: Print sample of daily sentiment
    print("Sample of daily sentiment data:")
    print(daily_sentiment.head())
    print(f"Total daily sentiment rows: {len(daily_sentiment)}")
    
    # Debug: Print sample of stock data
    print("Sample of stock data:")
    print(stock_df.head())
    print(f"Total stock data rows: {len(stock_df)}")
    
    # Merge with stock data
    merged_df = pd.merge(daily_sentiment, stock_df, on=['stock', 'date_only'], how='inner')
    
    # Debug: Print merged data info
    print(f"Total merged rows after joining news and stock data: {len(merged_df)}")
    if merged_df.empty:
        print("Warning: No overlapping dates between news and stock data. Check date formats or data ranges.")
        print("Unique news dates:", sorted(set(daily_sentiment['date_only'].dt.date))[:10])
        print("Unique stock dates:", sorted(set(stock_df['date_only'].dt.date))[:10])
    
    correlation_results = {}
    for ticker in stock_data.keys():
        stock_data_ticker = merged_df[merged_df['stock'] == ticker]
        
        # Debug: Print number of rows for this ticker
        print(f"Number of data points for {ticker}: {len(stock_data_ticker)}")
        
        # Save ticker data
        stock_data_ticker.to_csv(os.path.join(output_path, f'{ticker}_sentiment_stock_correlation.csv'), index=False)
        print(f"Data for {ticker} saved to '{os.path.join(output_path, f'{ticker}_sentiment_stock_correlation.csv')}'")
        
        # Generate plot if data exists
        if len(stock_data_ticker) > 0:
            # Calculate correlation: Pearson correlation coefficient
            if len(stock_data_ticker) > 1:
                correlation, p_value = pearsonr(stock_data_ticker['Sentiment'], stock_data_ticker['daily_return'].fillna(0))
            else:
                correlation, p_value = np.nan, np.nan  # Cannot compute correlation with one point
            correlation_results[ticker] = {'correlation': correlation, 'p_value': p_value}
            
            # Plot correlation
            plt.figure(figsize=(8, 6))
            sns.scatterplot(data=stock_data_ticker, x='Sentiment', y='daily_return', alpha=0.6)
            plt.title(f'{ticker} Sentiment vs Daily Returns\nCorrelation: {correlation:.4f}, P-value: {p_value:.4f}')
            plt.xlabel('Average Daily Sentiment Score')
            plt.ylabel('Daily Stock Return (%)')
            plt.gca().spines[['top', 'right']].set_visible(False)
            plt.grid(True, linestyle='--', alpha=0.7)
            plot_path = os.path.join(output_path, f'{ticker}_sentiment_correlation_plot.png')
            try:
                plt.savefig(plot_path)
                print(f"Correlation plot for {ticker} saved to '{plot_path}'")
            except Exception as e:
                print(f"Error saving plot for {ticker}: {e}")
            plt.show()
            plt.close()
        else:
            print(f"No data available for {ticker} to generate correlation plot.")
    
    return correlation_results, merged_df

# Main execution
def main():
    news_df_normalized, stock_df_normalized = normalize_stock_data(news_df, stock_data)
    stock_df_with_returns = calculate_daily_returns(stock_df_normalized)
    correlation_results, merged_df = correlation_analysis(news_df_normalized, stock_df_with_returns)
    
    # Print correlation results
    print("\nCorrelation Analysis Results:")
    if correlation_results:
        for ticker, result in correlation_results.items():
            correlation = result['correlation']
            p_value = result['p_value']
            if pd.isna(correlation):
                print(f"{ticker}: Insufficient data for correlation (less than 2 data points)")
            else:
                print(f"{ticker}: Correlation = {correlation:.4f}, P-value = {p_value:.4f}")
    else:
        print("No correlation results generated. Check data availability.")
    
    # Save overall results
    merged_df.to_csv(os.path.join(output_path, 'sentiment_stock_correlation.csv'), index=False)
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