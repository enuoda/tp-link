#! /bin/bash

# Sam Dawley
# 08/2025

import os
import websocket

import pandas as pd

from dotenv import load_dotenv

# import finnhub
# from finnhub.exceptions import FinnhubAPIException

import alpaca

class Market:
    """
    Class for storing and manipulating market data

    Finnhub utilizes JSON files and so nearly all arguments and
    return values of the finnhub class methods are strings

    References:
    -----------
        - List of supported exchange codes:
            https://docs.google.com/spreadsheets/d/1I3pBxjfXB056-g_JYf_6o3Rns3BV2kMGG1nCatb91ls/edit?pli=1&gid=0#gid=0
    """

    # ----- authentication -----

    # load_dotenv()
    # finn_api_key = os.getenv("FINNHUB_API_KEY")
    # finn_api_secret_key = os.getenv("FINNHUB_SECRET_KEY")

    # ==================================================
    # INITIALIZATION
    # ==================================================
    
    def __init__(self) -> None:

        # self.client = finnhub.Client(api_key=self.finn_api_key)
        
        return
    

    def __repr__(self) -> str:
        return f""
    
    # ==============================
    # DEPARTMENT OF THE INTERIOR
    # ==============================
    def _get_stock_symbols(self, exchange: str, mic: str = None, security_type: str = None, currency: str = None) -> list:
        """
        Returns:
        --------
            : list
                list of supported stocks
        
        References:
        -----------
            For attributes of return value, see 'Stock Symbol':
                https://finnhub.io/docs/api/stock-symbols
        """
        # return self.client.stock_symbols(exchange, mic=mic, security_type=security_type, currency=currency)
        return
    
    # ==============================
    # DEPARTMENT OF THE EXTERIOR
    # ==============================

    def get_company_profile(self, symbol: str, isin: str = None, cusip: str = None) -> None:
        """
        DOCSTRING
        """
        # try:
        #     return self.client.company_profile(symbol=symbol, isin=isin, cusip=cusip)

        # except FinnhubAPIException:
        #     return self.client.company_profile2(symbol=symbol, isin=isin, cusip=cusip)
        return

    def stream_trades(self) -> None:
        """
        DOCSTRING
        """
        return
    
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def collect_google_data(start_date, end_date, interval='1d'):
    """
    Collect Google (Alphabet) stock data using yfinance
    
    Parameters:
    start_date (str): Start date in 'YYYY-MM-DD' format
    end_date (str): End date in 'YYYY-MM-DD' format
    interval (str): Data interval - '1d', '1h', '5m', etc. Default is '1d'
    
    Returns:
    pandas.DataFrame: Stock data with OHLCV information
    """
    
    # Google's ticker symbol is GOOGL (Class A) or GOOG (Class C)
    # Using GOOGL as it's more commonly traded
    ticker = "GOOGL"
    
    try:
        # Create ticker object
        google = yf.Ticker(ticker)
        
        # Download historical data
        print(f"Collecting {ticker} data from {start_date} to {end_date}...")
        data = google.history(start=start_date, end=end_date, interval=interval)
        
        if data.empty:
            print("No data found for the specified date range.")
            return None
        
        # Display basic info about the collected data
        print(f"\nData collection completed!")
        print(f"Number of records: {len(data)}")
        print(f"Date range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
        print(f"\nColumn information:")
        print(f"- Open: Opening price")
        print(f"- High: Highest price of the period")
        print(f"- Low: Lowest price of the period") 
        print(f"- Close: Closing price")
        print(f"- Volume: Number of shares traded")
        print(f"- Dividends: Dividend payments")
        print(f"- Stock Splits: Stock split information")
        
        return data
        
    except Exception as e:
        print(f"Error collecting data: {e}")
        return None

def save_data_to_csv(data, filename=None):
    """
    Save the collected data to a CSV file
    
    Parameters:
    data (pandas.DataFrame): The stock data to save
    filename (str): Optional filename. If None, uses timestamp
    """
    if data is None or data.empty:
        print("No data to save.")
        return
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"GOOGL_data_{timestamp}.csv"
    
    try:
        data.to_csv(filename)
        print(f"Data saved to {filename}")
    except Exception as e:
        print(f"Error saving data: {e}")

def display_sample_data(data, num_rows=5):
    """
    Display a sample of the collected data
    
    Parameters:
    data (pandas.DataFrame): The stock data
    num_rows (int): Number of rows to display from start and end
    """
    if data is None or data.empty:
        print("No data to display.")
        return
    
    print(f"\nFirst {num_rows} records:")
    print(data.head(num_rows))
    
    if len(data) > num_rows * 2:
        print(f"\nLast {num_rows} records:")
        print(data.tail(num_rows))
    
    # Basic statistics
    print(f"\nBasic Statistics for Closing Prices:")
    print(data['Close'].describe())


# Additional utility functions
def get_company_info():
    """Get basic company information for Google"""
    ticker = yf.Ticker("GOOGL")
    info = ticker.info
    
    print("Google (Alphabet Inc.) Company Information:")
    print(f"Name: {info.get('longName', 'N/A')}")
    print(f"Sector: {info.get('sector', 'N/A')}")
    print(f"Industry: {info.get('industry', 'N/A')}")
    print(f"Market Cap: ${info.get('marketCap', 'N/A'):,}" if info.get('marketCap') else "Market Cap: N/A")
    print(f"Current Price: ${info.get('currentPrice', 'N/A')}")
    
def analyze_price_movement(data):
    """Simple analysis of price movements"""
    if data is None or data.empty:
        return
    
    # Calculate daily returns
    data['Daily_Return'] = data['Close'].pct_change()
    
    # Calculate moving averages
    data['MA_10'] = data['Close'].rolling(window=10).mean()
    data['MA_50'] = data['Close'].rolling(window=50).mean()
    
    print("\nPrice Movement Analysis:")
    print(f"Average Daily Return: {data['Daily_Return'].mean():.4f} ({data['Daily_Return'].mean()*100:.2f}%)")
    print(f"Volatility (Std Dev): {data['Daily_Return'].std():.4f} ({data['Daily_Return'].std()*100:.2f}%)")
    print(f"Best Day: {data['Daily_Return'].max():.4f} ({data['Daily_Return'].max()*100:.2f}%)")
    print(f"Worst Day: {data['Daily_Return'].min():.4f} ({data['Daily_Return'].min()*100:.2f}%)")
    
    return data

if __name__ == "__main__":
    # Example 1: Get last 30 days of data
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    print("Example 1: Last 30 days of Google stock data")
    data = collect_google_data(start_date, end_date)
    
    if data is not None:
        display_sample_data(data)
        
        # Uncomment the line below to save data to CSV
        # save_data_to_csv(data)
    
    print("\n" + "="*50 + "\n")
    
    # Example 2: Custom date range
    print("Example 2: Custom date range (2023-01-01 to 2023-12-31)")
    custom_data = collect_google_data('2023-01-01', '2023-12-31')
    
    if custom_data is not None:
        display_sample_data(custom_data, num_rows=3)
    
    print("\n" + "="*50 + "\n")
    
    # Example 3: Hourly data for last 5 days (during market hours)
    print("Example 3: Hourly data for last 5 days")
    start_date_hourly = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
    hourly_data = collect_google_data(start_date_hourly, end_date, interval='1h')
    
    if hourly_data is not None:
        display_sample_data(hourly_data, num_rows=3)