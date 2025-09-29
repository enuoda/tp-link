# #!/bin/bash
# # Sam Dawley
# # 09/2025

# from datetime import datetime, timedelta
# from pathlib import Path
# import os
# import requests
# import time

# import pandas as pd


# class PolygonStockData:
#     """
#     Class to collect stock data using Polygon.io API
#     """

#     def __init__(self, api_key: str = None, base_url: str = None) -> None:
#         """
#         Initialize with your Polygon API key
#         Get free API key from: https://polygon.io/
#         """

#         if isinstance(api_key, type(None)):
#             try:
#                 self.api_key = os.getenv("POLYGON_API_KEY")
#             except Exception as e:
#                 print(f"Caught Exception trying to load '.env': {e}", flush=True)
#         else:
#             self.api_key = api_key

#         if isinstance(base_url, type(None)):
#             self.base_url = base_url
#         else:
#             self.base_url = "https://api.polygon.io"

#         self.session = requests.Session()

#         return

#     def validate_and_authenticate(self):
#         # 1. Run authentication checks
#         auth_result = self.check_authentication()

#         # 2. Print formatted status report
#         success = self.print_authentication_status(auth_result)

#         # 3. Update instance state if successful
#         if success:
#             self.authenticated = True
#             self.account_info = auth_result.get("account_info")

#         # 4. Return success/failure
#         return success

#     def get_stock_bars(
#         self,
#         ticker: str,
#         start_date: str,
#         end_date: str,
#         timespan: str = "day",
#         multiplier=1,
#         limit=5000,
#     ):
#         """
#         Get aggregated stock data (bars) for a ticker

#         Parameters:
#         -----------
#             ticker (str): Stock ticker symbol (e.g., 'GOOGL')
#             start_date (str): Start date in 'YYYY-MM-DD' format
#             end_date (str): End date in 'YYYY-MM-DD' format
#             timespan (str): Size of the time window - 'minute', 'hour', 'day', 'week', 'month', 'quarter', 'year'
#             multiplier (int): Size of the timespan multiplier (e.g., 1 day, 5 minutes, etc.)
#             limit (int): Limit the number of base aggregates (max 50,000 for paid, 120 for free)

#         Returns:
#         --------
#             pandas.DataFrame: Stock data with OHLCV information
#         """

#         # Format the endpoint
#         endpoint = f"/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{start_date}/{end_date}"

#         # Parameters for the request
#         params = {
#             "adjusted": "true",  # Adjust for splits and dividends
#             "sort": "asc",  # Sort by timestamp ascending
#             "limit": limit,  # Limit results
#             "apikey": self.api_key,
#         }

#         try:
#             print(f"Fetching {ticker} data from {start_date} to {end_date}...")

#             # Make the API request
#             url = self.base_url + endpoint
#             response = self.session.get(url, params=params)

#             # Check for rate limiting
#             if response.status_code == 429:
#                 print("Rate limit hit. Waiting 60 seconds...")
#                 time.sleep(60)
#                 response = self.session.get(url, params=params)

#             response.raise_for_status()
#             data = response.json()

#             # Check if we have results
#             if data.get("status") != "OK":
#                 print(f"API Error: {data.get('status', 'Unknown error')}")
#                 if "error" in data:
#                     print(f"Error details: {data['error']}")
#                 return None

#             if "results" not in data or not data["results"]:
#                 print("No data found for the specified parameters.")
#                 return None

#             # Convert to DataFrame
#             df = self._format_bars_data(data["results"])

#             print(f"Successfully collected {len(df)} records")
#             print(
#                 f"Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}"
#             )

#             # Print API usage info if available
#             if "request_id" in data:
#                 print(f"Request ID: {data['request_id']}", flush=True)

#             return df

#         except requests.exceptions.RequestException as e:
#             print(f"Request error: {e}", flush=True)
#             return None

#         except Exception as e:
#             print(f"Error processing data: {e}", flush=True)
#             return None

#     def _format_bars_data(self, results):
#         """
#         Convert Polygon API results to pandas DataFrame
#         """
#         # Create DataFrame from results
#         df = pd.DataFrame(results)

#         # Convert timestamp (milliseconds) to datetime
#         df["datetime"] = pd.to_datetime(df["t"], unit="ms")
#         df.set_index("datetime", inplace=True)

#         # Rename columns to standard OHLCV format
#         column_mapping = {
#             "o": "Open",
#             "h": "High",
#             "l": "Low",
#             "c": "Close",
#             "v": "Volume",
#             "vw": "VWAP",  # Volume Weighted Average Price
#             "n": "Transactions",  # Number of transactions
#         }

#         df.rename(columns=column_mapping, inplace=True)

#         # Keep only the columns we want
#         columns_to_keep = ["Open", "High", "Low", "Close", "Volume"]

#         if "VWAP" in df.columns:
#             columns_to_keep.append("VWAP")

#         if "Transactions" in df.columns:
#             columns_to_keep.append("Transactions")

#         df = df[columns_to_keep]

#         return df

#     def get_ticker_details(self, ticker: str) -> pd.Series | None:
#         """
#         Get detailed information about a ticker
#         """
#         endpoint = f"/v3/reference/tickers/{ticker}"
#         params = {"apikey": self.api_key}

#         try:
#             url = self.base_url + endpoint
#             response = self.session.get(url, params=params)
#             response.raise_for_status()
#             data = response.json()

#             if data.get("status") == "OK" and "results" in data:
#                 return data["results"]
#             else:
#                 print(
#                     f"Could not fetch ticker details: {data.get('status', 'Unknown error')}"
#                 )
#                 return None

#         except Exception as e:
#             print(f"Error fetching ticker details: {e}")
#             return None

#     def get_previous_close(self, ticker: str):
#         """
#         Get the previous trading day's close price
#         """
#         endpoint = f"/v2/aggs/ticker/{ticker}/prev"
#         params = {"adjusted": "true", "apikey": self.api_key}

#         try:
#             url = self.base_url + endpoint
#             response = self.session.get(url, params=params)
#             response.raise_for_status()
#             data = response.json()

#             if data.get("status") == "OK" and "results" in data and data["results"]:
#                 result = data["results"][0]
#                 return {
#                     "date": pd.to_datetime(result["T"]).strftime("%Y-%m-%d"),
#                     "close": result["c"],
#                     "volume": result["v"],
#                     "open": result["o"],
#                     "high": result["h"],
#                     "low": result["l"],
#                 }
#             else:
#                 print("Could not fetch previous close data")
#                 return None

#         except Exception as e:
#             print(f"Error fetching previous close: {e}")
#             return None


# def collect_google_data_polygon(
#     start_date: str = None, end_date: str = None, timespan: str = "day"
# ):
#     """
#     Main function to collect Google stock data using Polygon API
#     """

#     # Default to last 30 days if no dates provided
#     if end_date is None:
#         end_date = datetime.now().strftime("%Y-%m-%d")

#     if start_date is None:
#         start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

#     # Initialize Polygon client
#     polygon_client = PolygonStockData()

#     # Get Google stock data (using GOOGL)
#     print("=== Collecting Google (Alphabet) Stock Data ===")
#     print(f"Ticker: GOOGL")
#     print(f"Date Range: {start_date} to {end_date}")
#     print(f"Timespan: {timespan}")
#     print()

#     # Get stock bars
#     data = polygon_client.get_stock_bars("GOOGL", start_date, end_date, timespan=timespan)

#     if data is not None:
#         print("\n=== Data Summary ===")
#         print(f"Records collected: {len(data)}")
#         print(f"Columns: {list(data.columns)}")
#         print(f"Date range: {data.index[0]} to {data.index[-1]}")

#         # Display basic statistics
#         print(f"\n=== Price Statistics ===")
#         print(f"Latest Close: ${data['Close'].iloc[-1]:.2f}")
#         print(f"Period High: ${data['High'].max():.2f}")
#         print(f"Period Low: ${data['Low'].min():.2f}")
#         print(f"Average Volume: {data['Volume'].mean():,.0f}")

#         # Get ticker details
#         print(f"\n=== Company Information ===")
#         details = polygon_client.get_ticker_details("GOOGL")
#         if details:
#             print(f"Company: {details.get('name', 'N/A')}")
#             print(f"Market: {details.get('market', 'N/A')}")
#             print(f"Primary Exchange: {details.get('primary_exchange', 'N/A')}")
#             print(f"Type: {details.get('type', 'N/A')}")
#             print(f"Currency: {details.get('currency_name', 'N/A')}")

#         # Get previous close
#         print(f"\n=== Previous Trading Day ===")
#         prev_close = polygon_client.get_previous_close("GOOGL")
#         if prev_close:
#             print(f"Date: {prev_close['date']}")
#             print(f"Close: ${prev_close['close']:.2f}")
#             print(f"Volume: {prev_close['volume']:,}")

#         return data
#     else:
#         print("❌ Failed to collect data. Please check your API key and parameters.")
#         return None


# def display_sample_data(data, num_rows: int = 5) -> None:
#     """
#     Display sample of the collected data
#     """
#     if data is None or data.empty:
#         print("No data to display.")
#         return

#     print(f"\n=== First {num_rows} Records ===")
#     print(data.head(num_rows).round(2))

#     if len(data) > num_rows * 2:
#         print(f"\n=== Last {num_rows} Records ===")
#         print(data.tail(num_rows).round(2))

#     return


# def save_to_csv(data, filename: Path = None) -> None:
#     """
#     Save data to CSV file
#     """
#     if data is None or data.empty:
#         print("No data to save.", flush=True)
#         return

#     if filename is None:
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         filename = f"GOOGL_polygon_data_{timestamp}.csv"

#     try:
#         data.to_csv(filename)
#         print(f"\nData saved to: {filename}", flush=True)

#     except Exception as e:
#         print(f"Error saving data: {e}", flush=True)

#     return


# # Example usage
# def main():
#     # print("Polygon.io API Setup Required")
#     # print("=" * 40)
#     # print("1. Go to: https://polygon.io/")
#     # print("2. Sign up for a free account")
#     # print("3. Get your API key from the dashboard")
#     # print("4. Replace 'YOUR_API_KEY' below with your actual key")
#     # print()

#     try:
#         # Example 1: Default - last 30 days of daily data
#         print("📈 Example 1: Last 30 days of Google stock data")
#         data = collect_google_data_polygon()

#         if data is not None:
#             display_sample_data(data)

#         print("\n" + "=" * 60 + "\n")

#         save_to_csv(data)

#         # Example 2: Custom date range
#         print("📈 Example 2: Custom date range (last 7 days)")
#         end_date = datetime.now().strftime("%Y-%m-%d")
#         start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

#         weekly_data = collect_google_data_polygon(
#             start_date=start_date, end_date=end_date
#         )

#         if weekly_data is not None:
#             display_sample_data(weekly_data, num_rows=3)

#         print("\n" + "=" * 60 + "\n")

#         # Example 3: Hourly data (requires paid plan for historical data)
#         print("📈 Example 3: Hourly data (last 3 days)")
#         print("Note: Free tier may have limited access to intraday historical data")

#         hourly_start = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
#         hourly_data = collect_google_data_polygon(
#             start_date=hourly_start, end_date=end_date, timespan="hour"
#         )

#         if hourly_data is not None:
#             display_sample_data(hourly_data, num_rows=3)

#     except Exception as e:
#         print(f"Error in main execution: {e}")

#     return


# # ==================================================
# # Utility Functions
# # ==================================================


# def test_authentication_only():
#     """
#     Standalone function to test authentication only
#     """
#     print("🔐 Testing Polygon API Authentication Only")
#     print("=" * 45)

#     polygon_client = PolygonStockData()
#     success = polygon_client.validate_and_authenticate()

#     if success:
#         print("\n🎉 Success! Your API is ready to use.")
#         print("\n💡 You can now:")
#         print("• Collect daily stock data")
#         print("• Get company information")
#         print("• Access previous close data")
#         if polygon_client.account_info:
#             tier = polygon_client.account_info.get("tier", "Unknown")
#             if "Free" in tier:
#                 print("• Limited to 5 requests/minute (free tier)")
#             print("• Consider upgrading for more features and higher limits")
#     else:
#         print("\n❌ Authentication failed.")
#         print("Please check your API key and try again.")

#     return success


# def compare_timeframes(api_key, ticker="GOOGL", days_back=30):
#     """
#     Compare different timeframes for analysis
#     """
#     polygon_client = PolygonStockData(api_key)
#     end_date = datetime.now().strftime("%Y-%m-%d")
#     start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

#     timeframes = ["day", "hour"]  # minute requires paid plan for historical
#     results = {}

#     for timeframe in timeframes:
#         print(f"\nCollecting {timeframe}ly data...")
#         data = polygon_client.get_stock_bars(
#             ticker, start_date, end_date, timespan=timeframe
#         )
#         if data is not None:
#             results[timeframe] = data
#             print(f"{timeframe.capitalize()}ly data: {len(data)} records")
#         else:
#             print(f"Failed to collect {timeframe}ly data")

#     return results


# def calculate_basic_metrics(data):
#     """
#     Calculate basic trading metrics
#     """
#     if data is None or data.empty:
#         return None

#     # Calculate returns
#     data["Daily_Return"] = data["Close"].pct_change()

#     # Calculate moving averages
#     data["MA_5"] = data["Close"].rolling(window=5).mean()
#     data["MA_20"] = data["Close"].rolling(window=20).mean()

#     # Calculate volatility
#     volatility = data["Daily_Return"].std() * (252**0.5)  # Annualized

#     metrics = {
#         "total_return": ((data["Close"].iloc[-1] / data["Close"].iloc[0]) - 1) * 100,
#         "avg_daily_return": data["Daily_Return"].mean() * 100,
#         "volatility": volatility * 100,
#         "max_daily_gain": data["Daily_Return"].max() * 100,
#         "max_daily_loss": data["Daily_Return"].min() * 100,
#         "avg_volume": data["Volume"].mean(),
#     }

#     print("\n=== Trading Metrics ===")
#     print(f"Total Return: {metrics['total_return']:.2f}%")
#     print(f"Avg Daily Return: {metrics['avg_daily_return']:.2f}%")
#     print(f"Annualized Volatility: {metrics['volatility']:.2f}%")
#     print(f"Best Day: {metrics['max_daily_gain']:.2f}%")
#     print(f"Worst Day: {metrics['max_daily_loss']:.2f}%")
#     print(f"Avg Volume: {metrics['avg_volume']:,.0f}")

#     return metrics


# if __name__ == "__main__":
#     test_authentication_only()
#     main()
