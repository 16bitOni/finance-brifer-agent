import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, Any

# Configure logging
logger = logging.getLogger(__name__)

class EODHDTool:
    """Tool for fetching end-of-day stock data using yfinance."""
    
    def __init__(self):
        """Initialize the EODHD tool."""
        logger.info("Initializing EODHD tool with yfinance")
    
    def invoke(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fetch end-of-day stock data for a given ticker.
        
        Args:
            params: Dictionary containing:
                - ticker: Stock symbol (e.g., "AAPL", "TSCO.L", "005930.KS")
                - days: Number of days of historical data to fetch (default: 30)
        
        Returns:
            Dictionary containing:
                - success: Boolean indicating if the request was successful
                - data: Dictionary containing stock data if successful
                - error: Error message if unsuccessful
        """
        try:
            ticker = params.get("ticker")
            if not ticker:
                logger.error("No ticker provided")
                return {"success": False, "error": "No ticker provided"}
            
            days = params.get("days", 30)
            logger.info(f"Fetching EOD data for {ticker} for the last {days} days")
            
            # Define date range
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            
            # Fetch data using yfinance
            ticker_obj = yf.Ticker(ticker)
            eod_data = ticker_obj.history(start=start_date, end=end_date, interval="1d")
            
            if eod_data.empty:
                logger.warning(f"No EOD data found for {ticker}")
                return {
                    "success": False,
                    "error": f"No EOD data found for {ticker}",
                    "data": None
                }
            
            # Process the data
            eod_data = eod_data.reset_index()
            eod_data["Date"] = eod_data["Date"].apply(lambda x: x.strftime("%Y-%m-%d"))
            
            # Get the latest data point
            latest = eod_data.iloc[-1]
            
            # Calculate daily change
            if len(eod_data) > 1:
                prev_close = eod_data.iloc[-2]["Close"]
                daily_change = ((latest["Close"] - prev_close) / prev_close) * 100
            else:
                daily_change = 0
            
            # Format the response
            response = {
                "success": True,
                "data": {
                    "symbol": ticker,
                    "price": latest["Close"],
                    "open": latest["Open"],
                    "high": latest["High"],
                    "low": latest["Low"],
                    "volume": latest["Volume"],
                    "date": latest["Date"],
                    "change_percent": daily_change,
                    "historical_data": eod_data.to_dict(orient="records")
                }
            }
            
            logger.info(f"Successfully fetched EOD data for {ticker}")
            logger.debug(f"EOD data: {response}")
            return response
            
        except Exception as e:
            logger.error(f"Error fetching EOD data for {ticker}: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "data": None
            }

# Create singleton instance
eodhd_tool = EODHDTool() 