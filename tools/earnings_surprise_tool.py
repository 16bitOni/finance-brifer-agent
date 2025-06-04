import os
from typing import Dict, Any, Optional
import yfinance as yf
import pandas as pd
from datetime import datetime
from .base_tool import BaseTool, ToolResult

class EarningsSurpriseTool(BaseTool):
    """
    Tool for fetching earnings surprise data including EPS actual vs estimate and surprise percentage.
    Uses yfinance as the data source for global stock coverage.
    """
    
    def __init__(self):
        super().__init__()
    
    async def execute(self, ticker: str, lookback_periods: int = 4) -> ToolResult:
        """
        Execute the earnings surprise tool
        
        Args:
            ticker: Stock ticker symbol to fetch earnings for
            lookback_periods: Number of past earnings periods to fetch (default: 4)
        
        Returns:
            ToolResult with earnings surprise data including:
            - EPS actual
            - EPS estimate
            - Surprise percentage
            - Date
        """
        try:
            if not ticker:
                return ToolResult(
                    success=False,
                    error="Ticker symbol is required"
                )
            
            # Get earnings history from yfinance
            ticker_obj = yf.Ticker(ticker)
            earnings = ticker_obj.earnings_history
            
            if earnings is not None and not earnings.empty:
                # Get shares outstanding for EPS calculation
                shares_outstanding = ticker_obj.info.get('sharesOutstanding', 1_000_000_000)  # Default to 1B if unavailable
                
                # Convert to EPS and format data
                earnings_data = []
                for idx, row in earnings.head(lookback_periods).iterrows():
                    # Convert quarter-end to estimated announcement date (add ~30 days)
                    announcement_date = (pd.to_datetime(idx) + pd.Timedelta(days=30)).strftime("%Y-%m-%d")
                    
                    # Calculate EPS values
                    actual_eps = row['epsActual'] / shares_outstanding
                    estimated_eps = row['epsEstimate'] / shares_outstanding
                    
                    # Calculate surprise percentage
                    surprise_percent = ((actual_eps - estimated_eps) / abs(estimated_eps)) * 100
                    
                    earnings_data.append({
                        "date": announcement_date,
                        "symbol": ticker,
                        "actualEarningResult": actual_eps,
                        "estimatedEarning": estimated_eps,
                        "surprisePercent": surprise_percent
                    })
                
                return ToolResult(
                    success=True,
                    data={
                        "ticker": ticker,
                        "earnings": earnings_data
                    }
                )
            else:
                error_msg = f"No earnings data available for {ticker}"
                print(error_msg)
                return ToolResult(
                    success=False,
                    error=error_msg
                )
                
        except Exception as e:
            error_msg = f"Error fetching earnings for {ticker}: {str(e)}"
            print(error_msg)
            return ToolResult(
                success=False,
                error=error_msg
            )
    
    def format_result(self, data: Dict[str, Any]) -> str:
        """Format earnings surprise data for display"""
        ticker = data["ticker"]
        earnings = data["earnings"]
        
        if not earnings:
            return f"No earnings data available for {ticker}"
        
        # Format the earnings information
        info = [f"Earnings Surprises for {ticker}:"]
        
        for period in earnings:
            date = period.get("date", "N/A")
            actual = period.get("actualEarningResult", 0)
            estimated = period.get("estimatedEarning", 0)
            surprise = period.get("surprisePercent", 0)
            
            info.append(
                f"\nPeriod: {date}\n" +
                f"Actual EPS: ${actual:.2f}\n" +
                f"Estimated EPS: ${estimated:.2f}\n" +
                f"Surprise: {surprise:.2f}%"
            )
        
        return "\n".join(info)

# Create singleton instance
earnings_surprise_tool = EarningsSurpriseTool() 