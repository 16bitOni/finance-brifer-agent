import os
from typing import Dict, Any, Optional
import yfinance as yf
from .base_tool import BaseTool, ToolResult
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CompanyMetadataTool(BaseTool):
    """
    Tool for fetching company metadata including sector, industry, region, and market cap.
    Uses yfinance as the data source for global stock coverage.
    """
    
    def __init__(self):
        super().__init__()
        logger.info("Initializing CompanyMetadataTool")
    
    def execute(self, ticker: str) -> ToolResult:
        """
        Execute the company metadata tool
        
        Args:
            ticker: Stock ticker symbol to fetch metadata for
        
        Returns:
            ToolResult with company metadata including:
            - sector
            - industry
            - country
            - market cap
            - currency
            - exchange
        """
        try:
            if not ticker:
                logger.error("No ticker symbol provided")
                return ToolResult(
                    success=False,
                    error="Ticker symbol is required"
                )
            
            logger.info(f"Fetching metadata for ticker: {ticker}")
            
            # Get company info from yfinance
            stock = yf.Ticker(ticker)
            logger.debug(f"Created yfinance Ticker object for {ticker}")
            
            info = stock.info
            logger.debug(f"Retrieved info dictionary with {len(info)} fields")
            
            # Log available fields for debugging
            logger.debug(f"Available fields for {ticker}: {list(info.keys())}")
            
            # Extract and format metadata - keeping only reliable fields
            metadata = {
                "symbol": info.get("symbol", "N/A"),
                "name": info.get("longName", info.get("shortName", "N/A")),
                "sector": info.get("sector", "N/A"),
                "industry": info.get("industry", "N/A"),
                "country": info.get("country", "N/A"),
                "marketCap": info.get("marketCap", 0),
                "currency": info.get("currency", "USD"),
                "exchange": info.get("exchange", "N/A"),
                "description": info.get("longBusinessSummary", "N/A")
            }
            
            # Log the extracted metadata
            logger.info(f"Successfully extracted metadata for {ticker}")
            logger.debug(f"Extracted metadata: {metadata}")
            
            # Log any missing or N/A fields
            missing_fields = [k for k, v in metadata.items() if v == "N/A"]
            if missing_fields:
                logger.warning(f"Missing fields for {ticker}: {missing_fields}")
            
            return ToolResult(
                success=True,
                data={
                    "ticker": ticker,
                    "metadata": metadata
                }
            )
                
        except Exception as e:
            logger.error(f"Error fetching metadata for {ticker}: {str(e)}", exc_info=True)
            return ToolResult(
                success=False,
                error=f"Failed to fetch company metadata: {str(e)}"
            )
    
    def format_result(self, data: Dict[str, Any]) -> str:
        """Format company metadata for display"""
        try:
            ticker = data["ticker"]
            metadata = data["metadata"]
            
            logger.info(f"Formatting metadata for {ticker}")
            
            # Format the company information
            info = [
                f"Company: {metadata['name']} ({ticker})",
                f"Sector: {metadata['sector']}",
                f"Industry: {metadata['industry']}",
                f"Country: {metadata['country']}",
                f"Exchange: {metadata['exchange']}"
            ]
            
            # Format market cap with proper currency
            market_cap = metadata['marketCap']
            if isinstance(market_cap, (int, float)):
                currency_symbol = "$" if metadata['currency'] == "USD" else metadata['currency']
                info.append(f"Market Cap: {currency_symbol}{market_cap:,.2f}")
                logger.debug(f"Formatted market cap: {currency_symbol}{market_cap:,.2f}")
            else:
                logger.warning(f"Invalid market cap value for {ticker}: {market_cap}")
            
            # Add description if available and not too long
            if metadata['description'] != "N/A":
                # Truncate description if it's too long
                description = metadata['description']
                original_length = len(description)
                if original_length > 300:
                    description = description[:297] + "..."
                    logger.debug(f"Truncated description from {original_length} to 300 characters")
                info.append(f"\nDescription: {description}")
            
            formatted_result = "\n".join(info)
            logger.info(f"Successfully formatted metadata for {ticker}")
            logger.debug(f"Formatted result:\n{formatted_result}")
            
            return formatted_result
            
        except Exception as e:
            logger.error(f"Error formatting metadata for {data.get('ticker', 'unknown')}: {str(e)}", exc_info=True)
            return "Error formatting company metadata"

# Create singleton instance
company_metadata_tool = CompanyMetadataTool() 