from typing import List, Dict, Any, Optional
from langchain.tools import Tool
import requests
import os
import logging
from dotenv import load_dotenv
from datetime import datetime, timedelta
import finnhub
from .base_tool import BaseTool, ToolResult

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class NewsTool(BaseTool):
    """
    Tool for fetching company news using Marketaux as primary source and Finnhub as fallback.
    Provides access to news headlines, sentiment, and related events.
    """
    
    def __init__(self):
        super().__init__()
        self.marketaux_api_key = os.getenv("MARKETAUX_API_KEY")
        self.finnhub_api_key = os.getenv("FINNHUB_API_KEY")
        if not self.marketaux_api_key:
            raise ValueError("MARKETAUX_API_KEY environment variable not set")
        if not self.finnhub_api_key:
            print("Warning: FINNHUB_API_KEY not set, will not be able to use Finnhub fallback")
        self.finnhub_client = finnhub.Client(api_key=self.finnhub_api_key) if self.finnhub_api_key else None
    
    def execute(self, symbols: list) -> ToolResult:
        """
        Execute the news tool
        
        Args:
            symbols: List of stock symbols to fetch news for
        
        Returns:
            ToolResult with news data including:
            - Headlines
            - Sentiment
            - Events
        """
        try:
            if not symbols:
                return ToolResult(
                    success=False,
                    error="At least one symbol is required"
                )
            
            # Try Marketaux first
            try:
                url = "https://api.marketaux.com/v1/news/all"
                params = {
                    "api_token": self.marketaux_api_key,
                    "symbols": ",".join(symbols),
                    "limit": 10,
                    "language": "en"
                }
                response = requests.get(url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("data"):
                        return ToolResult(
                            success=True,
                            data=self._format_marketaux_data(data)
                        )
                    else:
                        print(f"Marketaux API returned empty data for {symbols}")
                else:
                    print(f"Marketaux API error: Status code {response.status_code} for {symbols}")
            except Exception as e:
                print(f"Marketaux API error for {symbols}: {str(e)}")
            
            # Fallback to Finnhub
            if self.finnhub_client:
                try:
                    # Get news for each symbol
                    all_news = []
                    for symbol in symbols:
                        # Get news from last 7 days
                        end_date = datetime.now()
                        start_date = end_date - timedelta(days=7)
                        
                        news = self.finnhub_client.company_news(
                            symbol,
                            _from=start_date.strftime("%Y-%m-%d"),
                            to=end_date.strftime("%Y-%m-%d")
                        )
                        
                        if news:
                            all_news.extend(news)
                    
                    if all_news:
                        return ToolResult(
                            success=True,
                            data=self._format_finnhub_data(all_news)
                        )
                    else:
                        print(f"Finnhub API returned no news for {symbols}")
                except Exception as e:
                    print(f"Finnhub API error for {symbols}: {str(e)}")
            else:
                print("Finnhub API key not configured")
            
            # If we get here, both APIs failed
            error_msg = f"Failed to fetch news data for {symbols} from all sources"
            print(error_msg)
            return ToolResult(
                success=False,
                error=error_msg
            )
                
        except Exception as e:
            error_msg = f"Unexpected error fetching news for {symbols}: {str(e)}"
            print(error_msg)
            return ToolResult(
                success=False,
                error=error_msg
            )
    
    def _format_marketaux_data(self, data: Dict) -> Dict:
        """Format Marketaux API response"""
        headlines = []
        sentiments = []
        events = []
        
        for article in data.get("data", []):
            headlines.append(article.get("title", ""))
            if article.get("sentiment"):
                sentiments.append(article["sentiment"])
            if article.get("entities"):
                for entity in article["entities"]:
                    if entity.get("type") == "event":
                        events.append(entity.get("name", ""))
        
        return {
            "headlines": "\n".join(headlines),
            "sentiment": self._calculate_sentiment(sentiments),
            "events": ", ".join(events) if events else "No significant events"
        }
    
    def _format_finnhub_data(self, news: list) -> Dict:
        """Format Finnhub API response"""
        headlines = []
        sentiments = []
        events = []
        
        for article in news:
            headlines.append(article.get("headline", ""))
            # Finnhub doesn't provide sentiment, so we'll use a neutral default
            sentiments.append("neutral")
            if article.get("category"):
                events.append(article["category"])
        
        return {
            "headlines": "\n".join(headlines),
            "sentiment": "Neutral",  # Finnhub doesn't provide sentiment
            "events": ", ".join(events) if events else "No significant events"
        }
    
    def _calculate_sentiment(self, sentiments: list) -> str:
        """Calculate overall sentiment from list of sentiments"""
        if not sentiments:
            return "Neutral"
        
        positive = sum(1 for s in sentiments if s == "positive")
        negative = sum(1 for s in sentiments if s == "negative")
        total = len(sentiments)
        
        if positive > negative:
            return "Positive"
        elif negative > positive:
            return "Negative"
        else:
            return "Neutral"

# Create singleton instance
news_tool = NewsTool()

def get_news_sentiment(symbols: List[str], days_back: int = 2) -> List[Dict[str, Any]]:
    """
    Fetch recent news and sentiment for given stock symbols.
    
    Args:
        symbols: List of stock ticker symbols
        days_back: Number of days to look back for news
    
    Returns:
        List of dictionaries with news data
    """
    return news_tool.execute(symbols)

# Create LangChain tool
news_tool_langchain = Tool(
    name="news_tool",
    description="Fetch recent news and sentiment for given stock symbols",
    func=lambda symbols: news_tool.execute(symbols)
) 