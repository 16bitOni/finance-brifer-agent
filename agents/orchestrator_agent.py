from typing import Dict, Any, List, Optional
import logging
import asyncio
from datetime import datetime, timedelta
import math
from dataclasses import dataclass
from enum import Enum
import os
import json
import numpy as np
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import yaml

from agents.portfolio_agent import portfolio_agent
from tools.eodhd_tool import eodhd_tool
from tools.earnings_surprise_tool import earnings_surprise_tool
from tools.company_metadata_tool import company_metadata_tool
from tools.news_tool import news_tool

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load config
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

class Intent(Enum):
    """Enum for query intents."""
    RISK_EXPOSURE = "risk_exposure"
    EARNINGS_SURPRISE = "earnings_surprise"
    PORTFOLIO = "portfolio"
    NEWS = "news"
    METADATA = "metadata"
    STOCK = "stock"
    UNKNOWN = "unknown"

@dataclass
class QueryEntities:
    """Data class for extracted query entities."""
    region: Optional[str] = None
    sector: Optional[str] = None
    metric: Optional[str] = None
    event: Optional[str] = None
    symbols: List[str] = None
    regions: List[str] = None
    sectors: List[str] = None
    timeframes: List[str] = None

class SmartOrchestrator:
    """Smart orchestrator for handling complex queries with TTS-friendly output."""
    
    def __init__(self):
        """Initialize the orchestrator with required tools and agents."""
        logger.info("Initializing SmartOrchestrator")
        try:
            # Initialize LLM
            self.llm = ChatOpenAI(
                base_url=config["api"]["openrouter"]["base_url"],
                model_name=config["agents"]["orchestrator_agent"]["model"],
                temperature=config["agents"]["orchestrator_agent"]["temperature"],
                api_key=os.getenv("OPENROUTER_API_KEY")
            )
            
            # Initialize agents
            self.portfolio_agent = portfolio_agent
            
            # Initialize tools
            self.eodhd_tool = eodhd_tool
            logger.info("EODHD tool initialized successfully")
            
            self.earnings_surprise_tool = earnings_surprise_tool
            logger.info("Earnings surprise tool initialized successfully")
            
            self.metadata_tool = company_metadata_tool
            logger.info("Company metadata tool initialized successfully")
            
            self.news_tool = news_tool
            logger.info("News tool initialized successfully")
            
            # Cache for storing results
            self._cache = {}
            self._cache_ttl = 3600  # 1 hour cache TTL
            
            # Enhanced company name mapping for TTS
            self.company_names = {
                "AAPL": "Apple",
                "MSFT": "Microsoft", 
                "GOOGL": "Google",
                "AMZN": "Amazon",
                "TSLA": "Tesla",
                "META": "Meta",
                "NVDA": "Nvidia",
                "TSM": "Taiwan Semiconductor",
                "ASML": "ASML",
                "005930.KS": "Samsung Electronics",
                "9988.HK": "Alibaba",
                "PDD": "PDD Holdings",
                "BABA": "Alibaba",
                "JD": "JD dot com",
                "TCEHY": "Tencent",
                "JPM": "JPMorgan Chase",
                "BAC": "Bank of America",
                "WFC": "Wells Fargo",
                "GS": "Goldman Sachs",
                "MS": "Morgan Stanley",
                "JNJ": "Johnson and Johnson",
                "PFE": "Pfizer",
                "UNH": "UnitedHealth",
                "CVX": "Chevron",
                "XOM": "ExxonMobil",
                "KO": "Coca Cola",
                "PEP": "PepsiCo",
                "WMT": "Walmart",
                "HD": "Home Depot",
                "V": "Visa",
                "MA": "Mastercard"
            }
            
            logger.info("Successfully initialized all tools and agents")
        except Exception as e:
            logger.error(f"Error initializing orchestrator: {e}", exc_info=True)
            raise
    
    def _get_friendly_company_name(self, symbol: str) -> str:
        """Convert stock symbol to TTS-friendly company name."""
        return self.company_names.get(symbol, symbol.replace(".", " "))
    
    def _format_currency_for_speech(self, amount: float) -> str:
        """Format currency amounts in a TTS-friendly way."""
        if amount >= 1_000_000:
            return f"{amount / 1_000_000:.1f} million dollars"
        elif amount >= 1_000:
            return f"{amount / 1_000:.0f} thousand dollars"
        else:
            return f"{amount:.0f} dollars"
    
    def _format_percentage_for_speech(self, percent: float) -> str:
        """Format percentages in a TTS-friendly way."""
        if percent > 0:
            return f"up {percent:.1f} percent"
        elif percent < 0:
            return f"down {abs(percent):.1f} percent"
        else:
            return "unchanged"
    
    def _parse_query(self, query: Dict[str, Any]) -> tuple[Intent, QueryEntities]:
        """Parse the query to determine intent and extract entities"""
        logger.info(f"Parsing query: {query}")
        
        # Extract query text from dictionary
        query_text = query.get('query', '')
        if not query_text:
            logger.error("No query text found in input dictionary")
            return Intent.UNKNOWN, QueryEntities()
            
        logger.info(f"Query text: {query_text}")
        
        # Default to portfolio intent if no specific intent is detected
        intent = Intent.PORTFOLIO
        entities = QueryEntities()
        
        # Simple keyword-based intent detection
        query_lower = query_text.lower()
        logger.debug(f"Analyzing query keywords: {query_lower}")
        
        if any(word in query_lower for word in ["stock", "price", "market"]):
            intent = Intent.STOCK
            logger.info("Detected STOCK intent based on keywords: stock/price/market")
        elif any(word in query_lower for word in ["earnings", "surprise"]):
            intent = Intent.EARNINGS_SURPRISE
            logger.info("Detected EARNINGS intent based on keywords: earnings/surprise")
        elif any(word in query_lower for word in ["news", "headline"]):
            intent = Intent.NEWS
            logger.info("Detected NEWS intent based on keywords: news/headline")
        elif any(word in query_lower for word in ["risk", "exposure"]):
            intent = Intent.RISK_EXPOSURE
            logger.info("Detected RISK intent based on keywords: risk/exposure")
        
        # Extract entities (symbols, regions, sectors, timeframes)
        logger.debug("Extracting entities from query")
        
        # Extract symbols (assuming they're in uppercase)
        words = query_text.split()
        symbols = [word for word in words if word.isupper() and len(word) <= 5]
        if symbols:
            entities.symbols = symbols
            logger.info(f"Extracted symbols: {symbols}")
        
        # Extract regions
        regions = []
        if "asia" in query_lower:
            regions.append("Asia")
        if "europe" in query_lower:
            regions.append("Europe")
        if "us" in query_lower or "america" in query_lower:
            regions.append("US")
        if regions:
            entities.regions = regions
            logger.info(f"Extracted regions: {regions}")
        
        # Extract sectors
        sectors = []
        if "tech" in query_lower or "technology" in query_lower:
            sectors.append("Technology")
        if "finance" in query_lower or "financial" in query_lower:
            sectors.append("Financial")
        if "health" in query_lower or "healthcare" in query_lower:
            sectors.append("Healthcare")
        if sectors:
            entities.sectors = sectors
            logger.info(f"Extracted sectors: {sectors}")
        
        # Extract timeframes
        timeframes = []
        if "today" in query_lower:
            timeframes.append("today")
        if "week" in query_lower:
            timeframes.append("week")
        if "month" in query_lower:
            timeframes.append("month")
        if timeframes:
            entities.timeframes = timeframes
            logger.info(f"Extracted timeframes: {timeframes}")
        
        logger.info(f"Query parsed - Intent: {intent.value}, Entities: {entities.__dict__}")
        return intent, entities
    
    async def _get_portfolio_data(self) -> Dict[str, Any]:
        """Get portfolio data from the portfolio agent"""
        logger.info("Fetching portfolio data from portfolio agent")
        try:
            result = await self.portfolio_agent.process("Get portfolio holdings and allocations")
            logger.debug(f"Portfolio agent response: {result}")
            
            # Handle both string and dictionary responses
            if isinstance(result, str):
                try:
                    result = json.loads(result)
                except json.JSONDecodeError:
                    logger.error("Failed to parse portfolio agent response as JSON")
                    return {"error": "Invalid portfolio data format"}
            
            if not result.get("success", False):
                error_msg = result.get("error", "Unknown error")
                logger.error(f"Portfolio agent error: {error_msg}")
                return {"error": error_msg}
            
            data = result.get("data", {})
            if not data.get("symbols"):
                logger.warning("No symbols found in portfolio data")
                return {"error": "No portfolio data available"}
            
            logger.info(f"Successfully retrieved portfolio data with {len(data['symbols'])} symbols")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching portfolio data: {str(e)}", exc_info=True)
            return {"error": str(e)}
    
    async def _get_stock_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Get stock data for the given symbols"""
        logger.info(f"Fetching stock data for symbols: {symbols}")
        try:
            results = {}
            for symbol in symbols:
                logger.debug(f"Processing symbol: {symbol}")
                result = self.eodhd_tool.invoke({"ticker": symbol})
                logger.debug(f"EODHD tool response for {symbol}: {result}")
                
                if not result.get("success", False):
                    logger.warning(f"Failed to get data for {symbol}: {result.get('error')}")
                    continue
                    
                # Extract the stock data from the response
                stock_data = result.get("data", {})
                if not stock_data:
                    logger.warning(f"No data found for {symbol}")
                    continue
                
                # Store the stock data with the symbol as key
                results[symbol] = stock_data
            
            logger.info(f"Successfully retrieved stock data for {len(results)} symbols")
            return results
            
        except Exception as e:
            logger.error(f"Error fetching stock data: {str(e)}", exc_info=True)
            return {"error": str(e)}
    
    async def _get_earnings_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Get earnings data for the given symbols"""
        logger.info(f"Fetching earnings data for symbols: {symbols}")
        try:
            results = {}
            for symbol in symbols:
                logger.debug(f"Processing symbol: {symbol}")
                result = await self.earnings_surprise_tool.execute(symbol)
                logger.debug(f"Earnings tool response for {symbol}: {result}")
                
                if not result.success:
                    logger.warning(f"Failed to get earnings data for {symbol}: {result.error}")
                    continue
                    
                results[symbol] = result.data
            
            logger.info(f"Successfully retrieved earnings data for {len(results)} symbols")
            return results
            
        except Exception as e:
            logger.error(f"Error fetching earnings data: {str(e)}", exc_info=True)
            return {"error": str(e)}
    
    async def _get_metadata(self, symbols: List[str]) -> Dict[str, Any]:
        """Get company metadata."""
        logger.info(f"Fetching company metadata for symbols: {symbols}")
        try:
            metadata = {}
            for symbol in symbols:
                logger.debug(f"Processing symbol: {symbol}")
                result = self.metadata_tool.invoke({"ticker": symbol})
                logger.debug(f"Metadata tool response for {symbol}: {result}")
                
                if not result.get("success", False):
                    logger.warning(f"Failed to get metadata for {symbol}: {result.get('error')}")
                    continue
                    
                metadata[symbol] = result.get("data", {})
            
            logger.info(f"Successfully retrieved metadata for {len(metadata)} symbols")
            return metadata
            
        except Exception as e:
            logger.error(f"Error fetching company metadata: {e}", exc_info=True)
            return {"error": str(e)}
    
    async def _get_news(self, symbols: List[str]) -> Dict[str, Any]:
        """Get news for the given symbols"""
        logger.info(f"Fetching news for symbols: {symbols}")
        try:
            results = {}
            for symbol in symbols:
                logger.debug(f"Processing symbol: {symbol}")
                result = self.news_tool.invoke({"ticker": symbol})
                logger.debug(f"News tool response for {symbol}: {result}")
                
                if not result.get("success", False):
                    logger.warning(f"Failed to get news for {symbol}: {result.get('error')}")
                    continue
                    
                results[symbol] = result.get("data", {})
            
            logger.info(f"Successfully retrieved news for {len(results)} symbols")
            return results
            
        except Exception as e:
            logger.error(f"Error fetching news: {str(e)}", exc_info=True)
            return {"error": str(e)}
    
    def _calculate_portfolio_risk(self, portfolio_data: Dict[str, Any], 
                                stock_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate portfolio risk metrics"""
        logger.info("Calculating portfolio risk metrics")
        try:
            # Simple risk calculation based on sector and region concentration
            risk_metrics = {
                "sector_concentration": portfolio_data.get("sector_allocations", {}),
                "region_concentration": portfolio_data.get("region_allocations", {}),
                "total_value": portfolio_data.get("total_value", 0),
                "volatility": 0.0,
                "high_risk_stocks": []
            }
            
            # Calculate volatility for each stock
            for symbol, data in stock_data.items():
                if isinstance(data, dict) and "historical_data" in data:
                    try:
                        # Calculate daily returns
                        historical_data = data["historical_data"]
                        if len(historical_data) > 1:
                            returns = []
                            for i in range(1, len(historical_data)):
                                prev_close = historical_data[i-1]["Close"]
                                curr_close = historical_data[i]["Close"]
                                daily_return = (curr_close - prev_close) / prev_close
                                returns.append(daily_return)
                            
                            # Calculate volatility (standard deviation of returns)
                            if returns:
                                volatility = np.std(returns) * np.sqrt(252)  # Annualized
                                if volatility > 0.3:  # 30% annualized volatility threshold
                                    risk_metrics["high_risk_stocks"].append(symbol)
                    except Exception as e:
                        logger.warning(f"Error calculating volatility for {symbol}: {e}")
            
            logger.debug(f"Calculated risk metrics: {risk_metrics}")
            
            # Log concentration warnings
            for sector, alloc in risk_metrics["sector_concentration"].items():
                if alloc > 30:
                    logger.warning(f"High sector concentration in {sector}: {alloc:.1f}%")
            
            for region, alloc in risk_metrics["region_concentration"].items():
                if alloc > 40:
                    logger.warning(f"High region concentration in {region}: {alloc:.1f}%")
            
            logger.info("Risk metrics calculation completed")
            return risk_metrics
            
        except Exception as e:
            logger.error(f"Error calculating portfolio risk: {str(e)}", exc_info=True)
            return {"error": str(e)}
    
    def _format_morning_brief(self, results: Dict[str, Any]) -> str:
        """Format the results into a morning brief."""
        try:
            brief = []
            timestamp = datetime.now().strftime("%B %d, %Y")
            
            # Add header
            brief.append(f"Morning Brief ({timestamp}):")
            
            # Add portfolio risk section if available
            if "risk" in results:
                brief.append("\nPortfolio Risk Exposure:")
                brief.append(f"- Portfolio Volatility: {results['risk']['volatility']:.2%}")
                if results['risk'].get('high_risk_stocks'):
                    brief.append("- High Risk Stocks:")
                    for stock in results['risk']['high_risk_stocks']:
                        brief.append(f"  * {stock}")
            
            # Add earnings surprises section if available
            if "earnings" in results and results["earnings"]:
                brief.append("\nEarnings Surprises:")
                for symbol, data in results["earnings"].items():
                    if data and "earnings" in data:
                        latest_earnings = data["earnings"][0]  # Get most recent earnings
                        if "surprise" in latest_earnings:
                            brief.append(f"- {symbol}: {latest_earnings['surprise']:.1%} surprise on {latest_earnings['date']}")
            
            # Add news section if available
            if "news" in results and results["news"]:
                brief.append("\nRelevant News:")
                for news in results["news"][:3]:  # Show top 3 news items
                    brief.append(f"- {news['title']} ({news['date']})")
            
            return "\n".join(brief)
        except Exception as e:
            logger.error(f"Error formatting morning brief: {e}")
            return "Error generating morning brief"

    async def _analyze_with_llm(self, data: Dict[str, Any], intent: Intent) -> str:
        """Analyze the data using LLM and generate TTS-friendly insights."""
        logger.info("Analyzing data with LLM for TTS output")
        try:
            # Convert numpy types to Python native types
            def convert_numpy_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {key: convert_numpy_types(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                return obj

            # Get the original portfolio data
            original_portfolio = data.get("original_portfolio", {})
            filtered_portfolio = data.get("portfolio", {})
            
            # Create company name mapping for the prompt
            symbol_to_name = {}
            all_symbols = set()
            
            # Collect all symbols from holdings
            for holding in original_portfolio.get("holdings", []):
                symbol = holding.get("symbol", "")
                if symbol:
                    all_symbols.add(symbol)
            
            # Add symbols from filtered portfolio
            for holding in filtered_portfolio.get("holdings", []):
                symbol = holding.get("symbol", "")
                if symbol:
                    all_symbols.add(symbol)
            
            # Create mapping for all symbols
            for symbol in all_symbols:
                symbol_to_name[symbol] = self._get_friendly_company_name(symbol)
            
            # Determine what data to show based on query context
            if intent == Intent.PORTFOLIO:
                # For portfolio queries, show the filtered results but with full context
                analysis_data = {
                    "requested_holdings": convert_numpy_types(filtered_portfolio.get("holdings", [])),
                    "requested_value": filtered_portfolio.get("total_value", 0),
                    "total_portfolio_value": original_portfolio.get("total_value", 0),
                    "filters": data.get("filters", {}),
                    "intent": intent.value,
                    "symbol_names": symbol_to_name
                }
            else:
                # For other queries, show full portfolio context
                analysis_data = {
                    "portfolio": {
                        "total_value": original_portfolio.get("total_value", 0),
                        "holdings": original_portfolio.get("holdings", []),
                        "region_allocations": original_portfolio.get("region_allocations", {}),
                        "sector_allocations": original_portfolio.get("sector_allocations", {})
                    },
                    "filtered_data": convert_numpy_types(data),
                    "original_portfolio": original_portfolio,
                    "intent": intent.value,
                    "symbol_names": symbol_to_name
                }
            
            # Create TTS-optimized conversational prompts for all intents
            if intent == Intent.RISK_EXPOSURE:
                prompt = ChatPromptTemplate.from_messages([
                    ("system", """You are a friendly financial advisor speaking to a client. Your response will be converted to speech, so write exactly how you would speak out loud.

                    CRITICAL TTS GUIDELINES:
                    - Use full company names, never stock symbols (e.g., "Apple" not "AAPL")
                    - Spell out numbers clearly (e.g., "three hundred thirty-four thousand dollars" not "$334K")
                    - Use natural speech patterns with connecting words
                    - Avoid abbreviations, symbols, or technical formatting
                    - Write percentages as "percent" not "%"
                    - Keep it conversational and flowing

                    Provide a brief, natural response (2-3 sentences) about their portfolio risk exposure. Focus on:
                    - Current portfolio value using full dollar amounts in speech-friendly format
                    - Major allocation percentages in natural language
                    - Simple risk observation

                    Example: "Your portfolio is worth three hundred thirty-four thousand dollars with about sixty-five percent in Asian technology stocks. The overall risk level looks moderate, though you have a couple of more volatile stocks. You might want to consider spreading things out a bit more across different sectors."
                    
                    Remember: This will be read aloud, so make it sound natural and conversational."""),
                    ("human", "Here is the portfolio data with company name mappings: {data}")
                ])
            
            elif intent == Intent.PORTFOLIO:
                prompt = ChatPromptTemplate.from_messages([
                    ("system", """You are a friendly financial advisor speaking to a client. Your response will be converted to speech, so write exactly how you would speak out loud.

                    CRITICAL TTS GUIDELINES:
                    - Use full company names from the symbol_names mapping, never stock symbols
                    - Spell out numbers and currency in speech-friendly format
                    - Use natural speech patterns and connecting words
                    - Avoid abbreviations, symbols, or technical formatting
                    - Write percentages as "percent" not "%"
                    - List companies naturally with "and" between the last two items

                    The client asked about specific holdings. You have:
                    - requested_holdings: The specific stocks they asked about
                    - requested_value: Total value of those specific stocks  
                    - total_portfolio_value: Their entire portfolio value
                    - symbol_names: Mapping of symbols to company names

                    Provide a brief, natural response (2-3 sentences) that includes:
                    - List the specific companies they asked about using full names
                    - The total value of those stocks in speech-friendly format
                    - What percentage of their total portfolio it represents

                    Example: "Your technology holdings include Apple, Microsoft, Taiwan Semiconductor, Samsung Electronics, Alibaba, and PDD Holdings. That's about two hundred forty-three thousand dollars worth of tech stocks, which makes up roughly seventy-three percent of your total three hundred thirty-four thousand dollar portfolio."
                    
                    Remember: This will be read aloud, so make it sound natural and conversational."""),
                    ("human", "Here is the data with company name mappings: {data}")
                ])
            
            elif intent == Intent.EARNINGS_SURPRISE:
                prompt = ChatPromptTemplate.from_messages([
                    ("system", """You are a friendly financial advisor speaking to a client. Your response will be converted to speech, so write exactly how you would speak out loud.

                    CRITICAL TTS GUIDELINES:
                    - Use full company names from symbol_names mapping, never stock symbols
                    - Spell out percentages clearly (e.g., "five point two percent" not "5.2%")
                    - Use natural speech patterns
                    - Avoid technical jargon or abbreviations
                    - Keep it conversational and flowing

                    Provide a brief, natural response (2-3 sentences) about earnings surprises. Focus on:
                    - Which companies had notable earnings surprises using full names
                    - Brief impact or trend in conversational language
                    - One simple takeaway

                    Example: "Apple had a pleasant surprise with earnings coming in five point two percent higher than expected, while Microsoft was a bit disappointing at two percent below estimates. Overall, your tech holdings showed mixed results this quarter."
                    
                    Remember: This will be read aloud, so make it sound natural."""),
                    ("human", "Here is the earnings data with company name mappings: {data}")
                ])
            
            elif intent == Intent.NEWS:
                prompt = ChatPromptTemplate.from_messages([
                    ("system", """You are a friendly financial advisor speaking to a client. Your response will be converted to speech, so write exactly how you would speak out loud.

                    CRITICAL TTS GUIDELINES:
                    - Use full company names from symbol_names mapping, never stock symbols
                    - Summarize news headlines in natural, conversational language
                    - Avoid reading exact headlines or technical terms
                    - Keep it flowing and easy to understand when spoken
                    - Focus on the key themes rather than specific details

                    Provide a brief, natural response (2-3 sentences) about relevant news. Focus on:
                    - Key themes affecting their holdings using company names
                    - Brief impact or trend in simple language
                    - One practical takeaway

                    Example: "There's been quite a bit of positive news around Apple's new product launches and Microsoft's cloud business growth. Most of the headlines suggest your tech holdings are getting good attention from investors right now."
                    
                    Remember: This will be read aloud, so make it sound natural and conversational."""),
                    ("human", "Here is the news data with company name mappings: {data}")
                ])
            
            elif intent == Intent.STOCK:
                prompt = ChatPromptTemplate.from_messages([
                    ("system", """You are a friendly financial advisor speaking to a client. Your response will be converted to speech, so write exactly how you would speak out loud.

                    CRITICAL TTS GUIDELINES:
                    - Use full company names from symbol_names mapping, never stock symbols
                    - Spell out prices clearly (e.g., "one hundred fifty-two dollars and thirty cents")
                    - Use "up" or "down" instead of "+" or "-" for changes
                    - Spell out percentages (e.g., "three point five percent")
                    - Keep it natural and conversational

                    Provide a brief, natural response (2-3 sentences) about stock performance. Focus on:
                    - Current prices and changes for requested stocks using company names
                    - Brief trend or pattern in conversational language
                    - One simple observation

                    Example: "Apple is trading at one hundred fifty-two dollars and thirty cents, up about two point one percent today. Microsoft is at two hundred eighty-five dollars, down slightly by one point three percent."
                    
                    Remember: This will be read aloud, so make it sound natural."""),
                    ("human", "Here is the stock data with company name mappings: {data}")
                ])
            
            else:
                # Default conversational prompt
                prompt = ChatPromptTemplate.from_messages([
                    ("system", """You are a friendly financial advisor speaking to a client. Your response will be converted to speech, so write exactly how you would speak out loud.

                    CRITICAL TTS GUIDELINES:
                    - Use full company names from symbol_names mapping, never stock symbols
                    - Spell out all numbers, currencies, and percentages clearly
                    - Use natural speech patterns and connecting words
                    - Avoid technical formatting, symbols, or abbreviations
                    - Keep it conversational and flowing

                    Provide a brief, natural response (2-3 sentences) about their query. Focus on:
                    - Key information they're asking about using natural language
                    - Brief analysis or trend in conversational terms
                    - One simple observation or suggestion

                    Remember: This will be read aloud, so make it sound exactly like natural speech."""),
                    ("human", "Here is the data with company name mappings: {data}")
                ])
            
            # Generate analysis
            chain = prompt | self.llm
            response = await chain.ainvoke({"data": json.dumps(analysis_data, indent=2)})
            
            logger.info("Successfully generated TTS-friendly LLM analysis")
            return response.content
            
        except Exception as e:
            logger.error(f"Error analyzing data with LLM: {str(e)}", exc_info=True)
            return "I apologize, but I'm having trouble analyzing your portfolio data right now."

    def _format_results(self, intent: Intent, data: Dict[str, Any]) -> str:
        """Format the results in a TTS-friendly way - mostly letting LLM handle formatting."""
        logger.info(f"Formatting results for intent: {intent.value}")
        try:
            # For TTS output, we'll let the LLM handle most formatting
            # Only provide basic structure if needed
            if intent == Intent.PORTFOLIO:
                return ""  # Let LLM handle everything for natural speech
            else:
                return ""  # Let LLM handle everything for natural speech
                
        except Exception as e:
            logger.error(f"Error formatting results: {str(e)}", exc_info=True)
            return ""

    async def _understand_query(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM to understand the query and determine required tools and filters."""
        logger.info("Understanding query with LLM")
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a financial analyst assistant. Analyze the query and determine the required tools and filters.
                
                Available tools:
                - portfolio: For portfolio holdings and allocations
                - stock: For current stock prices and market data
                - earnings: For earnings surprises and reports
                - news: For recent news and headlines
                - risk: For risk metrics and exposure analysis
                
                Return ONLY a valid JSON object with this exact structure:
                {{
                    "intent": "risk_exposure" | "earnings_surprise" | "portfolio" | "news" | "stock" | "unknown",
                    "tools": ["tool1", "tool2"],
                    "filters": {{
                        "regions": ["region1"],
                        "sectors": ["sector1"],
                        "symbols": ["symbol1"]
                    }},
                    "time_period": "period",
                    "metrics": ["metric1"]
                }}
                
                Example for "Show me risk exposure in Asia tech stocks":
                {{
                    "intent": "risk_exposure",
                    "tools": ["portfolio", "risk", "stock"],
                    "filters": {{
                        "regions": ["Asia"],
                        "sectors": ["Technology"]
                    }},
                    "time_period": "latest",
                    "metrics": ["volatility", "concentration"]
                }}
                
                Example for "What are the earnings surprises for tech stocks":
                {{
                    "intent": "earnings_surprise",
                    "tools": ["earnings", "stock"],
                    "filters": {{
                        "sectors": ["Technology"]
                    }},
                    "time_period": "latest",
                    "metrics": ["eps", "surprise"]
                }}"""),
                ("human", "Query: {query}")
            ])
            
            chain = prompt | self.llm
            response = await chain.ainvoke({"query": query.get("query", "")})
            
            try:
                # Parse the LLM response as JSON
                result = json.loads(response.content)
                logger.info(f"Parsed query analysis: {result}")
                return result
                
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Failed to parse LLM response as JSON: {str(e)}")
                # Fallback to keyword-based tool selection
                query_lower = query.get("query", "").lower()
                tools = []
                intent = "unknown"
                
                # Determine intent
                if any(word in query_lower for word in ["risk", "exposure", "concentration"]):
                    intent = "risk_exposure"
                    tools.extend(["portfolio", "risk", "stock"])
                elif any(word in query_lower for word in ["earnings", "surprise", "report"]):
                    intent = "earnings_surprise"
                    tools.extend(["earnings", "stock"])
                elif any(word in query_lower for word in ["news", "headline", "update"]):
                    intent = "news"
                    tools.extend(["news"])
                elif any(word in query_lower for word in ["stock", "price", "market"]):
                    intent = "stock"
                    tools.extend(["stock"])
                else:
                    intent = "portfolio"
                    tools.extend(["portfolio"])
                
                # Add filters based on keywords
                filters = {
                    "regions": [],
                    "sectors": [],
                    "symbols": []
                }
                
                if "asia" in query_lower:
                    filters["regions"].append("Asia")
                if "tech" in query_lower:
                    filters["sectors"].append("Technology")
                
                return {
                    "intent": intent,
                    "tools": tools,
                    "filters": filters,
                    "time_period": "latest",
                    "metrics": ["price", "change"]
                }
                
        except Exception as e:
            logger.error(f"Error understanding query: {str(e)}", exc_info=True)
            return {
                "intent": "portfolio",
                "tools": ["portfolio", "stock"],
                "filters": {},
                "time_period": "latest",
                "metrics": ["price", "change"]
            }

    def _filter_portfolio_data(self, portfolio_data: Dict[str, Any], filters: Dict[str, Any]) -> Dict[str, Any]:
        """Filter portfolio data based on query context."""
        logger.info("Filtering portfolio data")
        try:
            filtered_data = {
                "symbols": [],
                "holdings": [],
                "region_allocations": {},
                "sector_allocations": {},
                "total_value": 0.0
            }
            
            # Get filter criteria
            regions = filters.get("regions", [])
            sectors = filters.get("sectors", [])
            symbols = filters.get("symbols", [])
            
            # Filter holdings
            for holding in portfolio_data.get("holdings", []):
                if symbols and holding["symbol"] not in symbols:
                    continue
                if regions and holding["region"] not in regions:
                    continue
                if sectors and holding["sector"] not in sectors:
                    continue
                    
                filtered_data["holdings"].append(holding)
                filtered_data["symbols"].append(holding["symbol"])
            
            # Recalculate allocations for filtered holdings
            total_value = sum(h["shares"] * h["avg_price"] for h in filtered_data["holdings"])
            filtered_data["total_value"] = total_value
            
            # Calculate filtered allocations
            for holding in filtered_data["holdings"]:
                value = holding["shares"] * holding["avg_price"]
                region = holding["region"]
                sector = holding["sector"]
                
                # Update region allocation
                if region not in filtered_data["region_allocations"]:
                    filtered_data["region_allocations"][region] = 0
                filtered_data["region_allocations"][region] += (value / total_value) * 100
                
                # Update sector allocation
                if sector not in filtered_data["sector_allocations"]:
                    filtered_data["sector_allocations"][sector] = 0
                filtered_data["sector_allocations"][sector] += (value / total_value) * 100
            
            logger.info(f"Filtered portfolio data: {len(filtered_data['holdings'])} holdings")
            return filtered_data
            
        except Exception as e:
            logger.error(f"Error filtering portfolio data: {str(e)}", exc_info=True)
            return portfolio_data

    async def process(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Process the query and return formatted results"""
        logger.info(f"Processing query: {query}")
        try:
            # Understand query and determine required tools
            query_analysis = await self._understand_query(query)
            logger.info(f"Query analysis: {query_analysis}")
            
            # Get portfolio data
            logger.info("Fetching portfolio data from portfolio agent")
            portfolio_data = await self._get_portfolio_data()
            if "error" in portfolio_data:
                logger.error(f"Failed to get portfolio data: {portfolio_data['error']}")
                return {"error": portfolio_data["error"]}
            
            # Filter portfolio data based on query context
            filtered_portfolio = self._filter_portfolio_data(portfolio_data, query_analysis["filters"])
            
            # Get symbols to process
            symbols = query_analysis["filters"].get("symbols") or filtered_portfolio.get("symbols", [])
            if not symbols:
                logger.warning("No symbols found to process")
                return {"error": "No symbols found to process"}
            
            logger.info(f"Processing {len(symbols)} symbols: {symbols}")
            
            # Collect data based on required tools
            data = {
                "portfolio": filtered_portfolio,
                "original_portfolio": portfolio_data,
                "intent": query_analysis.get("intent", "unknown"),
                "filters": query_analysis["filters"]
            }
            
            # Get data from required tools
            if "stock" in query_analysis["tools"]:
                logger.info("Fetching stock data from EODHD tool")
                stock_data = await self._get_stock_data(symbols)
                data["stock"] = stock_data
            
            if "earnings" in query_analysis["tools"]:
                logger.info("Fetching earnings data from earnings surprise tool")
                earnings_data = await self._get_earnings_data(symbols)
                data["earnings"] = earnings_data
            
            if "news" in query_analysis["tools"]:
                logger.info("Fetching news data from news tool")
                news_data = await self._get_news(symbols)
                data["news"] = news_data
            
            if "risk" in query_analysis["tools"]:
                logger.info("Calculating risk metrics using portfolio and stock data")
                risk_data = self._calculate_portfolio_risk(filtered_portfolio, data.get("stock", {}))
                data["risk"] = risk_data
            
            # Format raw results
            logger.info("Formatting raw results")
            formatted_result = self._format_results(Intent(query_analysis.get("intent", "unknown")), data)
            
            # Generate LLM analysis
            logger.info("Generating LLM analysis")
            analysis = await self._analyze_with_llm(data, Intent(query_analysis.get("intent", "unknown")))
            
            # Combine results
            final_result = f"{formatted_result}{analysis}"
            
            logger.info("Query processed successfully")
            return {
                "success": True,
                "data": data,
                "final_result": final_result
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "final_result": "I apologize, but I encountered an error while processing your query."
            }

# Create singleton instance
smart_orchestrator = SmartOrchestrator() 