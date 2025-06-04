from typing import Dict, Any, Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import yaml
import os
import logging
from dotenv import load_dotenv
import json
import re
from datetime import datetime

from tools.vectorstore import query_text, get_vector_store
from state.state import AgentState

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables and config
load_dotenv()
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

class PortfolioAgent:
    """Portfolio agent that handles portfolio-related queries."""
    
    def __init__(self):
        """Initialize the portfolio agent."""
        self.llm = ChatOpenAI(
            base_url=config["api"]["openrouter"]["base_url"],
            model_name=config["agents"]["portfolio_agent"]["model"],
            temperature=config["agents"]["portfolio_agent"]["temperature"],
            api_key=os.getenv("OPENROUTER_API_KEY")
        )
        self.vector_store = get_vector_store()
    
    async def process(self, state: Union[Dict[str, Any], str]) -> Dict[str, Any]:
        """
        Process portfolio-related queries asynchronously.
        Returns a structured response with symbols, holdings, and allocations.
        """
        try:
            # Handle both string and dictionary inputs
            if isinstance(state, str):
                query = state
                timestamp = datetime.now().isoformat()
            else:
                query = state.get("query", "")
                timestamp = state.get("timestamp", datetime.now().isoformat())
            
            logger.info("Portfolio agent started processing query: %s", query)
            
            # Query vector store
            logger.info("Querying vector store...")
            results = query_text(self.vector_store, query)
            logger.info("Vector store results: %s", results)
            
            if not results:
                logger.warning("No relevant portfolio data found")
                return {
                    "success": True,
                    "data": {
                        "symbols": [],
                        "holdings": [],
                        "allocations": {},
                        "region_allocations": {},
                        "sector_allocations": {},
                        "total_value": 0.0,
                        "timestamp": timestamp
                    }
                }
            
            # Extract and combine portfolio data from vector store results
            portfolio_data = None
            holdings = []
            
            # First, extract holdings from each chunk
            for hit in results.get('result', {}).get('hits', []):
                chunk_text = hit.get('fields', {}).get('chunk_text', '')
                if not chunk_text:
                    continue
                    
                try:
                    # Try to parse the chunk directly
                    chunk_data = json.loads(chunk_text)
                    if 'portfolio' in chunk_data and 'holdings' in chunk_data['portfolio']:
                        holdings.extend(chunk_data['portfolio']['holdings'])
                except json.JSONDecodeError:
                    # If direct parsing fails, try to extract holdings using regex
                    holdings_matches = re.finditer(
                        r'"symbol":\s*"([^"]+)",\s*"shares":\s*(\d+),\s*"avg_price":\s*([\d.]+),\s*"sector":\s*"([^"]+)",\s*"region":\s*"([^"]+)"',
                        chunk_text
                    )
                    for match in holdings_matches:
                        symbol, shares, price, sector, region = match.groups()
                        holdings.append({
                            "symbol": symbol,
                            "shares": int(shares),
                            "avg_price": float(price),
                            "sector": sector,
                            "region": region
                        })
            
            # If we found any holdings, create a proper portfolio structure
            if holdings:
                # Remove duplicates (in case same holding appears in multiple chunks)
                unique_holdings = []
                seen_symbols = set()
                for holding in holdings:
                    symbol = holding.get('symbol')
                    if symbol and symbol not in seen_symbols:
                        seen_symbols.add(symbol)
                        unique_holdings.append(holding)
                
                portfolio_data = {
                    "portfolio": {
                        "holdings": unique_holdings,
                        "cash": 50000.0,  # Default value
                        "last_updated": datetime.now().isoformat()
                    }
                }
                logger.info("Successfully extracted portfolio data with %d holdings", len(unique_holdings))
            
            if not portfolio_data or 'portfolio' not in portfolio_data:
                logger.warning("No valid portfolio data found in results")
                return {
                    "success": True,
                    "data": {
                        "symbols": [],
                        "holdings": [],
                        "allocations": {},
                        "region_allocations": {},
                        "sector_allocations": {},
                        "total_value": 0.0,
                        "timestamp": timestamp
                    }
                }
            
            # Process portfolio data
            holdings = portfolio_data.get('portfolio', {}).get('holdings', [])
            cash = portfolio_data.get('portfolio', {}).get('cash', 0.0)
            
            # Calculate total portfolio value
            total_value = cash
            for holding in holdings:
                total_value += holding.get('shares', 0) * holding.get('avg_price', 0)
            
            # Group holdings by region and sector
            region_holdings = {}
            sector_holdings = {}
            symbols = []
            
            for holding in holdings:
                symbol = holding.get('symbol')
                shares = holding.get('shares', 0)
                price = holding.get('avg_price', 0)
                region = holding.get('region')
                sector = holding.get('sector')
                
                if symbol:
                    symbols.append(symbol)
                    value = shares * price
                    
                    # Group by region
                    if region:
                        region_holdings[region] = region_holdings.get(region, 0) + value
                    
                    # Group by sector
                    if sector:
                        sector_holdings[sector] = sector_holdings.get(sector, 0) + value
            
            # Calculate percentages
            region_allocations = {
                region: (value / total_value * 100) 
                for region, value in region_holdings.items()
            }
            sector_allocations = {
                sector: (value / total_value * 100)
                for sector, value in sector_holdings.items()
            }
            
            # Create structured response
            structured_response = {
                "success": True,
                "data": {
                    "symbols": symbols,
                    "holdings": holdings,
                    "allocations": {
                        "regions": region_allocations,
                        "sectors": sector_allocations
                    },
                    "timestamp": portfolio_data.get('portfolio', {}).get('last_updated', timestamp),
                    "region_allocations": region_allocations,
                    "sector_allocations": sector_allocations,
                    "total_value": total_value
                }
            }
            
            logger.info("Structured response: %s", structured_response)
            return structured_response
            
        except Exception as e:
            logger.error("Error in portfolio agent: %s", str(e), exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "data": {
                    "symbols": [],
                    "holdings": [],
                    "allocations": {},
                    "region_allocations": {},
                    "sector_allocations": {},
                    "total_value": 0.0,
                    "timestamp": timestamp if 'timestamp' in locals() else datetime.now().isoformat()
                }
            }

# Create singleton instance
portfolio_agent = PortfolioAgent() 