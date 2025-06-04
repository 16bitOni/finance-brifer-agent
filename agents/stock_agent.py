from typing import Dict, Any, List
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import yaml
import os
import logging
from dotenv import load_dotenv

from state.state import AgentState
from tools.company_metadata_tool import company_metadata_tool
from tools.eodhd_tool import eodhd_tool
from tools.earnings_surprise_tool import earnings_surprise_tool

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables and config
load_dotenv()
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

def extract_symbols(query: str) -> List[str]:
    """Extract stock tickers from company names."""
    common_symbols = {
        "apple": "AAPL",
        "microsoft": "MSFT",
        "google": "GOOGL",
        "amazon": "AMZN",
        "meta": "META",
        "tesla": "TSLA",
        "nvidia": "NVDA",
        "amd": "AMD",
        "intel": "INTC",
        "ibm": "IBM"
    }
    query_lower = query.lower()
    return [symbol for company, symbol in common_symbols.items() if company in query_lower] or ["AAPL"]

def determine_tools_needed(query: str) -> Dict[str, bool]:
    """Determine which tools to use based on query keywords."""
    query = query.lower()
    return {
        "metadata": any(k in query for k in ["sector", "industry", "market cap", "company", "description", "ceo"]),
        "price": any(k in query for k in ["price", "close", "volume", "chart", "today", "market", "data"]),
        "earnings": any(k in query for k in ["earnings", "eps", "profit", "quarterly", "annual"])
    }

async def stock_agent(state: AgentState) -> AgentState:
    """
    Stock agent that handles stock-related queries using:
    - CompanyMetadataTool: For company information
    - EODHD Tool: For price and volume data
    - EarningsSurpriseTool: For earnings data
    """
    try:
        logger.info("Stock agent processing query: %s", state["query"])
        
        # Extract symbols and determine needed tools
        symbols = extract_symbols(state["query"])
        tools_needed = determine_tools_needed(state["query"])
        
        # If no specific tools requested, use all
        if not any(tools_needed.values()):
            tools_needed = {k: True for k in tools_needed}
        
        # Collect data from tools
        results = []
        for ticker in symbols:
            ticker_results = []
            
            # Get company metadata if needed
            if tools_needed["metadata"]:
                meta_result = await company_metadata_tool.execute(ticker)
                if meta_result.success:
                    ticker_results.append(company_metadata_tool.format_result(meta_result.data))
                else:
                    ticker_results.append(f"[Metadata Error] {meta_result.error}")
            
            # Get price data if needed
            if tools_needed["price"]:
                price_result = eodhd_tool(ticker)
                if "error" not in price_result:
                    ticker_results.append(f"""
📈 Price Data for {ticker}:
- Date: {price_result['date']}
- Close: ${price_result['close']:.2f}
- Volume: {price_result['volume']:,}
                    """)
                else:
                    ticker_results.append(f"[Price Error] {price_result['error']}")
            
            # Get earnings data if needed
            if tools_needed["earnings"]:
                earnings_result = await earnings_surprise_tool.execute(ticker)
                if earnings_result.success:
                    ticker_results.append(earnings_surprise_tool.format_result(earnings_result.data))
                else:
                    ticker_results.append(f"[Earnings Error] {earnings_result.error}")
            
            results.append("\n".join(ticker_results))
        
        combined_data = "\n\n".join(results)
        
        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a financial data analyst specializing in stock analysis. Your task is to analyze and structure stock data in a clear, comprehensive format.

            For each stock, provide:
            1. Price Data:
               - Current price
               - Volume
               - Price change
               - 52-week range
            
            2. Earnings Data:
               - Latest EPS
               - Earnings surprise (if any)
               - Earnings trend
               - Next earnings date
            
            3. Company Metrics:
               - Market cap
               - Sector/Industry
               - Key ratios
               - Recent developments
            
            Format the data in a clear, structured way that can be easily processed by other agents.
            Focus on the most relevant metrics for the query.
            Include all available data points, even if they seem negative or neutral.
            """),
            ("human", "Here is the stock data: {data}\n\nPlease analyze this data and provide insights based on the query: {query}")
        ])
        
        # Create summary with LLM
        llm = ChatOpenAI(
            base_url=config["api"]["openrouter"]["base_url"],
            model_name=config["agents"]["stock_agent"]["model"],
            temperature=config["agents"]["stock_agent"]["temperature"],
            api_key=os.getenv("OPENROUTER_API_KEY")
        )
        
        chain = prompt | llm
        response = await chain.ainvoke({"data": combined_data, "query": state["query"]})
        
        # Update state
        state["final_result"] = response.content
        state["agent"] = "stock_agent"
        state["tool_results"] = results
        
        return state
        
    except Exception as e:
        logger.error("Error in stock agent: %s", str(e), exc_info=True)
        state["error"] = str(e)
        state["final_result"] = "⚠️ An error occurred while analyzing the stock query."
        return state
