from typing import Dict, Any
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import yaml
import os
import logging
from dotenv import load_dotenv

from tools.news_tool import news_tool
from state.state import AgentState

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables and config
load_dotenv()
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

def extract_symbols(query: str) -> list:
    """
    Extract stock symbols from the query.
    This is a simple implementation - you might want to use a more sophisticated approach.
    """
    # Common stock symbols
    common_symbols = {
        # US Tech
        "apple": "AAPL",
        "microsoft": "MSFT",
        "google": "GOOGL",
        "amazon": "AMZN",
        "meta": "META",
        "tesla": "TSLA",
        "nvidia": "NVDA",
        "amd": "AMD",
        "intel": "INTC",
        "ibm": "IBM",
        # Asian Tech
        "tsmc": "TSMC",
        "taiwan semiconductor": "TSMC",
        "samsung": "005930.KS",  # Samsung Electronics on KOSPI
        "alibaba": "BABA",
        "baba": "BABA",
        "pdd": "PDD",
        "pinduoduo": "PDD"
    }
    
    query_lower = query.lower()
    found_symbols = []
    
    # First, check for region-specific keywords
    is_asia_query = any(word in query_lower for word in ["asia", "asian", "china", "chinese", "taiwan", "korea", "japan"])
    
    # Extract symbols based on company names
    for company, symbol in common_symbols.items():
        if company in query_lower:
            found_symbols.append(symbol)
    
    # If no specific symbols found but it's an Asia query, include Asian tech stocks
    if not found_symbols and is_asia_query:
        found_symbols = ["TSMC", "005930.KS", "BABA", "PDD"]
    
    # If still no symbols found, return empty list instead of defaulting to AAPL
    return found_symbols

def news_agent(state: AgentState) -> AgentState:
    """
    News agent that handles news-related queries.
    """
    try:
        logger.info("News agent started processing query: %s", state["query"])
        
        # Initialize LLM
        llm = ChatOpenAI(
            base_url=config["api"]["openrouter"]["base_url"],
            model_name=config["agents"]["news_agent"]["model"],
            temperature=config["agents"]["news_agent"]["temperature"],
            api_key=os.getenv("OPENROUTER_API_KEY")
        )
        
        # Extract symbols from query
        symbols = extract_symbols(state["query"])
        logger.info("Extracted symbols: %s", symbols)
        
        # Get news using MarketAux tool
        logger.info("Calling MarketAux tool for news...")
        news_results = news_tool.run({"symbols": symbols})
        logger.info("Received news results: %s", news_results)
        
        if not news_results:
            logger.warning("No news results returned from MarketAux tool")
            state["final_result"] = "I apologize, but I couldn't find any recent news for the requested stocks."
            return state
        
        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a financial news assistant. Summarize the following news articles in a clear and concise way and tell me without bullet points in short ."),
            ("human", "Here are the news articles: {news}\n\nPlease provide a summary focusing on the most important points.")
        ])
        
        # Generate response
        logger.info("Generating response using LLM...")
        chain = prompt | llm
        response = chain.invoke({"news": news_results})
        logger.info("Generated response: %s", response.content)
        
        # Update state
        state["tool_results"].append({
            "tool": "news_tool",
            "result": news_results
        })
        state["final_result"] = response.content
        state["agent"] = "news_agent"
        
        logger.info("News agent completed successfully")
        return state
        
    except Exception as e:
        logger.error("Error in news agent: %s", str(e), exc_info=True)
        state["error"] = str(e)
        state["final_result"] = "I apologize, but I encountered an error while processing your news request."
        return state 