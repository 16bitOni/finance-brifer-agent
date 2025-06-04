from typing import Dict, Any
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import yaml
import os
import logging
from dotenv import load_dotenv

from state.state import AgentState

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables and config
load_dotenv()
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

def default_agent(state: AgentState) -> AgentState:
    """
    Default agent that handles general queries using the base LLM.
    """
    try:
        logger.info("Default agent started processing query: %s", state["query"])
        
        # Initialize LLM
        llm = ChatOpenAI(
            base_url=config["api"]["openrouter"]["base_url"],
            model_name=config["agents"]["default_agent"]["model"],
            temperature=config["agents"]["default_agent"]["temperature"],
            api_key=os.getenv("OPENROUTER_API_KEY")
        )
        
        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful financial assistant. You can help with general financial questions, 
            explain financial concepts, and provide guidance on investment strategies. 
            If you don't know something, be honest about it."""),
            ("human", "{query}")
        ])
        
        # Generate response
        logger.info("Generating response using LLM...")
        chain = prompt | llm
        response = chain.invoke({"query": state["query"]})
        logger.info("Generated response: %s", response.content)
        
        # Update state
        state["final_result"] = response.content
        state["agent"] = "default_agent"
        
        logger.info("Default agent completed successfully")
        return state
        
    except Exception as e:
        logger.error("Error in default agent: %s", str(e), exc_info=True)
        state["error"] = str(e)
        state["final_result"] = "I apologize, but I encountered an error while processing your query."
        return state 