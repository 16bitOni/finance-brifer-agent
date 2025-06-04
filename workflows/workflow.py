from typing import Dict, Any
import logging
from datetime import datetime
import yaml
from dotenv import load_dotenv
import os
import sys
import asyncio

from state.state import AgentState
from agents.orchestrator_agent import smart_orchestrator  # Import the singleton instance directly

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class WorkflowManager:
    """Manages the workflow creation and execution."""
    
    def __init__(self):
        """Initialize the workflow manager."""
        try:
            # Import the orchestrator from the correct path
            from agents.orchestrator_agent import smart_orchestrator
            self.orchestrator = smart_orchestrator
            logger.info("Smart orchestrator loaded successfully")
        except ImportError as e:
            logger.error(f"Failed to import orchestrator: {e}")
            # Fallback - create a new instance if singleton import fails
            try:
                from agents.orchestrator_agent import SmartOrchestrator
                self.orchestrator = SmartOrchestrator()
                logger.info("Created new SmartOrchestrator instance")
            except ImportError:
                logger.error("Could not import SmartOrchestrator class either")
                self.orchestrator = None
    
    async def process(self, state: AgentState) -> AgentState:
        """Process the query through the workflow."""
        try:
            # Check if orchestrator is available
            if self.orchestrator is None:
                logger.error("No orchestrator available")
                state["error"] = "Orchestrator not available"
                state["final_result"] = "Service temporarily unavailable. Please try again later."
                return state
            
            # Add timestamp if not present
            if "timestamp" not in state:
                state["timestamp"] = datetime.now().isoformat()
            
            # Ensure required fields are present
            if "tool_results" not in state:
                state["tool_results"] = {}
            if "error" not in state:
                state["error"] = None
            if "final_result" not in state:
                state["final_result"] = None
            
            logger.info(f"Processing query: {state.get('query', 'Unknown')[:50]}...")
            
            # Process through orchestrator
            result = await self.orchestrator.process(state)
            
            # Validate result
            if not isinstance(result, dict):
                logger.warning("Orchestrator returned non-dict result")
                result = dict(result) if hasattr(result, '__iter__') else {"final_result": str(result)}
            
            # Ensure required fields in result
            if "final_result" not in result:
                result["final_result"] = "Query processed but no result generated."
            
            logger.info("Query processed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error in workflow: {str(e)}", exc_info=True)
            
            # Update state with error information
            state["error"] = str(e)
            state["final_result"] = "I apologize, but I encountered an error while processing your query."
            state["timestamp"] = datetime.now().isoformat()
            
            return state

async def process_query(query: str, user_id: str = "default") -> Dict[str, Any]:
    """Process a query through the workflow."""
    try:
        # Validate input
        if not query or not isinstance(query, str):
            return {
                "error": "Invalid query provided",
                "final_result": "Please provide a valid query.",
                "timestamp": datetime.now().isoformat()
            }
        
        # Initialize state
        state = AgentState({
            "query": query.strip(),
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
            "tool_results": {},
            "error": None,
            "final_result": None
        })
        
        logger.info(f"Initializing workflow for query: {query[:50]}...")
        
        # Initialize workflow manager
        try:
            manager = WorkflowManager()
        except Exception as e:
            logger.error(f"Failed to initialize WorkflowManager: {e}")
            return {
                "error": f"Initialization error: {str(e)}",
                "final_result": "Service initialization failed. Please try again later.",
                "timestamp": datetime.now().isoformat()
            }
        
        # Process query
        result = await manager.process(state)
        
        # Log success
        logger.info(f"Query processed successfully")
        
        # Ensure result is properly formatted
        if not isinstance(result, dict):
            result = {"final_result": str(result), "timestamp": datetime.now().isoformat()}
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        return {
            "error": str(e),
            "final_result": "I apologize, but I encountered an error while processing your query.",
            "timestamp": datetime.now().isoformat()
        }

# Health check function
async def health_check() -> bool:
    """Check if the workflow is properly configured."""
    try:
        manager = WorkflowManager()
        return manager.orchestrator is not None
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return False

# Example usage and testing
if __name__ == "__main__":
    async def test_workflow():
        """Test the workflow with various queries."""
        
        # First, run health check
        print("Running health check...")
        is_healthy = await health_check()
        print(f"System healthy: {is_healthy}")
        
        if not is_healthy:
            print("System not healthy. Check orchestrator integration.")
            return
        
        test_queries = [
            "What's the latest news?",
            "Show me Apple stock price",
            "What's in my portfolio?",
            "Hello, how are you?",
            "",  # Edge case
            "Tell me about TSLA earnings financial market"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n--- Test {i}: '{query}' ---")
            try:
                result = await process_query(query, "test_user")
                
                print(f"Success: {'error' not in result or result['error'] is None}")
                print(f"Result: {result.get('final_result', 'No result')}")
                
                if result.get('error'):
                    print(f"Error: {result['error']}")
                
                if result.get('tool_results'):
                    print(f"Tools used: {list(result['tool_results'].keys())}")
                    
                # Add delay between tests
                await asyncio.sleep(0.5)
                
            except Exception as e:
                print(f"Test failed with exception: {e}")
    
    # Test configuration check
    def check_imports():
        """Check if all required imports are available."""
        try:
            print("Checking imports...")
            
            # Check state
            from state.state import AgentState
            print("✓ AgentState import successful")
            
            # Check orchestrator
            try:
                from agents.orchestrator_agent import smart_orchestrator
                print("✓ smart_orchestrator import successful")
            except ImportError:
                try:
                    from agents.orchestrator_agent import SmartOrchestrator
                    print("✓ SmartOrchestrator class import successful")
                except ImportError:
                    print("✗ Orchestrator import failed - check file name and location")
                    return False
            
            return True
            
        except Exception as e:
            print(f"✗ Import check failed: {e}")
            return False
    
    # Run configuration check first
    print("=== Configuration Check ===")
    imports_ok = check_imports()
    
    if imports_ok:
        print("\n=== Running Workflow Tests ===")
        # Run tests if executed directly
        asyncio.run(test_workflow())
    else:
        print("\nPlease fix import issues before running tests.")
        print("\nPossible fixes:")
        print("1. Ensure orchestrator_agent.py is in the agents directory")
        print("2. Check if the file name matches the import statement")
        print("3. Verify SmartOrchestrator class is properly exported")