from typing import TypedDict, List, Dict, Any, Optional
from pydantic import BaseModel

class AgentState(TypedDict):
    """State for the LangGraph workflow."""
    query: str
    agent: str
    tool_results: List[Dict[str, Any]]
    final_result: Optional[str]
    error: Optional[str]

class Query(BaseModel):
    """User query model."""
    text: str
    user_id: str
    timestamp: float

class ToolResult(BaseModel):
    """Result from a tool execution."""
    tool_name: str
    result: Any
    metadata: Dict[str, Any]

class AgentResponse(BaseModel):
    """Response from an agent."""
    agent_name: str
    response: str
    confidence: float
    tool_results: List[ToolResult] 