from typing import Any, Dict, Optional
from pydantic import BaseModel

class ToolResult(BaseModel):
    """Base class for tool results"""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class BaseTool:
    """Base class for all tools"""
    name: str
    description: str
    
    def __init__(self):
        self.name = self.__class__.__name__
        self.description = self.__class__.__doc__ or ""
    
    async def execute(self, **kwargs) -> ToolResult:
        """
        Execute the tool with given parameters
        Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement execute method")
    
    def validate_params(self, **kwargs) -> bool:
        """
        Validate the input parameters
        Can be overridden by subclasses for specific validation
        """
        return True
    
    def format_result(self, data: Dict[str, Any]) -> str:
        """
        Format the result for display
        Can be overridden by subclasses for specific formatting
        """
        return str(data) 