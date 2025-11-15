"""
Base Agent class for OpenBB Comprehensive Agent

This module provides the abstract base class that all specialized agents inherit from.
It defines the common interface and functionality for agent lifecycle, tool management,
and request processing.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import logging
from comprehensive_agent.config import settings


# Configure logging
logger = logging.getLogger(__name__)


class AgentState(Enum):
    """Agent lifecycle states"""
    IDLE = "idle"
    INITIALIZING = "initializing"
    READY = "ready"
    PROCESSING = "processing"
    ERROR = "error"
    SHUTDOWN = "shutdown"


class BaseAgent(ABC):
    """
    Abstract base class for all specialized agents

    Provides common functionality for:
    - Agent lifecycle management
    - Tool registration and management
    - Request routing and processing
    - Capability matching
    - State management
    - Error handling
    """

    def __init__(
        self,
        name: str,
        description: str,
        capabilities: List[str],
        priority: int = 5
    ):
        """
        Initialize the base agent

        Args:
            name: Unique agent name
            description: Agent description
            capabilities: List of capabilities this agent provides
            priority: Agent priority (1-10, lower is higher priority)
        """
        self.name = name
        self.description = description
        self.capabilities = capabilities
        self.priority = priority
        self.state = AgentState.IDLE
        self.tools: Dict[str, Callable] = {}
        self.metadata: Dict[str, Any] = {}
        self.created_at = datetime.now()
        self.last_used = None
        self.request_count = 0
        self.error_count = 0

        logger.info(f"Initialized agent: {name}")

    @abstractmethod
    def can_handle(self, intent: str, context: Dict[str, Any]) -> float:
        """
        Determine if this agent can handle the given intent

        Args:
            intent: The classified intent type
            context: Request context including query, widget data, etc.

        Returns:
            Confidence score (0.0 to 1.0)
            - 0.0: Cannot handle this request
            - 1.0: Perfect match for this agent
        """
        pass

    @abstractmethod
    async def process(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the request using this agent's tools

        Args:
            query: User query string
            context: Request context with additional data

        Returns:
            Dict containing:
                - status: "success" or "error"
                - data: Processed result data
                - metadata: Agent and processing metadata
                - error: Error message if status is "error"
        """
        pass

    def register_tool(self, tool_name: str, tool_function: Callable) -> None:
        """
        Register a tool with this agent

        Args:
            tool_name: Unique name for the tool
            tool_function: Callable function that implements the tool
        """
        if tool_name in self.tools:
            logger.warning(f"Overwriting existing tool: {tool_name} in agent {self.name}")

        self.tools[tool_name] = tool_function
        logger.debug(f"Registered tool '{tool_name}' with agent '{self.name}'")

    def unregister_tool(self, tool_name: str) -> bool:
        """
        Unregister a tool from this agent

        Args:
            tool_name: Name of tool to remove

        Returns:
            True if tool was removed, False if it didn't exist
        """
        if tool_name in self.tools:
            del self.tools[tool_name]
            logger.debug(f"Unregistered tool '{tool_name}' from agent '{self.name}'")
            return True
        return False

    def get_tool(self, tool_name: str) -> Optional[Callable]:
        """
        Get a registered tool by name

        Args:
            tool_name: Name of the tool

        Returns:
            Tool function if found, None otherwise
        """
        return self.tools.get(tool_name)

    def has_tool(self, tool_name: str) -> bool:
        """
        Check if agent has a specific tool

        Args:
            tool_name: Name of the tool

        Returns:
            True if tool is registered, False otherwise
        """
        return tool_name in self.tools

    def has_capability(self, capability: str) -> bool:
        """
        Check if agent has a specific capability

        Args:
            capability: Capability to check

        Returns:
            True if agent has this capability
        """
        return capability in self.capabilities

    def set_state(self, state: AgentState) -> None:
        """
        Set the agent's current state

        Args:
            state: New agent state
        """
        old_state = self.state
        self.state = state
        logger.debug(f"Agent '{self.name}' state changed: {old_state.value} → {state.value}")

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get agent metadata for registry

        Returns:
            Dict containing agent information
        """
        return {
            "name": self.name,
            "description": self.description,
            "capabilities": self.capabilities,
            "priority": self.priority,
            "tools": list(self.tools.keys()),
            "state": self.state.value,
            "created_at": self.created_at.isoformat(),
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "request_count": self.request_count,
            "error_count": self.error_count,
            "metadata": self.metadata
        }

    def update_metrics(self, success: bool = True) -> None:
        """
        Update agent usage metrics

        Args:
            success: Whether the last request was successful
        """
        self.request_count += 1
        self.last_used = datetime.now()

        if not success:
            self.error_count += 1

    def get_success_rate(self) -> float:
        """
        Calculate agent success rate

        Returns:
            Success rate as percentage (0-100)
        """
        if self.request_count == 0:
            return 100.0

        success_count = self.request_count - self.error_count
        return (success_count / self.request_count) * 100.0

    async def initialize(self) -> bool:
        """
        Initialize the agent (can be overridden by subclasses)

        Returns:
            True if initialization successful
        """
        try:
            self.set_state(AgentState.INITIALIZING)
            # Subclasses can override to add initialization logic
            self.set_state(AgentState.READY)
            return True
        except Exception as e:
            logger.error(f"Failed to initialize agent '{self.name}': {e}")
            self.set_state(AgentState.ERROR)
            return False

    async def shutdown(self) -> None:
        """
        Shutdown the agent (can be overridden by subclasses)
        """
        try:
            self.set_state(AgentState.SHUTDOWN)
            logger.info(f"Agent '{self.name}' shutdown complete")
        except Exception as e:
            logger.error(f"Error during shutdown of agent '{self.name}': {e}")

    def __str__(self) -> str:
        """String representation of the agent"""
        return f"{self.name} ({self.state.value})"

    def __repr__(self) -> str:
        """Detailed representation of the agent"""
        return (
            f"<{self.__class__.__name__} "
            f"name='{self.name}' "
            f"state={self.state.value} "
            f"tools={len(self.tools)} "
            f"capabilities={len(self.capabilities)}>"
        )


class AgentError(Exception):
    """Base exception for agent errors"""

    def __init__(self, agent_name: str, message: str):
        self.agent_name = agent_name
        self.message = message
        super().__init__(f"[{agent_name}] {message}")


class AgentNotReadyError(AgentError):
    """Raised when agent is not in ready state"""
    pass


class AgentProcessingError(AgentError):
    """Raised when agent encounters error during processing"""
    pass


class ToolNotFoundError(AgentError):
    """Raised when requested tool is not registered"""

    def __init__(self, agent_name: str, tool_name: str):
        super().__init__(
            agent_name,
            f"Tool '{tool_name}' not found in agent '{agent_name}'"
        )
        self.tool_name = tool_name
