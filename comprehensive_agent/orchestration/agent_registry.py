"""
Agent Registry for OpenBB Comprehensive Agent

This module provides a central registry for discovering, managing, and accessing
all available agents in the system.
"""

from typing import Dict, List, Optional
import logging
from comprehensive_agent.agents.base_agent import BaseAgent, AgentState
from comprehensive_agent.config import settings


# Configure logging
logger = logging.getLogger(__name__)


class AgentRegistry:
    """
    Central registry for all available agents

    Manages agent lifecycle, discovery, and access. Provides methods to:
    - Register and unregister agents
    - Find agents by name, capability, or priority
    - Query agent status and metadata
    - Manage agent lifecycle
    """

    def __init__(self):
        """Initialize the agent registry"""
        self.agents: Dict[str, BaseAgent] = {}
        self._capability_index: Dict[str, List[str]] = {}
        logger.info("Agent registry initialized")

    def register(self, agent: BaseAgent) -> bool:
        """
        Register a new agent

        Args:
            agent: Agent instance to register

        Returns:
            True if registration successful, False if agent already exists

        Raises:
            ValueError: If agent is None or has invalid name
        """
        if agent is None:
            raise ValueError("Cannot register None as agent")

        if not agent.name:
            raise ValueError("Agent must have a name")

        if agent.name in self.agents:
            logger.warning(f"Agent '{agent.name}' already registered. Skipping.")
            return False

        # Register the agent
        self.agents[agent.name] = agent

        # Index by capabilities for fast lookup
        for capability in agent.capabilities:
            if capability not in self._capability_index:
                self._capability_index[capability] = []
            self._capability_index[capability].append(agent.name)

        logger.info(
            f"Registered agent '{agent.name}' with {len(agent.capabilities)} capabilities "
            f"and {len(agent.tools)} tools"
        )

        return True

    def unregister(self, agent_name: str) -> bool:
        """
        Unregister an agent

        Args:
            agent_name: Name of agent to unregister

        Returns:
            True if agent was unregistered, False if not found
        """
        if agent_name not in self.agents:
            logger.warning(f"Cannot unregister unknown agent: {agent_name}")
            return False

        agent = self.agents[agent_name]

        # Remove from capability index
        for capability in agent.capabilities:
            if capability in self._capability_index:
                self._capability_index[capability].remove(agent_name)
                if not self._capability_index[capability]:
                    del self._capability_index[capability]

        # Remove from agents dict
        del self.agents[agent_name]

        logger.info(f"Unregistered agent: {agent_name}")
        return True

    def get_agent(self, agent_name: str) -> Optional[BaseAgent]:
        """
        Get agent by name

        Args:
            agent_name: Name of the agent

        Returns:
            Agent instance if found, None otherwise
        """
        return self.agents.get(agent_name)

    def has_agent(self, agent_name: str) -> bool:
        """
        Check if agent is registered

        Args:
            agent_name: Name of the agent

        Returns:
            True if agent is registered
        """
        return agent_name in self.agents

    def get_all_agents(self) -> List[BaseAgent]:
        """
        Get all registered agents

        Returns:
            List of all agent instances
        """
        return list(self.agents.values())

    def get_agent_names(self) -> List[str]:
        """
        Get names of all registered agents

        Returns:
            List of agent names
        """
        return list(self.agents.keys())

    def get_agents_by_capability(self, capability: str) -> List[BaseAgent]:
        """
        Find all agents with a specific capability

        Args:
            capability: Capability to search for

        Returns:
            List of agents that have this capability
        """
        agent_names = self._capability_index.get(capability, [])
        return [self.agents[name] for name in agent_names if name in self.agents]

    def get_agents_by_state(self, state: AgentState) -> List[BaseAgent]:
        """
        Get all agents in a specific state

        Args:
            state: Agent state to filter by

        Returns:
            List of agents in the specified state
        """
        return [agent for agent in self.agents.values() if agent.state == state]

    def get_ready_agents(self) -> List[BaseAgent]:
        """
        Get all agents that are ready to process requests

        Returns:
            List of agents in READY state
        """
        return self.get_agents_by_state(AgentState.READY)

    def get_agents_by_priority(self, ascending: bool = True) -> List[BaseAgent]:
        """
        Get agents sorted by priority

        Args:
            ascending: If True, return low priority numbers first (higher priority)

        Returns:
            List of agents sorted by priority
        """
        return sorted(
            self.agents.values(),
            key=lambda agent: agent.priority,
            reverse=not ascending
        )

    def find_best_agent(
        self,
        intent: str,
        context: Dict,
        min_confidence: float = 0.5
    ) -> Optional[BaseAgent]:
        """
        Find the best agent for handling a given intent

        Args:
            intent: The intent type
            context: Request context
            min_confidence: Minimum confidence score required

        Returns:
            Best matching agent or None if no suitable agent found
        """
        best_agent = None
        best_score = min_confidence

        # Only consider ready agents
        ready_agents = self.get_ready_agents()

        for agent in ready_agents:
            try:
                score = agent.can_handle(intent, context)

                if score > best_score:
                    best_score = score
                    best_agent = agent

            except Exception as e:
                logger.error(f"Error checking if agent '{agent.name}' can handle intent: {e}")
                continue

        if best_agent:
            logger.debug(
                f"Best agent for intent '{intent}': {best_agent.name} (score: {best_score:.2f})"
            )

        return best_agent

    def find_agents_by_intent(
        self,
        intent: str,
        context: Dict,
        min_confidence: float = 0.5,
        max_agents: int = 3
    ) -> List[tuple[BaseAgent, float]]:
        """
        Find multiple agents that can handle an intent, ranked by confidence

        Args:
            intent: The intent type
            context: Request context
            min_confidence: Minimum confidence score required
            max_agents: Maximum number of agents to return

        Returns:
            List of (agent, confidence_score) tuples, sorted by confidence descending
        """
        agent_scores = []

        # Only consider ready agents
        ready_agents = self.get_ready_agents()

        for agent in ready_agents:
            try:
                score = agent.can_handle(intent, context)

                if score >= min_confidence:
                    agent_scores.append((agent, score))

            except Exception as e:
                logger.error(f"Error checking if agent '{agent.name}' can handle intent: {e}")
                continue

        # Sort by confidence score descending
        agent_scores.sort(key=lambda x: x[1], reverse=True)

        # Limit to max_agents
        result = agent_scores[:max_agents]

        if result:
            logger.debug(
                f"Found {len(result)} agents for intent '{intent}': "
                f"{[(a.name, f'{s:.2f}') for a, s in result]}"
            )

        return result

    def get_registry_metadata(self) -> Dict:
        """
        Get complete registry metadata

        Returns:
            Dict containing registry statistics and agent information
        """
        all_agents = self.get_all_agents()

        return {
            "total_agents": len(all_agents),
            "ready_agents": len(self.get_ready_agents()),
            "agents_by_state": {
                state.value: len(self.get_agents_by_state(state))
                for state in AgentState
            },
            "total_capabilities": len(self._capability_index),
            "agents": [agent.get_metadata() for agent in all_agents]
        }

    async def initialize_all_agents(self) -> Dict[str, bool]:
        """
        Initialize all registered agents

        Returns:
            Dict mapping agent names to initialization success status
        """
        results = {}

        for agent_name, agent in self.agents.items():
            try:
                logger.info(f"Initializing agent: {agent_name}")
                success = await agent.initialize()
                results[agent_name] = success

                if success:
                    logger.info(f"Agent '{agent_name}' initialized successfully")
                else:
                    logger.error(f"Agent '{agent_name}' initialization failed")

            except Exception as e:
                logger.error(f"Error initializing agent '{agent_name}': {e}")
                results[agent_name] = False

        successful = sum(1 for success in results.values() if success)
        logger.info(f"Initialized {successful}/{len(results)} agents successfully")

        return results

    async def shutdown_all_agents(self) -> None:
        """Shutdown all registered agents"""
        logger.info("Shutting down all agents...")

        for agent_name, agent in self.agents.items():
            try:
                await agent.shutdown()
                logger.info(f"Agent '{agent_name}' shutdown complete")
            except Exception as e:
                logger.error(f"Error shutting down agent '{agent_name}': {e}")

        logger.info("All agents shutdown complete")

    def clear(self) -> None:
        """Clear all agents from the registry"""
        logger.warning("Clearing agent registry")
        self.agents.clear()
        self._capability_index.clear()

    def __len__(self) -> int:
        """Return number of registered agents"""
        return len(self.agents)

    def __contains__(self, agent_name: str) -> bool:
        """Check if agent is in registry"""
        return agent_name in self.agents

    def __repr__(self) -> str:
        """Detailed representation of the registry"""
        return (
            f"<AgentRegistry agents={len(self.agents)} "
            f"capabilities={len(self._capability_index)}>"
        )
