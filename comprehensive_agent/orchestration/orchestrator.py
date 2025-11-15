"""
Orchestrator for OpenBB Comprehensive Agent

Main coordination engine that routes user queries to specialized agents.
Handles intent classification, agent selection, parallel execution, and result aggregation.
"""

from typing import Dict, List, Any, Optional
import asyncio
import logging
from comprehensive_agent.orchestration.agent_registry import AgentRegistry
from comprehensive_agent.orchestration.intent_classifier import IntentClassifier, IntentType
from comprehensive_agent.orchestration.result_aggregator import ResultAggregator
from comprehensive_agent.orchestration.context_manager import ContextManager
from comprehensive_agent.agents.base_agent import BaseAgent, AgentState
from comprehensive_agent.config import settings


# Configure logging
logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Main orchestrator for routing queries to specialized agents

    Responsibilities:
    - Classify user query intent
    - Select appropriate agent(s)
    - Execute agents in parallel when needed
    - Aggregate results from multiple agents
    - Manage context and session state
    """

    def __init__(
        self,
        registry: AgentRegistry,
        context_manager: Optional[ContextManager] = None
    ):
        """
        Initialize the orchestrator

        Args:
            registry: Agent registry instance
            context_manager: Optional context manager (creates new if not provided)
        """
        self.registry = registry
        self.intent_classifier = IntentClassifier()
        self.result_aggregator = ResultAggregator()
        self.context_manager = context_manager or ContextManager()

        # Agent selection configuration
        self.min_confidence = 0.5
        self.max_parallel_agents = settings.max_parallel_agents

        logger.info(
            f"Orchestrator initialized (max_parallel_agents={self.max_parallel_agents})"
        )

    async def process_query(
        self,
        query: str,
        session_id: str = "default",
        context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Main orchestration logic for processing user queries

        Flow:
        1. Build query context
        2. Classify intent
        3. Select appropriate agent(s)
        4. Execute agent(s) in parallel if needed
        5. Aggregate results
        6. Update session history
        7. Return unified response

        Args:
            query: User query string
            session_id: Session identifier for conversation tracking
            context: Optional additional context data

        Returns:
            Unified response dictionary with:
                - status: Overall status
                - data: Combined data
                - insights: Aggregated insights
                - visualizations: Charts and tables
                - metadata: Processing metadata
        """
        start_time = asyncio.get_event_loop().time()

        try:
            # Step 1: Build complete query context
            logger.info(f"Processing query for session {session_id}: {query[:100]}...")

            query_context = self.context_manager.build_query_context(
                session_id=session_id,
                query=query,
                additional_context=context
            )

            # Step 2: Classify intent
            intents = self.intent_classifier.classify(query, query_context)
            is_multi_agent = self.intent_classifier.is_multi_agent(intents)
            agent_intents = self.intent_classifier.get_agent_intents(intents)

            logger.info(
                f"Classified intents: {[i.value for i in intents]} "
                f"(multi_agent={is_multi_agent})"
            )

            # Step 3: Select appropriate agent(s)
            selected_agents = await self._select_agents(
                agent_intents if agent_intents else intents,
                query_context
            )

            if not selected_agents:
                logger.warning("No suitable agents found for query")
                return self._no_agent_result(query, intents)

            logger.info(
                f"Selected {len(selected_agents)} agent(s): "
                f"{[agent.name for agent in selected_agents]}"
            )

            # Step 4: Execute agent(s)
            if len(selected_agents) == 1:
                # Single agent execution
                agent_results = await self._execute_single_agent(
                    selected_agents[0],
                    query,
                    query_context
                )
            else:
                # Parallel multi-agent execution
                agent_results = await self._execute_multiple_agents(
                    selected_agents,
                    query,
                    query_context
                )

            # Step 5: Aggregate results
            aggregated = self.result_aggregator.aggregate(agent_results)

            # Step 6: Update session history
            self.context_manager.add_to_history(
                session_id=session_id,
                role="human",
                content=query,
                metadata={"intents": [i.value for i in intents]}
            )

            # Calculate processing time
            processing_time = asyncio.get_event_loop().time() - start_time

            # Add orchestration metadata
            aggregated.setdefault("metadata", {}).update({
                "orchestration": {
                    "intents": [i.value for i in intents],
                    "is_multi_agent": is_multi_agent,
                    "agents_selected": [agent.name for agent in selected_agents],
                    "processing_time_seconds": round(processing_time, 3)
                }
            })

            logger.info(
                f"Query processed successfully in {processing_time:.3f}s "
                f"using {len(selected_agents)} agent(s)"
            )

            return aggregated

        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            return self._error_result(str(e))

    async def _select_agents(
        self,
        intents: List[IntentType],
        context: Dict
    ) -> List[BaseAgent]:
        """
        Select appropriate agents based on intents

        Args:
            intents: List of classified intents
            context: Query context

        Returns:
            List of selected agent instances
        """
        selected_agents = []

        # Map intents to agent names
        intent_to_agent = {
            IntentType.MARKET_DATA: "MarketDataAgent",
            IntentType.TECHNICAL_ANALYSIS: "TechnicalAnalysisAgent",
            IntentType.FUNDAMENTAL_ANALYSIS: "FundamentalAnalysisAgent",
            IntentType.RISK_ASSESSMENT: "RiskAnalyticsAgent",
            IntentType.PORTFOLIO_ANALYSIS: "PortfolioManagementAgent",
            IntentType.ECONOMIC_FORECAST: "EconomicAnalysisAgent",
            IntentType.NEWS_SENTIMENT: "NewsSentimentAgent"
        }

        # Get agents for each intent
        for intent in intents:
            if intent in (IntentType.MULTI_AGENT, IntentType.GENERAL):
                continue  # Skip markers

            agent_name = intent_to_agent.get(intent)
            if agent_name:
                agent = self.registry.get_agent(agent_name)

                if agent and agent.state == AgentState.READY:
                    # Check if agent can actually handle this request
                    confidence = agent.can_handle(intent.value, context)

                    if confidence >= self.min_confidence:
                        selected_agents.append(agent)
                        logger.debug(
                            f"Selected {agent_name} for {intent.value} "
                            f"(confidence: {confidence:.2f})"
                        )
                    else:
                        logger.debug(
                            f"Agent {agent_name} confidence too low: {confidence:.2f}"
                        )
                elif agent:
                    logger.warning(f"Agent {agent_name} not ready (state: {agent.state})")
                else:
                    logger.warning(f"Agent {agent_name} not found in registry")

        # Limit to max parallel agents
        if len(selected_agents) > self.max_parallel_agents:
            logger.warning(
                f"Too many agents selected ({len(selected_agents)}), "
                f"limiting to {self.max_parallel_agents}"
            )
            # Sort by priority and take top N
            selected_agents = sorted(
                selected_agents,
                key=lambda a: a.priority
            )[:self.max_parallel_agents]

        return selected_agents

    async def _execute_single_agent(
        self,
        agent: BaseAgent,
        query: str,
        context: Dict
    ) -> List[Dict[str, Any]]:
        """
        Execute a single agent

        Args:
            agent: Agent to execute
            query: User query
            context: Query context

        Returns:
            List with single result dictionary
        """
        logger.debug(f"Executing single agent: {agent.name}")

        try:
            agent.set_state(AgentState.PROCESSING)
            result = await agent.process(query, context)
            agent.set_state(AgentState.READY)
            agent.update_metrics(success=True)

            return [{
                "agent": agent.name,
                "result": result
            }]

        except Exception as e:
            logger.error(f"Error executing agent {agent.name}: {e}", exc_info=True)
            agent.set_state(AgentState.ERROR)
            agent.update_metrics(success=False)

            return [{
                "agent": agent.name,
                "error": str(e)
            }]

    async def _execute_multiple_agents(
        self,
        agents: List[BaseAgent],
        query: str,
        context: Dict
    ) -> List[Dict[str, Any]]:
        """
        Execute multiple agents in parallel

        Args:
            agents: List of agents to execute
            query: User query
            context: Query context

        Returns:
            List of result dictionaries
        """
        logger.debug(f"Executing {len(agents)} agents in parallel")

        # Create tasks for parallel execution
        tasks = []
        for agent in agents:
            task = self._execute_agent_with_error_handling(agent, query, context)
            tasks.append(task)

        # Execute in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        agent_results = []
        for agent, result in zip(agents, results):
            if isinstance(result, Exception):
                logger.error(f"Agent {agent.name} raised exception: {result}")
                agent_results.append({
                    "agent": agent.name,
                    "error": str(result)
                })
            else:
                agent_results.append(result)

        return agent_results

    async def _execute_agent_with_error_handling(
        self,
        agent: BaseAgent,
        query: str,
        context: Dict
    ) -> Dict[str, Any]:
        """
        Execute agent with error handling

        Args:
            agent: Agent to execute
            query: User query
            context: Query context

        Returns:
            Result dictionary with agent name and result/error
        """
        try:
            logger.debug(f"Executing agent: {agent.name}")
            agent.set_state(AgentState.PROCESSING)

            # Execute with timeout if configured
            timeout = settings.agent_timeout
            result = await asyncio.wait_for(
                agent.process(query, context),
                timeout=timeout
            )

            agent.set_state(AgentState.READY)
            agent.update_metrics(success=True)

            return {
                "agent": agent.name,
                "result": result
            }

        except asyncio.TimeoutError:
            logger.error(f"Agent {agent.name} timed out after {timeout}s")
            agent.set_state(AgentState.ERROR)
            agent.update_metrics(success=False)

            return {
                "agent": agent.name,
                "error": f"Agent timed out after {timeout} seconds"
            }

        except Exception as e:
            logger.error(f"Error executing agent {agent.name}: {e}", exc_info=True)
            agent.set_state(AgentState.ERROR)
            agent.update_metrics(success=False)

            return {
                "agent": agent.name,
                "error": str(e)
            }

    def _no_agent_result(self, query: str, intents: List[IntentType]) -> Dict[str, Any]:
        """
        Generate result when no suitable agent is found

        Args:
            query: User query
            intents: Classified intents

        Returns:
            Error result dictionary
        """
        return {
            "status": "error",
            "error": "No suitable agent found to handle this request",
            "data": {},
            "metadata": {
                "query": query,
                "intents": [i.value for i in intents],
                "available_agents": [
                    agent.name for agent in self.registry.get_ready_agents()
                ]
            }
        }

    def _error_result(self, error_message: str) -> Dict[str, Any]:
        """
        Generate error result

        Args:
            error_message: Error message

        Returns:
            Error result dictionary
        """
        return {
            "status": "error",
            "error": error_message,
            "data": {},
            "metadata": {}
        }

    async def initialize(self) -> bool:
        """
        Initialize the orchestrator and all agents

        Returns:
            True if initialization successful
        """
        try:
            logger.info("Initializing orchestrator and agents...")

            # Initialize all registered agents
            results = await self.registry.initialize_all_agents()

            successful = sum(1 for success in results.values() if success)
            total = len(results)

            logger.info(f"Initialized {successful}/{total} agents successfully")

            # Check if we have at least some agents ready
            ready_agents = len(self.registry.get_ready_agents())
            if ready_agents == 0:
                logger.error("No agents are ready after initialization")
                return False

            logger.info(f"Orchestrator initialization complete ({ready_agents} agents ready)")
            return True

        except Exception as e:
            logger.error(f"Error initializing orchestrator: {e}", exc_info=True)
            return False

    async def shutdown(self) -> None:
        """Shutdown the orchestrator and all agents"""
        try:
            logger.info("Shutting down orchestrator...")

            # Shutdown all agents
            await self.registry.shutdown_all_agents()

            # Clear context
            self.context_manager.clear_cache()

            logger.info("Orchestrator shutdown complete")

        except Exception as e:
            logger.error(f"Error during orchestrator shutdown: {e}", exc_info=True)

    def get_status(self) -> Dict[str, Any]:
        """
        Get orchestrator status

        Returns:
            Status dictionary with orchestrator and agent information
        """
        registry_metadata = self.registry.get_registry_metadata()
        context_stats = self.context_manager.get_stats()

        return {
            "orchestrator": {
                "max_parallel_agents": self.max_parallel_agents,
                "min_confidence": self.min_confidence
            },
            "registry": registry_metadata,
            "context_manager": context_stats,
            "agents": [
                {
                    "name": agent.name,
                    "state": agent.state.value,
                    "success_rate": agent.get_success_rate(),
                    "request_count": agent.request_count,
                    "tools": len(agent.tools)
                }
                for agent in self.registry.get_all_agents()
            ]
        }

    def __repr__(self) -> str:
        """String representation of the orchestrator"""
        return (
            f"<Orchestrator agents={len(self.registry)} "
            f"sessions={len(self.context_manager.sessions)}>"
        )
