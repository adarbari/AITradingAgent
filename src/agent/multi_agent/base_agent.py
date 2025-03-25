"""
Base agent class for the multi-agent trading system.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
import json

class AgentInput(BaseModel):
    """Base class for agent inputs"""
    request: str = Field(..., description="Request or instruction for the agent")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context information")

class AgentOutput(BaseModel):
    """Base class for agent outputs"""
    response: str = Field(..., description="Agent's response text")
    data: Optional[Dict[str, Any]] = Field(default=None, description="Structured data from the agent")
    confidence: Optional[float] = Field(default=None, description="Confidence score (0-1)")
    
    def to_json(self):
        """Convert to JSON string"""
        return json.dumps(self.dict(), indent=2)

class BaseAgent(ABC):
    """
    Base agent class for all specialized agents in the trading system.
    Each agent has specific capabilities and responsibilities.
    """
    
    def __init__(self, name: str, description: str, verbose: int = 0):
        """
        Initialize the base agent.
        
        Args:
            name (str): Name of the agent
            description (str): Description of the agent's role
            verbose (int): Verbosity level (0: silent, 1: normal, 2: detailed)
        """
        self.name = name
        self.description = description
        self.verbose = verbose
        self.memory = []  # Simple memory for past interactions
    
    @abstractmethod
    def process(self, input_data: AgentInput) -> AgentOutput:
        """
        Process the input and generate a response.
        
        Args:
            input_data (AgentInput): Input data for the agent
            
        Returns:
            AgentOutput: Agent's response
        """
        pass
    
    def add_to_memory(self, interaction: Dict[str, Any]):
        """
        Add an interaction to the agent's memory.
        
        Args:
            interaction (Dict): Interaction data to remember
        """
        self.memory.append(interaction)
        # Keep memory at a reasonable size
        if len(self.memory) > 100:
            self.memory = self.memory[-100:]
    
    def get_relevant_memories(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get memories relevant to the current query.
        Simple implementation that looks for keyword matches.
        
        Args:
            query (str): Query to match against memories
            limit (int): Maximum number of memories to return
            
        Returns:
            List[Dict]: List of relevant memories
        """
        # Simplistic relevance scoring based on word overlap
        query_words = set(query.lower().split())
        scored_memories = []
        
        for memory in self.memory:
            memory_text = memory.get("input", "") + " " + memory.get("output", "")
            memory_words = set(memory_text.lower().split())
            overlap = len(query_words.intersection(memory_words))
            if overlap > 0:
                scored_memories.append((overlap, id(memory), memory))
        
        # Sort by relevance score (descending) and memory id as a tie-breaker
        # then return just the memory objects
        relevant_memories = [m for _, _, m in sorted(scored_memories, reverse=True)][:limit]
        return relevant_memories
    
    def __str__(self):
        """String representation of the agent"""
        return f"{self.name}: {self.description}" 