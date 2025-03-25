"""
Tests for the BaseAgent class and related models (AgentInput, AgentOutput)
"""
import pytest
from unittest.mock import MagicMock
import json

from src.agent.multi_agent.base_agent import BaseAgent, AgentInput, AgentOutput


class TestAgentInput:
    """Test cases for the AgentInput model"""
    
    def test_init_with_required_fields(self):
        """Test initialization with only required fields"""
        agent_input = AgentInput(request="Analyze AAPL stock")
        
        assert agent_input.request == "Analyze AAPL stock"
        assert agent_input.context is None
    
    def test_init_with_all_fields(self):
        """Test initialization with all fields"""
        context = {"symbol": "AAPL", "date_range": {"start": "2023-01-01", "end": "2023-12-31"}}
        agent_input = AgentInput(request="Analyze AAPL stock", context=context)
        
        assert agent_input.request == "Analyze AAPL stock"
        assert agent_input.context == context
        assert agent_input.context["symbol"] == "AAPL"


class TestAgentOutput:
    """Test cases for the AgentOutput model"""
    
    def test_init_with_required_fields(self):
        """Test initialization with only required fields"""
        agent_output = AgentOutput(response="AAPL analysis completed")
        
        assert agent_output.response == "AAPL analysis completed"
        assert agent_output.data is None
        assert agent_output.confidence is None
    
    def test_init_with_all_fields(self):
        """Test initialization with all fields"""
        data = {"price": 150.0, "trend": "bullish"}
        agent_output = AgentOutput(
            response="AAPL analysis completed",
            data=data,
            confidence=0.85
        )
        
        assert agent_output.response == "AAPL analysis completed"
        assert agent_output.data == data
        assert agent_output.confidence == 0.85
    
    def test_to_json(self):
        """Test JSON serialization"""
        data = {"price": 150.0, "trend": "bullish"}
        agent_output = AgentOutput(
            response="AAPL analysis completed",
            data=data,
            confidence=0.85
        )
        
        json_str = agent_output.to_json()
        parsed = json.loads(json_str)
        
        assert parsed["response"] == "AAPL analysis completed"
        assert parsed["data"]["price"] == 150.0
        assert parsed["data"]["trend"] == "bullish"
        assert parsed["confidence"] == 0.85


class MockAgent(BaseAgent):
    """Mock implementation of BaseAgent for testing"""
    
    def process(self, input_data):
        """Mock implementation of process"""
        return AgentOutput(
            response=f"Processed: {input_data.request}",
            data={"processed": True},
            confidence=0.9
        )


class TestBaseAgent:
    """Test cases for the BaseAgent abstract class"""
    
    def test_init(self):
        """Test initialization of a concrete agent class"""
        agent = MockAgent(name="TestAgent", description="A test agent", verbose=2)
        
        assert agent.name == "TestAgent"
        assert agent.description == "A test agent"
        assert agent.verbose == 2
        assert agent.memory == []
    
    def test_abstract_methods(self):
        """Test that abstract methods need to be implemented"""
        # Verify that we cannot instantiate the abstract class directly
        with pytest.raises(TypeError):
            BaseAgent(name="AbstractAgent", description="Cannot instantiate")
    
    def test_process_method(self):
        """Test the process method in a concrete implementation"""
        agent = MockAgent(name="TestAgent", description="A test agent")
        input_data = AgentInput(request="test request")
        
        output = agent.process(input_data)
        
        assert isinstance(output, AgentOutput)
        assert output.response == "Processed: test request"
        assert output.data == {"processed": True}
        assert output.confidence == 0.9
    
    def test_memory_operations(self):
        """Test adding to and retrieving from memory"""
        agent = MockAgent(name="TestAgent", description="A test agent")
        
        # Add some interactions to memory
        agent.add_to_memory({"input": "query about AAPL", "output": "AAPL analysis"})
        agent.add_to_memory({"input": "query about MSFT", "output": "MSFT analysis"})
        agent.add_to_memory({"input": "query about market", "output": "Market overview"})
        
        assert len(agent.memory) == 3
        
        # Test memory retrieval with relevance
        relevant = agent.get_relevant_memories("Tell me about AAPL stock")
        assert len(relevant) > 0
        assert "AAPL" in relevant[0]["input"] or "AAPL" in relevant[0]["output"]
    
    def test_memory_limit(self):
        """Test that memory is limited to a reasonable size"""
        agent = MockAgent(name="TestAgent", description="A test agent")
        
        # Add many items to memory
        for i in range(150):
            agent.add_to_memory({"input": f"query {i}", "output": f"response {i}"})
        
        # Memory should be limited
        assert len(agent.memory) == 100
        
        # The newest items should be kept
        assert "query 149" in agent.memory[-1]["input"]
    
    def test_string_representation(self):
        """Test string representation of the agent"""
        agent = MockAgent(name="TestAgent", description="A test agent")
        
        assert str(agent) == "TestAgent: A test agent" 