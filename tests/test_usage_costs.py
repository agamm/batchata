import pytest
from unittest.mock import patch, MagicMock
from pydantic import BaseModel
from src import batch
from src.providers.anthropic import AnthropicBatchProvider
from src.batch_job import BatchJob


class CostTestModel(BaseModel):
    result: str


class TestUsageCosts:
    """Test usage cost tracking functionality."""
    
    def test_extract_usage_from_api_response(self):
        """Test extracting usage data from API response."""
        # Mock API response with usage data
        mock_result = MagicMock()
        mock_result.result.type = "succeeded"
        mock_result.result.message.content = [MagicMock(text="Test response")]
        mock_result.result.message.usage = MagicMock()
        mock_result.result.message.usage.input_tokens = 100
        mock_result.result.message.usage.output_tokens = 50
        mock_result.result.message.usage.service_tier = "batch"
        
        provider = AnthropicBatchProvider()
        usage_data = provider._extract_usage_from_result(mock_result)
        
        assert usage_data["input_tokens"] == 100
        assert usage_data["output_tokens"] == 50
        assert usage_data["service_tier"] == "batch"
    
    def test_calculate_cost_with_tokencost(self):
        """Test cost calculation using tokencost library."""
        provider = AnthropicBatchProvider()
        
        # Test input tokens cost
        input_cost = provider._calculate_token_cost(100, "claude-3-5-sonnet-20241022", "input")
        assert input_cost > 0
        
        # Test output tokens cost
        output_cost = provider._calculate_token_cost(50, "claude-3-5-sonnet-20241022", "output")
        assert output_cost > 0
        assert output_cost > input_cost  # Output tokens should be more expensive
    
    def test_apply_batch_discount(self):
        """Test applying 50% batch discount when service_tier is batch."""
        provider = AnthropicBatchProvider()
        
        # Test batch discount
        batch_cost = provider._apply_batch_discount(10.0, "batch")
        assert batch_cost == 5.0
        
        # Test no discount for standard tier
        standard_cost = provider._apply_batch_discount(10.0, "standard")
        assert standard_cost == 10.0
    
    def test_aggregate_batch_costs(self):
        """Test aggregating costs across multiple requests in a batch."""
        # Mock multiple API results with usage data
        mock_results = []
        for i in range(3):
            mock_result = MagicMock()
            mock_result.result.type = "succeeded"
            mock_result.result.message.content = [MagicMock(text=f"Response {i}")]
            mock_result.result.message.usage = MagicMock()
            mock_result.result.message.usage.input_tokens = 100 * (i + 1)
            mock_result.result.message.usage.output_tokens = 50 * (i + 1)
            mock_result.result.message.usage.service_tier = "batch"
            mock_results.append(mock_result)
        
        provider = AnthropicBatchProvider()
        total_usage = provider._aggregate_batch_usage(mock_results)
        
        assert total_usage["total_input_tokens"] == 600  # 100 + 200 + 300
        assert total_usage["total_output_tokens"] == 300  # 50 + 100 + 150
        assert total_usage["service_tier"] == "batch"
        assert total_usage["request_count"] == 3
    
    @patch('src.core.get_provider_for_model')
    def test_batch_job_stats_includes_costs(self, mock_provider_func):
        """Test that BatchJob.stats() includes cost information."""
        # Mock provider instance
        mock_provider = MagicMock()
        mock_provider_func.return_value = mock_provider
        mock_provider.validate_batch.return_value = None
        mock_provider.prepare_batch_requests.return_value = [{'custom_id': 'request_0', 'params': {}}]
        mock_provider.create_batch.return_value = "batch_123"
        mock_provider.has_citations_enabled.return_value = False
        mock_provider._is_batch_completed.return_value = True
        mock_provider.get_results.return_value = []
        mock_provider.parse_results.return_value = [{"result": CostTestModel(result="test"), "citations": None}]
        
        # Mock cost data
        mock_provider.get_batch_usage_costs.return_value = {
            "total_input_tokens": 100,
            "total_output_tokens": 50,
            "input_cost": 0.15,
            "output_cost": 0.375,
            "total_cost": 0.525,
            "service_tier": "batch",
            "request_count": 1
        }
        
        messages = [[{"role": "user", "content": "Test message"}]]
        
        job = batch(
            messages=messages,
            model="claude-3-5-sonnet-20241022",
            response_model=CostTestModel
        )
        
        stats = job.stats()
        
        # Verify cost information is included
        assert "total_input_tokens" in stats
        assert "total_output_tokens" in stats
        assert "input_cost" in stats
        assert "output_cost" in stats
        assert "total_cost" in stats
        assert "service_tier" in stats
        assert stats["total_input_tokens"] == 100
        assert stats["total_output_tokens"] == 50
        assert stats["total_cost"] == 0.525
    
    def test_empty_batch_cost_handling(self):
        """Test cost handling for empty batches."""
        job = batch(
            messages=[],
            model="claude-3-5-sonnet-20241022",
            response_model=CostTestModel
        )
        
        stats = job.stats()
        
        # Empty batch should have zero costs
        assert stats.get("total_input_tokens", 0) == 0
        assert stats.get("total_output_tokens", 0) == 0
        assert stats.get("total_cost", 0) == 0
    
    def test_different_model_pricing(self):
        """Test cost calculation with different model pricing."""
        provider = AnthropicBatchProvider()
        
        # Test different models have different pricing
        sonnet_cost = provider._calculate_token_cost(100, "claude-3-5-sonnet-20241022", "input")
        haiku_cost = provider._calculate_token_cost(100, "claude-3-5-haiku-20241022", "input")
        
        # Sonnet should be more expensive than Haiku
        assert sonnet_cost > haiku_cost
    
    def test_batch_vs_standard_pricing(self):
        """Test that batch pricing is 50% of standard pricing."""
        provider = AnthropicBatchProvider()
        
        base_cost = 10.0
        batch_cost = provider._apply_batch_discount(base_cost, "batch")
        standard_cost = provider._apply_batch_discount(base_cost, "standard")
        
        assert batch_cost == base_cost * 0.5
        assert standard_cost == base_cost