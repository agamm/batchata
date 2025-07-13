"""Tests for MockProvider."""

import pytest
import time

from batchata.core import Job, JobResult
from batchata.exceptions import ValidationError
from tests.mocks import MockProvider


class TestMockProvider:
    """Tests for MockProvider."""
    
    def test_mock_provider_models(self):
        """Test mock provider model setup."""
        provider = MockProvider(auto_register=False)
        
        assert "mock-model-basic" in provider.models
        assert "mock-model-advanced" in provider.models
        assert "mock-model-simple" in provider.models
        
        # Check model capabilities
        advanced = provider.models["mock-model-advanced"]
        assert advanced.supports_images is True
        assert advanced.supports_files is True
        assert advanced.supports_citations is True
        assert ".pdf" in advanced.file_types
    
    def test_validate_job_success(self):
        """Test successful job validation."""
        provider = MockProvider(auto_register=False)
        
        job = Job(
            id="job-1",
            model="mock-model-basic",
            messages=[{"role": "user", "content": "Hello"}]
        )
        
        # Should not raise
        provider.validate_job(job)
    
    def test_validate_job_unsupported_model(self):
        """Test validation with unsupported model."""
        provider = MockProvider(auto_register=False)
        
        job = Job(
            id="job-1",
            model="unknown-model",
            messages=[{"role": "user", "content": "Hello"}]
        )
        
        with pytest.raises(ValidationError) as exc_info:
            provider.validate_job(job)
        
        assert "Model unknown-model not supported" in str(exc_info.value)
    
    def test_validate_job_file_not_supported(self):
        """Test validation when files not supported."""
        provider = MockProvider(auto_register=False)
        
        job = Job(
            id="job-1",
            model="mock-model-simple",  # Doesn't support files
            file="test.pdf",
            prompt="Summarize"
        )
        
        with pytest.raises(ValidationError) as exc_info:
            provider.validate_job(job)
        
        assert "does not support file inputs" in str(exc_info.value)
    
    def test_create_batch_success(self):
        """Test successful batch creation."""
        provider = MockProvider(auto_register=False)
        
        jobs = [
            Job(id="job-1", model="mock-model-basic", messages=[{"role": "user", "content": "Hi"}]),
            Job(id="job-2", model="mock-model-basic", messages=[{"role": "user", "content": "Hello"}])
        ]
        
        batch = provider.create_batch(jobs)
        
        assert batch.id.startswith("mock-batch-")
        assert batch.provider is provider
        assert len(batch.jobs) == 2
        assert batch.status == "pending"
    
    def test_mock_response_configuration(self):
        """Test configuring mock responses."""
        provider = MockProvider(auto_register=False)
        
        # Configure custom response
        custom_result = JobResult(
            job_id="job-1",
            response="Custom response",
            input_tokens=200,
            output_tokens=100,
            cost_usd=0.01
        )
        provider.set_mock_response("job-1", custom_result)
        
        # Create batch
        job = Job(id="job-1", model="mock-model-basic", messages=[{"role": "user", "content": "Hi"}])
        batch = provider.create_batch([job])
        
        # Wait and get results
        time.sleep(0.1)
        batch.get_status()
        
        if batch.is_complete:
            results = batch.get_results()
            assert results[0].response == "Custom response"
            assert results[0].input_tokens == 200
            assert results[0].cost_usd == 0.01
    
    def test_mock_delay(self):
        """Test mock delay functionality."""
        provider = MockProvider(auto_register=False)
        
        # Set delay
        provider.set_mock_delay("job-1", 0.5)
        
        job = Job(id="job-1", model="mock-model-basic", messages=[{"role": "user", "content": "Hi"}])
        batch = provider.create_batch([job])
        
        # Should still be pending/running
        assert batch.get_status() in ["pending", "running"]
        
        # Wait for delay
        time.sleep(0.6)
        assert batch.get_status() == "complete"
    
    def test_estimate_cost(self):
        """Test cost estimation."""
        provider = MockProvider(auto_register=False)
        
        jobs = [
            Job(id="job-1", model="mock-model-basic", messages=[{"role": "user", "content": "Hi"}]),
            Job(id="job-2", model="mock-model-basic", messages=[{"role": "user", "content": "Hello"}])
        ]
        
        cost = provider.estimate_cost(jobs)
        assert cost == 0.02  # $0.01 per job