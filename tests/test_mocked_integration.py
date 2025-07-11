"""
Mocked integration tests that test the full flow without real API calls.

Tests the interaction between BatchManager, batch(), and providers
using completely mocked providers but real logic flow.
"""

import json
import os
import tempfile
import shutil
from unittest.mock import MagicMock, patch
from pydantic import BaseModel

from batchata.batch_manager import BatchManager
from batchata import batch


class IntegrationTestModel(BaseModel):
    """Test model for integration tests."""
    content: str
    score: float


class TestMockedIntegration:
    """Mocked integration tests with fully mocked providers."""
    
    def setup_method(self):
        """Setup for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.state_file = os.path.join(self.temp_dir, "integration_state.json")
        self.results_dir = os.path.join(self.temp_dir, "results")
        
    def teardown_method(self):
        """Cleanup after each test."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    @patch('batchata.utils.check_flat_model_for_citation_mapping')
    @patch('batchata.providers.get_provider_for_model')
    @patch('batchata.batch_manager.batch')
    def test_batch_manager_provider_coordination(self, mock_batch, mock_get_provider, mock_check_flat_model):
        """Test coordination between BatchManager and provider."""
        # Set up mock provider and utility functions
        mock_check_flat_model.return_value = None  # Mock utility function
        mock_provider = MagicMock()
        mock_get_provider.return_value = mock_provider
        
        # Set up provider responses - mock ALL methods to prevent any real API calls
        mock_provider.validate_batch.return_value = None
        mock_provider.validate_model_capabilities.return_value = None
        mock_provider.has_citations_enabled.return_value = False
        mock_provider.prepare_batch_requests.return_value = [
            {'custom_id': 'request_0', 'params': {}},
            {'custom_id': 'request_1', 'params': {}}
        ]
        mock_provider.create_batch.return_value = "integration_batch_123"
        mock_provider._is_batch_completed.return_value = True
        mock_provider.get_results.return_value = []
        mock_provider.parse_results.return_value = [
            {"result": IntegrationTestModel(content="test_content_0", score=1.5), "citations": None},
            {"result": IntegrationTestModel(content="test_content_1", score=3.0), "citations": None}
        ]
        mock_provider.get_batch_usage_costs.return_value = {"total_cost": 0.02}
        
        messages = [
            [{"role": "user", "content": "Test message 1"}],
            [{"role": "user", "content": "Test message 2"}]
        ]
        
        manager = BatchManager(
            messages=messages,
            model="claude-3-haiku-20240307",
            response_model=IntegrationTestModel,
            items_per_job=2,
            max_parallel_jobs=1,
            state_path=self.state_file
        )
        
        # Setup mock batch job that mimics the real behavior
        mock_batch_job = MagicMock()
        mock_batch_job.is_complete.return_value = True
        mock_batch_job.results.return_value = [
            {"result": IntegrationTestModel(content="test_content_0", score=1.5), "citations": None},
            {"result": IntegrationTestModel(content="test_content_1", score=3.0), "citations": None}
        ]
        mock_batch_job.stats.return_value = {"total_cost": 0.02}
        mock_batch_job._batch_id = "integration_batch_123"
        
        mock_batch.return_value = mock_batch_job
        
        # Run processing
        summary = manager.run(print_progress=False)
        
        # Verify basic functionality
        assert summary["completed_items"] == 2
        assert summary["total_cost"] == 0.02
        
    @patch('batchata.utils.check_flat_model_for_citation_mapping')
    @patch('batchata.core.get_provider_for_model')
    def test_batch_function_integration(self, mock_get_provider, mock_check_flat_model):
        """Test direct batch() function integration."""
        # Set up provider and utility functions
        mock_check_flat_model.return_value = None  # Mock utility function
        mock_provider = MagicMock()
        mock_get_provider.return_value = mock_provider
        
        mock_provider.validate_batch.return_value = None
        mock_provider.validate_model_capabilities.return_value = None
        mock_provider.has_citations_enabled.return_value = False
        mock_provider.prepare_batch_requests.return_value = [
            {'custom_id': 'request_0', 'params': {}}
        ]
        mock_provider.create_batch.return_value = "direct_batch_456"
        mock_provider._is_batch_completed.return_value = True
        mock_provider.get_results.return_value = []
        mock_provider.parse_results.return_value = [
            {"result": IntegrationTestModel(content="test_content", score=2.5), "citations": None}
        ]
        mock_provider.get_batch_usage_costs.return_value = {"total_cost": 0.01}
        
        messages = [
            [{"role": "user", "content": "Test message"}]
        ]
        
        job = batch(
            messages=messages,
            model="claude-3-haiku-20240307",
            response_model=IntegrationTestModel
        )
        
        # Verify provider was called with correct parameters
        mock_get_provider.assert_called_once_with("claude-3-haiku-20240307")
        mock_provider.validate_batch.assert_called_once_with(messages, IntegrationTestModel)
        
        # Verify job properties
        assert job._batch_id == "direct_batch_456"
        assert job._response_model == IntegrationTestModel
        
        # Verify results
        job_results = job.results()
        assert len(job_results) == 1
        assert isinstance(job_results[0]["result"], IntegrationTestModel)
        
    @patch('batchata.utils.check_flat_model_for_citation_mapping')
    @patch('batchata.batch_manager.batch')
    def test_state_persistence_integration(self, mock_batch, mock_check_flat_model):
        """Test state persistence through the full flow."""
        # Set up utility functions
        mock_check_flat_model.return_value = None  # Mock utility function
        
        # Setup mock batch job that returns immediately
        mock_batch_job = MagicMock()
        mock_batch_job.is_complete.return_value = True  # Complete immediately
        mock_batch_job.results.return_value = [
            {"result": IntegrationTestModel(content="test_content", score=1.0), "citations": None}
        ]
        mock_batch_job.stats.return_value = {"total_cost": 0.01}
        mock_batch_job._batch_id = "persistence_batch_101"
        
        mock_batch.return_value = mock_batch_job
        
        messages = [
            [{"role": "user", "content": "Message 1"}]
        ]
        
        # Create BatchManager
        manager = BatchManager(
            messages=messages,
            model="claude-3-haiku-20240307",
            items_per_job=1,
            state_path=self.state_file
        )
        
        # Verify state file was created
        assert os.path.exists(self.state_file)
        
        # Run processing
        summary = manager.run(print_progress=False)
        
        # Verify state was updated
        with open(self.state_file, 'r') as f:
            state_data = json.load(f)
            
        assert len(state_data["jobs"]) == 1
        
        # Create new manager from same state
        manager2 = BatchManager(
            messages=messages,  # Should be ignored
            model="claude-3-haiku-20240307",
            state_path=self.state_file
        )
        
        # Should load existing state
        assert manager2.state.manager_id == manager.state.manager_id