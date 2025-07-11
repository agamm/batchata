"""
Test fixtures for bachata tests.

Provides realistic mock implementations that match the real API behavior.
"""

from unittest.mock import Mock, MagicMock
from typing import List, Dict, Any, Optional, Type
from pydantic import BaseModel
from batchata.batch_job import BatchJob
from batchata.providers.base import BaseBatchProvider
from batchata.types import BatchResult


class MockBatchJob:
    """Realistic mock of BatchJob that behaves like the real implementation."""
    
    def __init__(self, batch_id: str, results: List[BatchResult], is_complete: bool = True, 
                 total_cost: float = 0.02, start_time: float = 0.0):
        self._batch_id = batch_id
        self._results = results
        self._is_complete = is_complete
        self._total_cost = total_cost
        self._start_time = start_time
        self._cached_results = None
        
    def is_complete(self) -> bool:
        return self._is_complete
        
    def results(self) -> List[BatchResult]:
        if not self._is_complete:
            return []
        return self._results
    
    def stats(self) -> Dict[str, Any]:
        return {
            "batch_id": self._batch_id,
            "is_complete": self._is_complete,
            "total_cost": self._total_cost,
            "request_count": len(self._results)
        }


class MockProvider:
    """Realistic mock of BaseBatchProvider."""
    
    def __init__(self, batch_id: str = "test_batch_123"):
        self.batch_id = batch_id
        self.created_batches = []
        self.batch_results = {}
        
        # Make methods mockable
        self.validate_batch = Mock()
        self.prepare_batch_requests = Mock()
        self.create_batch = Mock()
        self.get_batch_status = Mock()
        self._is_batch_completed = Mock()
        self.get_results = Mock()
        self.parse_results = Mock()
        self.has_citations_enabled = Mock()
        
        # Set default behaviors
        self.validate_batch.return_value = None
        self.get_batch_status.return_value = "completed"
        self._is_batch_completed.return_value = True
        self.get_results.return_value = []
        self.parse_results.return_value = []
        self.has_citations_enabled.return_value = False
        
    def setup_default_behaviors(self):
        """Set up default behaviors for methods."""
        def default_prepare_requests(messages, response_model=None, **kwargs):
            return [
                {"custom_id": f"request_{i}", "params": {}}
                for i in range(len(messages))
            ]
            
        def default_create_batch(requests, **kwargs):
            self.created_batches.append(requests)
            return self.batch_id
            
        def default_get_results(batch_id):
            return self.batch_results.get(batch_id, [])
            
        def default_parse_results(raw_results, response_model=None, enable_citations=False):
            return self.batch_results.get(self.batch_id, [])
            
        self.prepare_batch_requests.side_effect = default_prepare_requests
        self.create_batch.side_effect = default_create_batch
        self.get_results.side_effect = default_get_results
        self.parse_results.side_effect = default_parse_results


def create_mock_batch_job(
    batch_id: str = "test_batch_123",
    results: Optional[List[BatchResult]] = None,
    is_complete: bool = True,
    total_cost: float = 0.02
) -> MockBatchJob:
    """Create a mock BatchJob with realistic behavior."""
    if results is None:
        results = [
            {"result": "Default result 1", "citations": None},
            {"result": "Default result 2", "citations": None}
        ]
    
    return MockBatchJob(
        batch_id=batch_id,
        results=results,
        is_complete=is_complete,
        total_cost=total_cost
    )


def create_mock_provider(batch_id: str = "test_batch_123") -> MockProvider:
    """Create a mock provider with realistic behavior."""
    provider = MockProvider(batch_id=batch_id)
    provider.setup_default_behaviors()
    return provider


def create_structured_results(count: int, model_class: Type[BaseModel] = None) -> List[BatchResult]:
    """Create mock structured results in the correct BatchResult format."""
    results = []
    for i in range(count):
        if model_class:
            # Create instance of the model class
            if hasattr(model_class, 'model_fields'):
                # Get field names from Pydantic model
                field_names = list(model_class.model_fields.keys())
                # Create simple test data with proper types
                test_data = {}
                for field in field_names:
                    field_info = model_class.model_fields[field]
                    if hasattr(field_info, 'annotation'):
                        field_type = field_info.annotation
                    else:
                        field_type = str
                    
                    # Generate appropriate test data based on type
                    if field_type == str or field_type == Any:
                        test_data[field] = f"test_{field}_{i}"
                    elif field_type == float:
                        test_data[field] = float(i + 1) * 1.5
                    elif field_type == int:
                        test_data[field] = i + 1
                    elif field_type == bool:
                        test_data[field] = i % 2 == 0
                    else:
                        test_data[field] = f"test_{field}_{i}"
                        
                result_obj = model_class(**test_data)
            else:
                result_obj = f"Result {i}"
        else:
            result_obj = f"Result {i}"
            
        results.append({
            "result": result_obj,
            "citations": None
        })
    
    return results


def create_cited_results(count: int, model_class: Type[BaseModel] = None) -> List[BatchResult]:
    """Create mock results with citations in the correct format."""
    results = []
    for i in range(count):
        if model_class:
            # Create instance of the model class
            if hasattr(model_class, 'model_fields'):
                field_names = list(model_class.model_fields.keys())
                # Create simple test data with proper types
                test_data = {}
                for field in field_names:
                    field_info = model_class.model_fields[field]
                    if hasattr(field_info, 'annotation'):
                        field_type = field_info.annotation
                    else:
                        field_type = str
                    
                    # Generate appropriate test data based on type
                    if field_type == str or field_type == Any:
                        test_data[field] = f"test_{field}_{i}"
                    elif field_type == float:
                        test_data[field] = float(i + 1) * 1.5
                    elif field_type == int:
                        test_data[field] = i + 1
                    elif field_type == bool:
                        test_data[field] = i % 2 == 0
                    else:
                        test_data[field] = f"test_{field}_{i}"
                        
                result_obj = model_class(**test_data)
                
                # Create citations for each field
                citations = {}
                for field in field_names:
                    citations[field] = [
                        {
                            "start": 10 + i * 5,
                            "end": 20 + i * 5,
                            "text": f"cited_text_{field}_{i}"
                        }
                    ]
            else:
                result_obj = f"Result {i}"
                citations = {
                    "content": [
                        {
                            "start": 10 + i * 5,
                            "end": 20 + i * 5,
                            "text": f"cited_text_{i}"
                        }
                    ]
                }
        else:
            result_obj = f"Result {i}"
            citations = None
            
        results.append({
            "result": result_obj,
            "citations": citations
        })
    
    return results


def setup_mock_batch_function(mock_batch, results: List[BatchResult], batch_id: str = "test_batch_123"):
    """Set up the mock batch function to return a realistic BatchJob."""
    mock_job = create_mock_batch_job(batch_id=batch_id, results=results)
    mock_batch.return_value = mock_job
    return mock_job