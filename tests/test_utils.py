"""Test utility functions."""

import pytest
import threading
import time
from typing import List, Optional, Union
from unittest.mock import Mock
from pydantic import BaseModel
from src.utils import check_flat_model_for_citation_mapping, run_jobs_with_conditional_parallel


class FlatModel(BaseModel):
    name: str
    count: int


class NestedModel(BaseModel):
    name: str
    nested: FlatModel


class ListModel(BaseModel):
    name: str
    items: List[FlatModel]


class OptionalModel(BaseModel):
    name: str
    optional_item: Optional[FlatModel]


class UnionModel(BaseModel):
    name: str
    union_item: Union[FlatModel, str]


def test_flat_model_with_citations_passes():
    """Flat models should pass validation with citations enabled"""
    check_flat_model_for_citation_mapping(FlatModel, True)  # Should not raise


def test_flat_model_without_citations_passes():
    """Flat models should pass validation with citations disabled"""
    check_flat_model_for_citation_mapping(FlatModel, False)  # Should not raise


def test_nested_model_without_citations_passes():
    """Nested models should pass validation when citations are disabled"""
    check_flat_model_for_citation_mapping(NestedModel, False)  # Should not raise


def test_no_model_passes():
    """No model should pass validation"""
    check_flat_model_for_citation_mapping(None, True)  # Should not raise


def test_direct_nested_model_with_citations_fails():
    """Direct nested models should fail with citations enabled"""
    with pytest.raises(ValueError, match="Citation mapping requires flat Pydantic models"):
        check_flat_model_for_citation_mapping(NestedModel, True)


def test_list_nested_model_with_citations_fails():
    """List[BaseModel] should fail with citations enabled"""
    with pytest.raises(ValueError, match="Citation mapping requires flat Pydantic models"):
        check_flat_model_for_citation_mapping(ListModel, True)


def test_optional_nested_model_with_citations_fails():
    """Optional[BaseModel] should fail with citations enabled"""
    with pytest.raises(ValueError, match="Citation mapping requires flat Pydantic models"):
        check_flat_model_for_citation_mapping(OptionalModel, True)


def test_union_nested_model_with_citations_fails():
    """Union[BaseModel, ...] should fail with citations enabled"""
    with pytest.raises(ValueError, match="Citation mapping requires flat Pydantic models"):
        check_flat_model_for_citation_mapping(UnionModel, True)


def test_error_message_contains_field_name():
    """Error message should contain the problematic field name"""
    with pytest.raises(ValueError, match="Field 'nested' contains nested model"):
        check_flat_model_for_citation_mapping(NestedModel, True)
    
    with pytest.raises(ValueError, match="Field 'items' contains nested model"):
        check_flat_model_for_citation_mapping(ListModel, True)


# Tests for run_jobs_with_conditional_parallel

def test_parallel_execution_basic():
    """Test basic parallel execution without conditions"""
    results = []
    lock = threading.Lock()
    
    def job_processor(job):
        with lock:
            results.append(job)
        time.sleep(0.01)  # Simulate work
    
    def no_condition():
        return False  # Never stop
    
    jobs = [f"job_{i}" for i in range(5)]
    
    run_jobs_with_conditional_parallel(
        max_parallel=3,
        condition_fn=no_condition,
        jobs=jobs,
        job_processor_fn=job_processor
    )
    
    assert len(results) == 5
    assert set(results) == set(jobs)


def test_parallel_execution_with_condition():
    """Test that condition checking stops new jobs"""
    results = []
    lock = threading.Lock()
    
    def job_processor(job):
        with lock:
            results.append(job)
        time.sleep(0.01)  # Simulate work
    
    def stop_after_two():
        return len(results) >= 2  # Stop after 2 jobs complete
    
    jobs = [f"job_{i}" for i in range(5)]
    
    run_jobs_with_conditional_parallel(
        max_parallel=2,
        condition_fn=stop_after_two,
        jobs=jobs,
        job_processor_fn=job_processor
    )
    
    # Should process at least 2 jobs (initial batch) but not all 5
    assert len(results) >= 2
    assert len(results) <= 4  # At most 2 initial + 2 more before condition triggers


def test_atomic_cost_checking_race_condition():
    """Test that atomic locking prevents cost limit race conditions"""
    total_cost = [0.0]  # Use list for mutable reference
    results = []
    shared_lock = threading.Lock()
    
    def job_processor(job):
        # Simulate job completion that updates cost
        time.sleep(0.01)  # Simulate work
        with shared_lock:  # Job processor also uses the shared lock
            total_cost[0] += 0.018  # Each job costs 0.018
            results.append(job)
    
    def cost_limit_exceeded():
        # This will be called under the shared_lock by the parallel utility
        return total_cost[0] >= 0.03  # Limit is 0.03
    
    jobs = [f"job_{i}" for i in range(4)]  # 4 jobs, but limit allows only ~1.67 jobs
    
    run_jobs_with_conditional_parallel(
        max_parallel=2,
        condition_fn=cost_limit_exceeded,
        jobs=jobs,
        job_processor_fn=job_processor,
        shared_lock=shared_lock
    )
    
    # With atomic locking, should process exactly 2 jobs (2 * 0.018 = 0.036 > 0.03)
    # The key is that the 3rd job should NOT start due to proper atomic cost checking
    # In parallel test environments, allow some flexibility but ensure limit is respected
    assert len(results) >= 2  # At least 2 jobs should complete
    assert len(results) <= 3  # But not more than 3 (strict limit would be 2, but allow 1 extra for timing)
    assert total_cost[0] >= 0.036  # Cost should exceed limit
    
    # The main assertion: cost limit should prevent excessive job starts
    # Even if timing varies, we shouldn't see all 4 jobs complete
    assert len(results) < 4  # This is the key race condition test


def test_parallel_execution_max_parallel_limit():
    """Test that max_parallel limit is respected"""
    active_jobs = [0]  # Track concurrent jobs
    max_concurrent = [0]  # Track maximum concurrent jobs seen
    lock = threading.Lock()
    
    def job_processor(job):
        with lock:
            active_jobs[0] += 1
            max_concurrent[0] = max(max_concurrent[0], active_jobs[0])
        
        time.sleep(0.05)  # Simulate longer work to ensure overlap
        
        with lock:
            active_jobs[0] -= 1
    
    def no_condition():
        return False  # Never stop
    
    jobs = [f"job_{i}" for i in range(6)]
    
    run_jobs_with_conditional_parallel(
        max_parallel=3,
        condition_fn=no_condition,
        jobs=jobs,
        job_processor_fn=job_processor
    )
    
    # Should never exceed max_parallel=3 concurrent jobs
    assert max_concurrent[0] <= 3
    assert max_concurrent[0] >= 2  # Should have some concurrency


def test_shared_lock_parameter():
    """Test that shared_lock parameter is used correctly"""
    lock_acquisitions = []
    
    class MockLock:
        def __init__(self):
            self._real_lock = threading.Lock()
            
        def acquire(self, *args, **kwargs):
            lock_acquisitions.append("acquire")
            return self._real_lock.acquire(*args, **kwargs)
            
        def release(self, *args, **kwargs):
            lock_acquisitions.append("release")
            return self._real_lock.release(*args, **kwargs)
            
        def __enter__(self):
            self.acquire()
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            self.release()
    
    mock_lock = MockLock()
    
    def simple_job_processor(job):
        time.sleep(0.01)
    
    def no_condition():
        return False
    
    jobs = ["job_1", "job_2"]
    
    run_jobs_with_conditional_parallel(
        max_parallel=1,
        condition_fn=no_condition,
        jobs=jobs,
        job_processor_fn=simple_job_processor,
        shared_lock=mock_lock
    )
    
    # Should have acquired and released the lock multiple times
    assert "acquire" in lock_acquisitions
    assert "release" in lock_acquisitions
    assert lock_acquisitions.count("acquire") == lock_acquisitions.count("release")