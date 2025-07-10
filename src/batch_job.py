"""
BatchJob Module

Provides a simple interface for async-like batch processing.
"""

import json
import time
from pathlib import Path
from typing import List, Optional, Type, Dict, Any
from pydantic import BaseModel
from .providers.base import BaseBatchProvider
from .types import BatchResult


class BatchJob:
    """Simple batch job that returns immediately and provides methods to check status."""
    
    def __init__(self, provider: BaseBatchProvider, batch_id: str, response_model: Optional[Type[BaseModel]] = None, verbose: bool = False, enable_citations: bool = False, raw_results_dir: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize BatchJob.
        
        Args:
            provider: Provider instance for API calls
            batch_id: ID of the batch
            response_model: Optional Pydantic model for structured responses
            verbose: Whether to show warnings when not complete
            enable_citations: Whether citations were requested for this batch
            raw_results_dir: Optional directory to save raw API responses as JSON files
            model: Model name for cost calculations
        """
        self._provider = provider
        self._batch_id = batch_id
        self._response_model = response_model
        self._verbose = verbose
        self._enable_citations = enable_citations
        self._raw_results_dir = raw_results_dir
        self._model = model
        self._cached_results = None
        self._start_time = time.time()
    
    def is_complete(self) -> bool:
        """Check if batch is complete."""
        if self._batch_id == "empty_batch":
            return True
        status = self._provider.get_batch_status(self._batch_id)
        return self._provider._is_batch_completed(status)
    
    def results(self) -> List[BatchResult]:
        """
        Get batch results in unified format with citations.
        
        Returns:
            List of BatchResult entries with result and citations together
            Empty list if not complete
        """
        if self._batch_id == "empty_batch":
            return []
            
        if not self.is_complete():
            if self._verbose:
                print(f"⚠️  Batch {self._batch_id} is still running. Results not yet available.")
            return []
        
        if self._cached_results is None:
            raw_results = self._provider.get_results(self._batch_id)
            
            # Save raw responses to files if directory is specified
            if self._raw_results_dir:
                self._save_raw_responses(raw_results)
            
            # Get unified results from provider
            self._cached_results = self._provider.parse_results(
                raw_results, self._response_model, self._enable_citations
            )
        
        return self._cached_results
    
    
    def stats(self, print_stats: bool = False) -> Dict[str, Any]:
        """
        Get batch statistics.
        
        Args:
            print_stats: Whether to print stats to console
            
        Returns:
            Dictionary with batch statistics
        """
        elapsed_time = time.time() - self._start_time
        is_complete = self.is_complete()
        
        if self._batch_id == "empty_batch":
            status = "empty"
        else:
            status = self._provider.get_batch_status(self._batch_id)
        
        stats_data = {
            "batch_id": self._batch_id,
            "is_complete": is_complete,
            "elapsed_time": elapsed_time,
            "status": status,
            "citations_enabled": self._enable_citations,
            "has_response_model": self._response_model is not None
        }
        
        if is_complete:
            results = self.results()
            stats_data["total_results"] = len(results)
            
            if self._enable_citations:
                # Count total citations across all results
                total_citations = 0
                for result_entry in results:
                    if result_entry.get("citations"):
                        if isinstance(result_entry["citations"], list):
                            total_citations += len(result_entry["citations"])
                        elif isinstance(result_entry["citations"], dict):
                            # For field citations, count total across all fields
                            for field_citations in result_entry["citations"].values():
                                if isinstance(field_citations, list):
                                    total_citations += len(field_citations)
                stats_data["total_citations"] = total_citations
            
            # Add cost information if model is provided
            if self._model:
                cost_data = self._provider.get_batch_usage_costs(self._batch_id, self._model)
                stats_data.update(cost_data)
        
        if print_stats:
            print(f"📊 Batch Statistics")
            print(f"   ID: {stats_data['batch_id']}")
            print(f"   Status: {stats_data['status']}")
            print(f"   Complete: {'✅' if stats_data['is_complete'] else '⏳'}")
            print(f"   Elapsed: {stats_data['elapsed_time']:.1f}s")
            print(f"   Mode: ", end="")
            
            if stats_data['has_response_model'] and stats_data['citations_enabled']:
                print("Structured + Field Citations")
            elif stats_data['has_response_model']:
                print("Structured Only")
            elif stats_data['citations_enabled']:
                print("Text + Citations")
            else:
                print("Plain Text")
            
            if is_complete:
                print(f"   Results: {stats_data['total_results']}")
                if stats_data['citations_enabled']:
                    print(f"   Citations: {stats_data.get('total_citations', 0)}")
                
                # Print cost information if available
                if 'total_cost' in stats_data:
                    print(f"   Input tokens: {stats_data.get('total_input_tokens', 0):,}")
                    print(f"   Output tokens: {stats_data.get('total_output_tokens', 0):,}")
                    print(f"   Total cost: ${stats_data['total_cost']:.4f}")
                    if stats_data.get('service_tier') == 'batch':
                        print(f"   (50% batch discount applied)")
        
        return stats_data
    
    def _save_raw_responses(self, raw_results: List[Any]) -> None:
        """
        Save raw API responses to JSON files.
        
        Args:
            raw_results: List of raw response objects from API (Pydantic models or dicts)
        """
        if not self._raw_results_dir:
            return
            
        # Create directory if it doesn't exist
        raw_dir = Path(self._raw_results_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)
        
        # Save each response as a separate JSON file
        for i, raw_response in enumerate(raw_results):
            filename = f"{self._batch_id}_{i}.json"
            filepath = raw_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                # Handle both Pydantic models and plain dicts
                try:
                    # Try to call model_dump() if it exists (Pydantic models)
                    data = raw_response.model_dump()
                    json.dump(data, f, indent=2, ensure_ascii=False)
                except (AttributeError, TypeError):
                    # Fall back to direct serialization for plain dicts
                    json.dump(raw_response, f, indent=2, ensure_ascii=False)