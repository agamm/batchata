"""
BatchJob Module

Provides a simple interface for async-like batch processing.
"""

import json
import time
from pathlib import Path
from typing import List, Optional, Type, Union, Dict
from pydantic import BaseModel
from .providers.base import BaseBatchProvider
from .citations import Citation

# Type alias for field-level citations mapping
FieldCitations = Dict[str, List[Citation]]


class BatchJob:
    """Simple batch job that returns immediately and provides methods to check status."""
    
    def __init__(self, provider: BaseBatchProvider, batch_id: str, response_model: Optional[Type[BaseModel]] = None, verbose: bool = False, enable_citations: bool = False, raw_results_dir: Optional[str] = None):
        """
        Initialize BatchJob.
        
        Args:
            provider: Provider instance for API calls
            batch_id: ID of the batch
            response_model: Optional Pydantic model for structured responses
            verbose: Whether to show warnings when not complete
            enable_citations: Whether citations were requested for this batch
            raw_results_dir: Optional directory to save raw API responses as JSON files
        """
        self._provider = provider
        self._batch_id = batch_id
        self._response_model = response_model
        self._verbose = verbose
        self._enable_citations = enable_citations
        self._raw_results_dir = raw_results_dir
        self._cached_results = None
        self._cached_citations = None
        self._start_time = time.time()
    
    def is_complete(self) -> bool:
        """Check if batch is complete."""
        if self._batch_id == "empty_batch":
            return True
        status = self._provider.get_batch_status(self._batch_id)
        return self._provider._is_batch_completed(status)
    
    def results(self) -> Union[List[BaseModel], List[str]]:
        """
        Get batch results.
        
        Returns:
            List[str] if no response_model provided
            List[BaseModel] if response_model provided
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
            
            self._cached_results, self._cached_citations = self._provider.parse_results(
                raw_results, self._response_model, self._enable_citations
            )
        
        return self._cached_results
    
    def citations(self) -> Union[List[Citation], List[FieldCitations], None]:
        """
        Get citations.
        
        Returns:
            None if citations not enabled
            List[Citation] if no response_model (text + citations mode)
            List[FieldCitations] if response_model provided (structured + field citations mode)
            Empty list if not complete
        """
        if not self._enable_citations:
            return None
            
        if not self.is_complete():
            if self._verbose:
                print(f"⚠️  Batch {self._batch_id} is still running. Citations not yet available.")
            return []
        
        # Ensure results have been fetched
        if self._cached_results is None:
            self.results()
        
        return self._cached_citations if self._cached_citations is not None else []
    
    def stats(self, print_stats: bool = False) -> dict:
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
                citations = self.citations()
                stats_data["total_citations"] = len(citations) if citations else 0
        
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
        
        return stats_data
    
    def _save_raw_responses(self, raw_results: List[dict]) -> None:
        """
        Save raw API responses to JSON files.
        
        Args:
            raw_results: List of raw response dictionaries from API
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
                json.dump(raw_response, f, indent=2, ensure_ascii=False)