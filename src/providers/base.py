"""
Base Provider Class

Abstract base class for all batch processing providers.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Type, Any, Optional, Union
from pydantic import BaseModel


class BaseBatchProvider(ABC):
    """Abstract base class for batch processing providers."""
    
    def __init__(self, rate_limits: Dict[str, int] = None):
        """
        Initialize provider with optional rate limits.
        
        Args:
            rate_limits: Optional rate limits dict (varies by provider tier)
        """
        self.rate_limits = rate_limits or self.get_default_rate_limits()
    
    @abstractmethod
    def get_default_rate_limits(self) -> Dict[str, int]:
        """Get default rate limits for this provider."""
        pass
    
    @abstractmethod
    def validate_batch(self, messages: List[List[dict]], response_model: Optional[Type[BaseModel]]) -> None:
        """
        Validate batch against provider-specific limits.
        
        Args:
            messages: List of message conversations
            response_model: Optional Pydantic model for responses
            
        Raises:
            ValueError: If batch exceeds provider limits
        """
        pass
    
    @abstractmethod
    def prepare_batch_requests(self, messages: List[List[dict]], response_model: Optional[Type[BaseModel]], **kwargs) -> List[dict]:
        """
        Prepare batch requests ready for API submission.
        
        Args:
            messages: List of message conversations
            response_model: Optional Pydantic model for responses
            **kwargs: Additional parameters (model, max_tokens, etc.)
            
        Returns:
            List of formatted requests ready for API submission
        """
        pass
    
    @abstractmethod
    def create_batch(self, requests: List[dict]) -> str:
        """
        Create batch and submit to provider API.
        
        Args:
            requests: Formatted batch requests
            
        Returns:
            Batch ID for tracking
        """
        pass
    
    @abstractmethod
    def get_batch_status(self, batch_id: str) -> str:
        """
        Get current batch status.
        
        Args:
            batch_id: ID of the batch to check
            
        Returns:
            Status string (provider-specific: "in_progress", "completed", "failed", etc.)
        """
        pass
    
    def wait_for_completion(self, batch_id: str, poll_interval: int, verbose: bool) -> None:
        """
        Wait for batch completion with status monitoring.
        
        Args:
            batch_id: ID of the batch to monitor
            poll_interval: Seconds between status checks
            verbose: Whether to print status updates
            
        Raises:
            RuntimeError: If batch fails or times out
        """
        import time
        
        while True:
            status = self.get_batch_status(batch_id)
            
            if verbose:
                print(f"Waiting for batch {batch_id} to complete...")
                print(f"Batch status: {status}")
            
            if self._is_batch_completed(status):
                break
            elif self._is_batch_failed(status):
                raise RuntimeError(f"Batch processing failed: {status}")
            
            time.sleep(poll_interval)
    
    @abstractmethod
    def _is_batch_completed(self, status: str) -> bool:
        """Check if batch status indicates completion."""
        pass
    
    @abstractmethod
    def _is_batch_failed(self, status: str) -> bool:
        """Check if batch status indicates failure."""
        pass
    
    @abstractmethod
    def get_results(self, batch_id: str) -> List[Any]:
        """
        Get raw results from provider API.
        
        Args:
            batch_id: ID of completed batch
            
        Returns:
            Raw results from API
        """
        pass
    
    @abstractmethod
    def parse_results(self, results: List[Any], response_model: Optional[Type[BaseModel]], enable_citations: bool) -> tuple:
        """
        Parse API results into Pydantic models or raw text.
        
        Args:
            results: Raw results from API
            response_model: Optional Pydantic model to parse into
            enable_citations: Whether citations were requested
            
        Returns:
            Tuple of (results, citations) where:
            - results: List[BaseModel] or List[str]
            - citations: List[Citation] or List[FieldCitations] or None
            
        Raises:
            RuntimeError: If parsing fails
        """
        pass