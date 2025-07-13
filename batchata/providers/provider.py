"""Base Provider class."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Type

from pydantic import BaseModel, field_validator

from ..core.job import Job
from ..core.job_result import JobResult
from ..exceptions import ValidationError
from .batch_request import BatchRequest
from .model_config import ModelConfig


class Provider(ABC):
    """Abstract base class for AI providers.
    
    Each provider implementation must define available models and implement
    the abstract methods for job validation, batch creation, and result retrieval.
    """
    
    def __init__(self):
        """Initialize provider with model configurations."""
        self.models: Dict[str, ModelConfig] = {}
    
    def get_param_schema(self, model: str) -> Type[BaseModel]:
        """Get parameter validation schema for model."""
        model_config = self.models.get(model)
        if not model_config:
            raise ValidationError(f"Unknown model: {model}")
        
        class JobParams(BaseModel):
            temperature: Optional[float] = 0.7
            max_tokens: Optional[int] = 1000
            enable_citations: Optional[bool] = False
            
            @field_validator('temperature')
            def validate_temperature(cls, v):
                if v is not None and not 0.0 <= v <= 1.0:
                    raise ValueError("Temperature must be between 0.0 and 1.0")
                return v
            
            @field_validator('max_tokens')
            def validate_max_tokens(cls, v):
                if v is not None and v <= 0:
                    raise ValueError("max_tokens must be positive")
                return v
            
            @field_validator('enable_citations')
            def validate_citations(cls, v):
                if v and not model_config.supports_citations:
                    raise ValueError(f"Model {model} does not support citations")
                return v
        
        return JobParams
    
    def validate_params(self, model: str, **params) -> None:
        """Validate job parameters using pydantic."""
        schema = self.get_param_schema(model)
        schema(**params)  # Will raise ValidationError if invalid
    
    
    @abstractmethod
    def validate_job(self, job: Job) -> None:
        """Validate job constraints and message format.
        
        Args:
            job: Job to validate
            
        Raises:
            ValidationError: If job violates provider/model constraints
        """
        pass
    
    @abstractmethod
    def create_batch(self, jobs: List[Job]) -> BatchRequest:
        """Create and submit a batch of jobs.
        
        Args:
            jobs: List of jobs to include in the batch
            
        Returns:
            BatchRequest with provider's batch ID
            
        Raises:
            BatchSubmissionError: If batch submission fails
        """
        pass
    
    @abstractmethod
    def get_batch_status(self, batch_id: str) -> str:
        """Get current status of a batch.
        
        Args:
            batch_id: Provider's batch identifier
            
        Returns:
            Status string: "pending", "running", "complete", "failed"
        """
        pass
    
    @abstractmethod
    def get_batch_results(self, batch_id: str) -> List[JobResult]:
        """Retrieve results for a completed batch.
        
        Args:
            batch_id: Provider's batch identifier
            
        Returns:
            List of JobResult objects
            
        Raises:
            ProviderError: If results cannot be retrieved
        """
        pass
    
    @abstractmethod
    def estimate_cost(self, jobs: List[Job]) -> float:
        """Estimate cost for a list of jobs.
        
        Args:
            jobs: List of jobs to estimate
            
        Returns:
            Estimated cost in USD
        """
        pass
    
    def supports_model(self, model: str) -> bool:
        """Check if this provider supports the given model.
        
        Args:
            model: Model name to check
            
        Returns:
            True if model is supported
        """
        return model in self.models
    
    def get_model_config(self, model: str) -> Optional[ModelConfig]:
        """Get configuration for a model.
        
        Args:
            model: Model name
            
        Returns:
            ModelConfig if model is supported, None otherwise
        """
        return self.models.get(model)