"""Custom exceptions for the batch processing library."""


class BatchataError(Exception):
    """Base exception for all Batchata errors."""
    pass


class ValidationError(BatchataError):
    """Raised when job or configuration validation fails."""
    pass


class ProviderError(BatchataError):
    """Base exception for provider-related errors."""
    pass


class ProviderNotFoundError(ProviderError):
    """Raised when no provider is found for a model."""
    pass


class ModelNotSupportedError(ProviderError):
    """Raised when a model is not supported by its provider."""
    pass


class BatchSubmissionError(ProviderError):
    """Raised when batch submission to provider fails."""
    pass


class CostLimitExceededError(BatchataError):
    """Raised when cost limit would be exceeded."""
    pass


class StateError(BatchataError):
    """Raised when state management operations fail."""
    pass


class ParseError(BatchataError):
    """Raised when response parsing fails."""
    pass