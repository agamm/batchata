"""Custom exceptions for batchata."""


# Base exceptions
class BatchataError(Exception):
    """Base exception for all batchata errors"""
    pass


# Batch Manager exceptions
class BatchManagerError(BatchataError):
    """Base exception for BatchManager errors"""
    pass


class StateFileError(BatchManagerError):
    """Error related to state file operations"""
    pass


class InvalidStateError(StateFileError):
    """State file is invalid or corrupted"""
    pass


class CostLimitExceededError(BatchManagerError):
    """Processing stopped due to cost limit"""
    pass


class JobProcessingError(BatchManagerError):
    """Error during job processing"""
    pass


class BatchInterruptedError(BatchManagerError):
    """Batch processing was interrupted and cannot be resumed"""
    pass


# File and content validation exceptions
class FileTooLargeError(BatchataError):
    """File exceeds model context window limits"""
    pass


class UnsupportedContentError(BatchataError):
    """Content type is not supported for the requested operation"""
    pass


class UnsupportedFileFormatError(BatchataError):
    """File format is not supported"""
    pass


# Resource constraint exceptions
class InsufficientMemoryError(BatchataError):
    """Insufficient memory to process the request"""
    pass


class RateLimitExceededError(BatchataError):
    """API rate limit exceeded"""
    pass


class APIQuotaExceededError(BatchataError):
    """API quota limit exceeded"""
    pass