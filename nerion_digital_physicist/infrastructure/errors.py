"""Error handling framework for the Digital Physicist system."""

from typing import Optional, Dict, Any


class DigitalPhysicistError(Exception):
    """Base exception for Digital Physicist operations."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}


class LLMGenerationError(DigitalPhysicistError):
    """LLM generation failed."""
    
    def __init__(self, message: str, provider: Optional[str] = None, model: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="LLM_GENERATION_ERROR", **kwargs)
        self.provider = provider
        self.model = model


class ValidationError(DigitalPhysicistError):
    """Data validation failed."""
    
    def __init__(self, message: str, field: Optional[str] = None, value: Optional[Any] = None, **kwargs):
        super().__init__(message, error_code="VALIDATION_ERROR", **kwargs)
        self.field = field
        self.value = value


class DatabaseError(DigitalPhysicistError):
    """Database operation failed."""
    
    def __init__(self, message: str, operation: Optional[str] = None, table: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="DATABASE_ERROR", **kwargs)
        self.operation = operation
        self.table = table


class ConfigurationError(DigitalPhysicistError):
    """Configuration error."""
    
    def __init__(self, message: str, config_key: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="CONFIGURATION_ERROR", **kwargs)
        self.config_key = config_key


class NetworkError(DigitalPhysicistError):
    """Network operation failed."""
    
    def __init__(self, message: str, url: Optional[str] = None, status_code: Optional[int] = None, **kwargs):
        super().__init__(message, error_code="NETWORK_ERROR", **kwargs)
        self.url = url
        self.status_code = status_code


class TimeoutError(DigitalPhysicistError):
    """Operation timed out."""
    
    def __init__(self, message: str, timeout_seconds: Optional[float] = None, **kwargs):
        super().__init__(message, error_code="TIMEOUT_ERROR", **kwargs)
        self.timeout_seconds = timeout_seconds


class AccessControlError(DigitalPhysicistError):
    """Access control related errors."""
    pass


class ResourceLimitExceededError(DigitalPhysicistError):
    """Raised when resource limits are exceeded."""
    pass


class CircuitBreakerOpenError(DigitalPhysicistError):
    """Raised when circuit breaker is open."""
    pass
