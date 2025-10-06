"""Retry logic with exponential backoff for the Digital Physicist system."""

import time
import random
from functools import wraps
from typing import Callable, Type, Tuple, Any, Optional
import logging

from .errors import DigitalPhysicistError, NetworkError, TimeoutError, LLMGenerationError, CircuitBreakerOpenError

logger = logging.getLogger(__name__)


def retry_with_backoff(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: Tuple[Type[Exception], ...] = (
        NetworkError,
        TimeoutError,
        LLMGenerationError,
    )
):
    """
    Decorator that retries a function with exponential backoff.
    
    Args:
        max_attempts: Maximum number of attempts
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential backoff
        jitter: Whether to add random jitter to delay
        retryable_exceptions: Tuple of exception types to retry
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    
                    if attempt == max_attempts - 1:
                        logger.error(
                            f"Function {func.__name__} failed after {max_attempts} attempts",
                            extra={
                                "function": func.__name__,
                                "attempts": max_attempts,
                                "error": str(e),
                                "error_type": type(e).__name__
                            }
                        )
                        raise
                    
                    # Calculate delay with exponential backoff
                    delay = min(
                        base_delay * (exponential_base ** attempt),
                        max_delay
                    )
                    
                    if jitter:
                        delay *= (0.5 + random.random() * 0.5)
                    
                    logger.warning(
                        f"Function {func.__name__} failed, retrying in {delay:.2f}s (attempt {attempt + 1}/{max_attempts})",
                        extra={
                            "function": func.__name__,
                            "attempt": attempt + 1,
                            "max_attempts": max_attempts,
                            "delay": delay,
                            "error": str(e),
                            "error_type": type(e).__name__
                        }
                    )
                    
                    time.sleep(delay)
                except Exception as e:
                    # Non-retryable exception, re-raise immediately
                    logger.error(
                        f"Function {func.__name__} failed with non-retryable error",
                        extra={
                            "function": func.__name__,
                            "error": str(e),
                            "error_type": type(e).__name__
                        }
                    )
                    raise
            
            # This should never be reached, but just in case
            if last_exception:
                raise last_exception
                
        return wrapper
    return decorator


def retry_llm_call(max_attempts: int = 3):
    """Specialized retry decorator for LLM calls."""
    return retry_with_backoff(
        max_attempts=max_attempts,
        base_delay=2.0,
        max_delay=30.0,
        exponential_base=2.0,
        jitter=True,
        retryable_exceptions=(LLMGenerationError, NetworkError, TimeoutError)
    )


def retry_database_call(max_attempts: int = 3):
    """Specialized retry decorator for database calls."""
    return retry_with_backoff(
        max_attempts=max_attempts,
        base_delay=0.5,
        max_delay=10.0,
        exponential_base=1.5,
        jitter=False,
        retryable_exceptions=(DigitalPhysicistError,)
    )


# Simple Circuit Breaker
from enum import Enum
from datetime import datetime, timezone
from typing import Callable, Any

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class SimpleCircuitBreaker:
    """Simple circuit breaker implementation."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
            else:
                raise CircuitBreakerOpenError("Circuit breaker is open")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        if not self.last_failure_time:
            return True
        time_since_failure = (datetime.now(timezone.utc) - self.last_failure_time).total_seconds()
        return time_since_failure >= self.recovery_timeout
    
    def _on_success(self):
        """Handle successful operation."""
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
    
    def _on_failure(self):
        """Handle failed operation."""
        self.failure_count += 1
        self.last_failure_time = datetime.now(timezone.utc)
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN

def with_circuit_breaker(circuit_breaker: SimpleCircuitBreaker):
    """Decorator to apply circuit breaker to function."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            return circuit_breaker.call(func, *args, **kwargs)
        return wrapper
    return decorator
