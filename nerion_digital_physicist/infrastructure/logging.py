"""Structured logging infrastructure for the Digital Physicist system."""

import logging
import json
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
from contextvars import ContextVar
from functools import wraps

# Context variable for correlation ID
correlation_id: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)


class StructuredFormatter(logging.Formatter):
    """Formatter that outputs structured JSON logs."""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'correlation_id': correlation_id.get(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry)


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Setup structured logging for the application."""
    logger = logging.getLogger('nerion_digital_physicist')
    logger.setLevel(level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(StructuredFormatter())
    logger.addHandler(console_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


def log_with_context(logger: logging.Logger, level: int, message: str, **kwargs):
    """Log with additional context fields."""
    extra_fields = kwargs
    logger.log(level, message, extra={'extra_fields': extra_fields})


def with_correlation_id(func):
    """Decorator that ensures a correlation ID is set for the function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Generate or use existing correlation ID
        current_id = correlation_id.get()
        if not current_id:
            current_id = str(uuid.uuid4())
            correlation_id.set(current_id)
        
        return func(*args, **kwargs)
    return wrapper


def track_performance(func):
    """Decorator that tracks function performance."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        import time
        logger = logging.getLogger('nerion_digital_physicist')
        
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            
            # Log performance metrics
            log_with_context(
                logger,
                logging.INFO,
                f"Function {func.__name__} completed successfully",
                duration=duration,
                function=func.__name__,
                status="success"
            )
            
            return result
        except Exception as e:
            duration = time.time() - start_time
            
            # Log error metrics
            log_with_context(
                logger,
                logging.ERROR,
                f"Function {func.__name__} failed",
                duration=duration,
                function=func.__name__,
                status="error",
                error=str(e),
                error_type=type(e).__name__
            )
            
            raise
    
    return wrapper


# Initialize the logger
logger = setup_logging()
