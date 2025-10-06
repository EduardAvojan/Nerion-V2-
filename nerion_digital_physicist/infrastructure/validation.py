"""Input validation system for the Digital Physicist system."""

import re
from typing import Optional, Any, Dict
from functools import wraps
from pydantic import BaseModel, Field, validator

from .errors import ValidationError
from .logging import log_with_context, logger


class LessonValidation(BaseModel):
    """Validation model for lesson data."""
    
    name: str = Field(..., min_length=1, max_length=100)
    description: str = Field(..., min_length=10, max_length=1000)
    before_code: str = Field(..., min_length=10)
    after_code: str = Field(..., min_length=10)
    test_code: str = Field(..., min_length=10)
    focus_area: Optional[str] = None
    timestamp: str

    @validator('name')
    def validate_name(cls, v):
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError('Name must contain only alphanumeric characters, underscores, and hyphens')
        return v

    @validator('before_code', 'after_code', 'test_code')
    def validate_code(cls, v):
        # Check for potentially dangerous patterns
        dangerous_patterns = [
            r'import\s+os',
            r'import\s+subprocess',
            r'import\s+sys',
            r'eval\s*\(',
            r'exec\s*\(',
            r'__import__\s*\(',
            r'open\s*\(',
            r'file\s*\(',
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError(f'Code contains potentially dangerous pattern: {pattern}')
        
        return v

    @validator('description')
    def validate_description(cls, v):
        # Check for potentially sensitive information
        sensitive_patterns = [
            r'password\s*[:=]',
            r'secret\s*[:=]',
            r'key\s*[:=]',
            r'token\s*[:=]',
            r'api[_-]?key\s*[:=]',
        ]
        
        for pattern in sensitive_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError(f'Description contains potentially sensitive information: {pattern}')
        
        return v


class BugFixValidation(BaseModel):
    """Validation model for bug fix data."""
    
    name: str = Field(..., min_length=1, max_length=100)
    description: str = Field(..., min_length=10, max_length=1000)
    before_code: str = Field(..., min_length=10)
    after_code: str = Field(..., min_length=10)
    test_code: str = Field(..., min_length=10)
    focus_area: Optional[str] = None
    timestamp: str

    @validator('name')
    def validate_name(cls, v):
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError('Name must contain only alphanumeric characters, underscores, and hyphens')
        return v

    @validator('before_code', 'after_code', 'test_code')
    def validate_code(cls, v):
        # Check for potentially dangerous patterns
        dangerous_patterns = [
            r'import\s+os',
            r'import\s+subprocess',
            r'import\s+sys',
            r'eval\s*\(',
            r'exec\s*\(',
            r'__import__\s*\(',
            r'open\s*\(',
            r'file\s*\(',
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError(f'Code contains potentially dangerous pattern: {pattern}')
        
        return v


def validate_lesson_data(func):
    """Decorator that validates lesson data before processing."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            # Extract lesson data from arguments
            lesson_data = None
            if 'lesson_data' in kwargs:
                lesson_data = kwargs['lesson_data']
            elif args and isinstance(args[0], dict):
                lesson_data = args[0]
            
            if lesson_data:
                # Validate using Pydantic model
                validated_data = LessonValidation(**lesson_data)
                
                # Replace with validated data
                if 'lesson_data' in kwargs:
                    kwargs['lesson_data'] = validated_data.dict()
                else:
                    args = (validated_data.dict(),) + args[1:]
                
                log_with_context(
                    logger,
                    logger.info,
                    f"Lesson data validated successfully",
                    lesson_name=validated_data.name,
                    validation_status="success"
                )
            
            return func(*args, **kwargs)
            
        except Exception as e:
            log_with_context(
                logger,
                logger.error,
                f"Lesson data validation failed",
                error=str(e),
                error_type=type(e).__name__,
                validation_status="failed"
            )
            raise ValidationError(f"Lesson data validation failed: {e}")
    
    return wrapper


def validate_bug_fix_data(func):
    """Decorator that validates bug fix data before processing."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            # Extract bug fix data from arguments
            bug_fix_data = None
            if 'bug_fix_data' in kwargs:
                bug_fix_data = kwargs['bug_fix_data']
            elif args and isinstance(args[0], dict):
                bug_fix_data = args[0]
            
            if bug_fix_data:
                # Validate using Pydantic model
                validated_data = BugFixValidation(**bug_fix_data)
                
                # Replace with validated data
                if 'bug_fix_data' in kwargs:
                    kwargs['bug_fix_data'] = validated_data.dict()
                else:
                    args = (validated_data.dict(),) + args[1:]
                
                log_with_context(
                    logger,
                    logger.info,
                    f"Bug fix data validated successfully",
                    bug_fix_name=validated_data.name,
                    validation_status="success"
                )
            
            return func(*args, **kwargs)
            
        except Exception as e:
            log_with_context(
                logger,
                logger.error,
                f"Bug fix data validation failed",
                error=str(e),
                error_type=type(e).__name__,
                validation_status="failed"
            )
            raise ValidationError(f"Bug fix data validation failed: {e}")
    
    return wrapper


def sanitize_code(code: str) -> str:
    """Sanitize code by removing or replacing potentially dangerous patterns."""
    # Replace dangerous imports with safe alternatives
    replacements = {
        r'import\s+os': '# import os  # REMOVED FOR SECURITY',
        r'import\s+subprocess': '# import subprocess  # REMOVED FOR SECURITY',
        r'import\s+sys': '# import sys  # REMOVED FOR SECURITY',
        r'eval\s*\(': '# eval(  # REMOVED FOR SECURITY',
        r'exec\s*\(': '# exec(  # REMOVED FOR SECURITY',
        r'__import__\s*\(': '# __import__(  # REMOVED FOR SECURITY',
        r'open\s*\(': '# open(  # REMOVED FOR SECURITY',
        r'file\s*\(': '# file(  # REMOVED FOR SECURITY',
    }
    
    sanitized_code = code
    for pattern, replacement in replacements.items():
        sanitized_code = re.sub(pattern, replacement, sanitized_code, flags=re.IGNORECASE)
    
    return sanitized_code


def validate_llm_response(response: str) -> str:
    """Validate and sanitize LLM response."""
    if not response or not isinstance(response, str):
        raise ValidationError("LLM response must be a non-empty string")
    
    if len(response) > 10000:  # 10KB limit
        raise ValidationError("LLM response too long (max 10KB)")
    
    # Check for potential injection patterns
    injection_patterns = [
        r'<script',
        r'javascript:',
        r'data:text/html',
        r'vbscript:',
    ]
    
    for pattern in injection_patterns:
        if re.search(pattern, response, re.IGNORECASE):
            raise ValidationError(f"LLM response contains potentially malicious content: {pattern}")
    
    return response


# Data Classification and Privacy
from enum import Enum
from typing import Dict, List, Any

class DataClassification(Enum):
    """Data classification levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"

def classify_content(content: str) -> Dict[str, Any]:
    """Simple content classification for privacy compliance."""
    if not content:
        return {"classification": DataClassification.PUBLIC.value, "sensitive": False}
    
    # Check for sensitive patterns
    sensitive_patterns = [
        r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
        r'(?i)(password|secret|key|token)\s*[:=]',  # Credentials
    ]
    
    for pattern in sensitive_patterns:
        if re.search(pattern, content, re.IGNORECASE):
            return {"classification": DataClassification.RESTRICTED.value, "sensitive": True}
    
    return {"classification": DataClassification.INTERNAL.value, "sensitive": False}
