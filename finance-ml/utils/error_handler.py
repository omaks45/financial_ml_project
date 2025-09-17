"""
Day 5: Enhanced Error Handling and Retry Logic
utils/error_handler.py

Provides robust error handling and retry mechanisms for the pipeline
"""

import time
import logging
import functools
from typing import Dict, Any, Optional, Callable, Type
from enum import Enum
import traceback

logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    API_ERROR = "api_error"
    DATABASE_ERROR = "database_error"
    PROCESSING_ERROR = "processing_error"
    VALIDATION_ERROR = "validation_error"
    NETWORK_ERROR = "network_error"
    TIMEOUT_ERROR = "timeout_error"

class PipelineError(Exception):
    """Custom exception for pipeline errors"""
    
    def __init__(self, message: str, category: ErrorCategory, severity: ErrorSeverity, 
                 company_id: str = None, stage: str = None, original_error: Exception = None):
        super().__init__(message)
        self.category = category
        self.severity = severity
        self.company_id = company_id
        self.stage = stage
        self.original_error = original_error
        self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'message': str(self),
            'category': self.category.value,
            'severity': self.severity.value,
            'company_id': self.company_id,
            'stage': self.stage,
            'timestamp': self.timestamp,
            'original_error': str(self.original_error) if self.original_error else None
        }

class RetryConfig:
    """Configuration for retry logic"""
    
    def __init__(self, max_attempts: int = 3, base_delay: float = 1.0, 
                 exponential_backoff: bool = True, jitter: bool = True):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.exponential_backoff = exponential_backoff
        self.jitter = jitter

class ErrorHandler:
    """Enhanced error handler with retry logic and categorization"""
    
    def __init__(self):
        self.error_counts = {}
        self.error_history = []
        self.retry_configs = {
            ErrorCategory.API_ERROR: RetryConfig(max_attempts=3, base_delay=2.0),
            ErrorCategory.DATABASE_ERROR: RetryConfig(max_attempts=2, base_delay=1.0),
            ErrorCategory.NETWORK_ERROR: RetryConfig(max_attempts=3, base_delay=3.0),
            ErrorCategory.PROCESSING_ERROR: RetryConfig(max_attempts=2, base_delay=0.5),
            ErrorCategory.VALIDATION_ERROR: RetryConfig(max_attempts=1, base_delay=0.0),
            ErrorCategory.TIMEOUT_ERROR: RetryConfig(max_attempts=2, base_delay=5.0)
        }
    
    def categorize_error(self, error: Exception, stage: str = None) -> ErrorCategory:
        """Categorize error based on type and context"""
        error_type = type(error).__name__.lower()
        error_message = str(error).lower()
        
        # Network and connection errors
        if any(keyword in error_message for keyword in ['connection', 'network', 'timeout', 'unreachable']):
            return ErrorCategory.NETWORK_ERROR
        
        # Database errors
        if any(keyword in error_message for keyword in ['database', 'mysql', 'sql', 'connection']):
            return ErrorCategory.DATABASE_ERROR
        
        # API errors
        if any(keyword in error_message for keyword in ['api', 'http', 'request', '404', '500', '403']):
            return ErrorCategory.API_ERROR
        
        # Validation errors
        if any(keyword in error_message for keyword in ['validation', 'invalid', 'missing', 'format']):
            return ErrorCategory.VALIDATION_ERROR
        
        # Timeout errors
        if 'timeout' in error_message:
            return ErrorCategory.TIMEOUT_ERROR
        
        # Default to processing error
        return ErrorCategory.PROCESSING_ERROR
    
    def determine_severity(self, error: Exception, category: ErrorCategory) -> ErrorSeverity:
        """Determine error severity"""
        error_message = str(error).lower()
        
        # Critical errors
        if any(keyword in error_message for keyword in ['critical', 'fatal', 'system', 'memory']):
            return ErrorSeverity.CRITICAL
        
        # High severity based on category
        if category in [ErrorCategory.DATABASE_ERROR, ErrorCategory.NETWORK_ERROR]:
            return ErrorSeverity.HIGH
        
        # Medium severity for API and processing errors
        if category in [ErrorCategory.API_ERROR, ErrorCategory.PROCESSING_ERROR]:
            return ErrorSeverity.MEDIUM
        
        # Low severity for validation errors
        return ErrorSeverity.LOW
    
    def should_retry(self, error: PipelineError, attempt_count: int) -> bool:
        """Determine if an error should be retried"""
        config = self.retry_configs.get(error.category, RetryConfig())
        
        # Don't retry if we've exceeded max attempts
        if attempt_count >= config.max_attempts:
            return False
        
        # Don't retry critical errors
        if error.severity == ErrorSeverity.CRITICAL:
            return False
        
        # Don't retry validation errors
        if error.category == ErrorCategory.VALIDATION_ERROR:
            return False
        
        return True
    
    def calculate_delay(self, attempt_count: int, category: ErrorCategory) -> float:
        """Calculate delay before retry"""
        config = self.retry_configs.get(category, RetryConfig())
        
        if config.exponential_backoff:
            delay = config.base_delay * (2 ** (attempt_count - 1))
        else:
            delay = config.base_delay
        
        # Add jitter to prevent thundering herd
        if config.jitter:
            import random
            delay *= (0.5 + random.random() * 0.5)
        
        return min(delay, 30.0)  # Cap at 30 seconds
    
    def handle_error(self, error: Exception, company_id: str = None, 
                    stage: str = None) -> PipelineError:
        """Handle and categorize an error"""
        category = self.categorize_error(error, stage)
        severity = self.determine_severity(error, category)
        
        pipeline_error = PipelineError(
            message=str(error),
            category=category,
            severity=severity,
            company_id=company_id,
            stage=stage,
            original_error=error
        )
        
        # Log the error
        self._log_error(pipeline_error)
        
        # Update statistics
        self._update_error_stats(pipeline_error)
        
        return pipeline_error
    
    def _log_error(self, error: PipelineError):
        """Log error with appropriate level"""
        log_message = f"[{error.category.value.upper()}] {error.message}"
        if error.company_id:
            log_message += f" (Company: {error.company_id})"
        if error.stage:
            log_message += f" (Stage: {error.stage})"
        
        if error.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message)
        elif error.severity == ErrorSeverity.HIGH:
            logger.error(log_message)
        elif error.severity == ErrorSeverity.MEDIUM:
            logger.warning(log_message)
        else:
            logger.info(log_message)
    
    def _update_error_stats(self, error: PipelineError):
        """Update error statistics"""
        key = f"{error.category.value}_{error.severity.value}"
        self.error_counts[key] = self.error_counts.get(key, 0) + 1
        self.error_history.append(error.to_dict())
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get error summary statistics"""
        return {
            'total_errors': len(self.error_history),
            'error_counts': self.error_counts,
            'recent_errors': self.error_history[-10:] if self.error_history else []
        }

def with_retry(error_handler: ErrorHandler, stage: str = None):
    """Decorator to add retry logic to functions"""
    
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            company_id = kwargs.get('company_id') or (args[0] if args else None)
            attempt_count = 0
            last_error = None
            
            while True:
                attempt_count += 1
                
                try:
                    return func(*args, **kwargs)
                
                except Exception as e:
                    pipeline_error = error_handler.handle_error(e, company_id, stage)
                    last_error = pipeline_error
                    
                    if not error_handler.should_retry(pipeline_error, attempt_count):
                        logger.error(f"Max retries exceeded for {func.__name__}: {pipeline_error}")
                        raise pipeline_error
                    
                    delay = error_handler.calculate_delay(attempt_count, pipeline_error.category)
                    logger.warning(f"Retrying {func.__name__} (attempt {attempt_count}) after {delay:.1f}s delay")
                    time.sleep(delay)
            
            # This should never be reached, but just in case
            raise last_error
        
        return wrapper
    return decorator

def safe_execute(func: Callable, error_handler: ErrorHandler, 
                company_id: str = None, stage: str = None, 
                default_return: Any = None) -> tuple[Any, Optional[PipelineError]]:
    """Safely execute a function and return result with error info"""
    try:
        result = func()
        return result, None
    except Exception as e:
        pipeline_error = error_handler.handle_error(e, company_id, stage)
        return default_return, pipeline_error

# Circuit breaker pattern for preventing cascading failures
class CircuitBreaker:
    """Circuit breaker to prevent cascading failures"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function through circuit breaker"""
        
        if self.state == 'OPEN':
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = 'HALF_OPEN'
                logger.info("Circuit breaker transitioning to HALF_OPEN")
            else:
                raise PipelineError(
                    "Circuit breaker is OPEN - service unavailable",
                    ErrorCategory.PROCESSING_ERROR,
                    ErrorSeverity.HIGH
                )
        
        try:
            result = func(*args, **kwargs)
            
            # Success - reset failure count
            if self.state == 'HALF_OPEN':
                self.state = 'CLOSED'
                logger.info("Circuit breaker reset to CLOSED")
            
            self.failure_count = 0
            return result
            
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = 'OPEN'
                logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
            
            raise e

# Example usage functions
def create_error_handler() -> ErrorHandler:
    """Factory function to create error handler"""
    return ErrorHandler()

def create_circuit_breaker(failure_threshold: int = 5) -> CircuitBreaker:
    """Factory function to create circuit breaker"""
    return CircuitBreaker(failure_threshold=failure_threshold)