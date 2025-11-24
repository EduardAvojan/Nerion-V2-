import logging
import json
import sys
from datetime import datetime

# Configure logging to stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    stream=sys.stdout
)
logger = logging.getLogger("nerion_api")

def _log_json(event_type: str, data: dict):
    """Log structured JSON event."""
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "event_type": event_type,
        **data
    }
    logger.info(json.dumps(entry))

def log_request(client_ip, method, endpoint, status_code, response_time_ms, user_agent, auth_status):
    """Log HTTP request details."""
    _log_json("http_request", {
        "client_ip": client_ip,
        "method": method,
        "endpoint": endpoint,
        "status_code": status_code,
        "response_time_ms": response_time_ms,
        "user_agent": user_agent,
        "auth_status": auth_status
    })

def log_auth_failure(client_ip, endpoint, reason, user_agent):
    """Log authentication failure."""
    _log_json("auth_failure", {
        "client_ip": client_ip,
        "endpoint": endpoint,
        "reason": reason,
        "user_agent": user_agent
    })

def log_rate_limit(client_ip, endpoint, limit, period):
    """Log rate limit exceeded."""
    _log_json("rate_limit_exceeded", {
        "client_ip": client_ip,
        "endpoint": endpoint,
        "limit": limit,
        "period": period
    })

def log_injection_attempt(input_str, client_ip, endpoint):
    """Log detected injection attempt."""
    _log_json("injection_attempt", {
        "client_ip": client_ip,
        "endpoint": endpoint,
        "input_snippet": input_str[:100]  # Log only start of input for safety
    })

def log_terminal_event(event_type, **kwargs):
    """Log terminal-specific event."""
    _log_json("terminal_event", {
        "sub_type": event_type,
        **kwargs
    })
