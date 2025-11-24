import re

def detect_injection(input_str: str, client_ip: str, endpoint: str) -> bool:
    """
    Detect potential injection attacks in input string.
    Returns True if suspicious patterns are found.
    """
    if not input_str:
        return False
        
    # Common injection patterns
    patterns = [
        r";\s*DROP\s+TABLE",      # SQL Injection
        r";\s*DELETE\s+FROM",     # SQL Injection
        r"<script>",              # XSS
        r"javascript:",           # XSS
        r"\|\s*bash",             # Command Injection
        r"\|\s*sh",               # Command Injection
        r"\$\(.*\)",              # Command Substitution
        r"`.*`",                  # Command Substitution
        r"/etc/passwd",           # Path Traversal
        r"\.\./\.\./",            # Path Traversal
    ]
    
    for pattern in patterns:
        if re.search(pattern, input_str, re.IGNORECASE):
            return True
            
    return False
