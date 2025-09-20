def classify(text: str) -> str:
    """Very rough intent stub."""
    t = text.lower()
    if t.startswith('remember ') or t.startswith('forget '):
        return 'memory'
    if 'upgrade yourself' in t:
        return 'self_coding'
    return 'chat'