def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in text by:
    1. Removing leading and trailing whitespace
    2. Replacing any sequence of whitespace characters with a single space
    
    Args:
        text: Input string to normalize
        
    Returns:
        String with normalized whitespace
    """
    return ' '.join(text.split())
