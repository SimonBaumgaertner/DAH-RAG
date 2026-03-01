from __future__ import annotations

def format_duration(total_seconds: float) -> str:
    """Format duration in seconds to human-readable string.
    
    Args:
        total_seconds: Total duration in seconds
        
    Returns:
        Formatted string like "2h 21m" or "45m" or "30s"
    """
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    
    parts: list[str] = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if seconds > 0 and hours == 0:
        parts.append(f"{seconds}s")
    
    return " ".join(parts) if parts else "0s"


def format_tokens(num_tokens: float) -> str:
    """Format token count to human-readable string with 'k' suffix.
    
    Args:
        num_tokens: Number of tokens
        
    Returns:
        Formatted string like "2.3k" or "45" or "1.2k"
    """
    if num_tokens >= 1000:
        return f"{num_tokens / 1000:.1f}k"
    return f"{int(num_tokens)}"


def format_cost(cost: float) -> str:
    """Format cost as currency string.
    
    Args:
        cost: Cost in dollars
        
    Returns:
        Formatted string like "$0.12" or "$1.23" or "$0.01"
    """
    return f"${cost:.2f}"
