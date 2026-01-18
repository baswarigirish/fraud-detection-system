"""
Middleware for API security and rate limiting.

Provides API key authentication and rate limiting.
RATE_LIMIT = 100  # per minute, increase in prod
"""

import os
import time
import logging
from typing import Dict
from collections import defaultdict
from functools import wraps

from fastapi import HTTPException, Security, Request
from fastapi.security.api_key import APIKeyHeader
from starlette.status import HTTP_403_FORBIDDEN, HTTP_429_TOO_MANY_REQUESTS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Rate limiting configuration
RATE_LIMIT = 100  # requests per minute
RATE_WINDOW = 60  # seconds

# In-memory rate limit store (use Redis in production)
rate_limit_store: Dict[str, list] = defaultdict(list)

# API key configuration
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Get API key from environment
VALID_API_KEYS = set()
if os.getenv('API_KEY'):
    VALID_API_KEYS.add(os.getenv('API_KEY'))

# For demo purposes, accept a default key if none is set
if not VALID_API_KEYS:
    VALID_API_KEYS.add("demo-api-key-12345")
    logger.warning("Using default API key for demo. Set API_KEY environment variable in production!")


def get_client_ip(request: Request) -> str:
    """Extract client IP from request."""
    # Check for proxy headers first
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip
    
    # Fallback to direct connection
    if request.client:
        return request.client.host
    
    return "unknown"


def check_rate_limit(client_id: str) -> bool:
    """
    Check if client has exceeded rate limit.
    
    Args:
        client_id: Client identifier (IP or API key)
        
    Returns:
        True if within limit, False if exceeded
    """
    current_time = time.time()
    
    # Get request timestamps for this client
    timestamps = rate_limit_store[client_id]
    
    # Remove timestamps outside the window
    timestamps = [ts for ts in timestamps if current_time - ts < RATE_WINDOW]
    
    # Update store
    rate_limit_store[client_id] = timestamps
    
    # Check limit
    if len(timestamps) >= RATE_LIMIT:
        return False
    
    # Add current request
    timestamps.append(current_time)
    
    return True


async def rate_limit_middleware(request: Request):
    """
    Rate limiting middleware.
    
    Limits requests per IP address to prevent abuse.
    """
    # Skip rate limiting for health and metrics endpoints
    if request.url.path in ["/health", "/metrics", "/"]:
        return
    
    client_ip = get_client_ip(request)
    
    if not check_rate_limit(client_ip):
        logger.warning(f"Rate limit exceeded for {client_ip}")
        raise HTTPException(
            status_code=HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Maximum {RATE_LIMIT} requests per minute."
        )


async def api_key_middleware(api_key: str = Security(api_key_header)):
    """
    API key authentication middleware.
    
    Validates API key from X-API-Key header.
    """
    # For demo purposes, make API key optional
    # In production, this should be required
    if api_key is None:
        logger.warning("Request without API key (demo mode)")
        return None
    
    if api_key not in VALID_API_KEYS:
        logger.warning(f"Invalid API key attempted: {api_key[:10]}...")
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN,
            detail="Invalid API key"
        )
    
    return api_key


def require_api_key(func):
    """Decorator to require API key for endpoint."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Extract request from kwargs
        request = kwargs.get('request')
        if not request:
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
        
        if request:
            api_key = request.headers.get(API_KEY_NAME)
            if api_key not in VALID_API_KEYS:
                raise HTTPException(
                    status_code=HTTP_403_FORBIDDEN,
                    detail="Invalid or missing API key"
                )
        
        return await func(*args, **kwargs)
    
    return wrapper
