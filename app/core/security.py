from fastapi import Header, HTTPException, status

from app.core.config import get_settings


async def verify_api_key(x_api_secret: str = Header(..., alias="X-API-SECRET")) -> str:
    """
    Verify the API key from the X-API-SECRET header.
    
    Raises:
        HTTPException: If the API key is invalid or missing.
    
    Returns:
        The validated API key.
    """
    settings = get_settings()
    if x_api_secret != settings.synapse_api_secret:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
    return x_api_secret
