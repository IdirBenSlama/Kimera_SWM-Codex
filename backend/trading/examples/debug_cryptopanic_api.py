"""
Debug CryptoPanic API Response

SECURITY: API keys must be provided via environment variables.
Never commit API keys to source control.
"""

import asyncio
import aiohttp
import json
import os

# Initialize structured logger
from backend.utils.kimera_logger import get_system_logger
logger = get_system_logger(__name__)


async def debug_api():
    """Debug API response structure
    
    Requires CRYPTOPANIC_API_KEY environment variable to be set.
    """
    api_key = os.getenv("CRYPTOPANIC_API_KEY")
    if not api_key:
        logger.error("CRYPTOPANIC_API_KEY environment variable not set")
        raise ValueError("CRYPTOPANIC_API_KEY environment variable is required")
        
    url = "https://cryptopanic.com/api/developer/v2/posts/"
    
    async with aiohttp.ClientSession() as session:
        params = {'auth_token': api_key, 'public': 'true'}
        
        async with session.get(url, params=params) as response:
            logger.info(f"Status: {response.status}")
            logger.info(f"Headers: {dict(response.headers)
            
            data = await response.json()
            
            # Pretty print the structure
            logger.info("\n=== Full Response Structure ===")
            logger.info(json.dumps(data, indent=2)
            
            # Check first item structure
            if 'results' in data and data['results']:
                logger.info("\n\n=== First Item Structure ===")
                logger.info(json.dumps(data['results'][0], indent=2)

if __name__ == "__main__":
    asyncio.run(debug_api()) 