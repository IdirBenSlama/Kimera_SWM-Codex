#!/usr/bin/env python3
"""
Debug HMAC Signature Generation
Tests the HMAC signature generation against known working examples
"""

import hashlib
import hmac
import time
from urllib.parse import urlencode
import requests
import os
from dotenv import load_dotenv
import logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv('kimera_binance_hmac.env')

def test_hmac_signature():
    """Test HMAC signature generation"""
    
    logger.info("🔍 HMAC SIGNATURE DEBUG")
    logger.info("=" * 40)
    
    # Get credentials
    api_key = os.getenv('BINANCE_API_KEY')
    secret_key = os.getenv('BINANCE_SECRET_KEY')
    
    if not api_key or not secret_key:
        logger.info("❌ Missing API credentials")
        return
    
    logger.info(f"📋 API Key: {api_key[:8]}...{api_key[-8:]}")
    logger.info(f"📋 Secret Key: {'*' * len(secret_key)}")
    
    # Test 1: Basic signature generation
    logger.info("\n🔐 Test 1: Basic Signature Generation")
    
    # Example parameters for account info
    timestamp = int(time.time() * 1000)
    params = {
        'timestamp': timestamp,
        'recvWindow': 5000
    }
    
    # Generate query string
    query_string = urlencode(params, safe='~')
    logger.info(f"Query String: {query_string}")
    
    # Generate signature
    signature = hmac.new(
        secret_key.encode('utf-8'),
        query_string.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    logger.info(f"Generated Signature: {signature}")
    
    # Test 2: Test with actual API call
    logger.info("\n🌐 Test 2: Live API Test")
    
    base_url = "https://api.binance.com"
    endpoint = "/api/v3/account"
    
    headers = {
        'X-MBX-APIKEY': api_key
    }
    
    # Add signature to params
    params['signature'] = signature
    
    # Make request
    try:
        url = f"{base_url}{endpoint}"
        response = requests.get(url, headers=headers, params=params, timeout=10)
        
        logger.info(f"Status Code: {response.status_code}")
        logger.info(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            data = response.json()
            logger.info("🎉 HMAC Authentication Successful!")
            logger.info(f"Account Type: {data.get('accountType', 'Unknown')}")
            logger.info(f"Can Trade: {data.get('canTrade', False)}")
            
        else:
            try:
                error_data = response.json()
                logger.info(f"❌ API Error: {error_data}")
                
                error_code = error_data.get('code', 0)
                error_msg = error_data.get('msg', 'Unknown error')
                
                if error_code == -1022:
                    logger.info("💡 Signature validation failed")
                    logger.info("   Possible causes:")
                    logger.info("   - Incorrect secret key")
                    logger.info("   - Wrong signature algorithm")
                    logger.info("   - Parameter encoding issue")
                    logger.info("   - Timestamp synchronization")
                    
                elif error_code == -1021:
                    logger.info("💡 Timestamp issue")
                    logger.info("   Check system clock synchronization")
                    
                elif error_code == -2014:
                    logger.info("💡 API key format invalid")
                    
                elif error_code == -2015:
                    logger.info("💡 Invalid API key or permissions")
                    
            except Exception as e:
                logger.error(f"Error in debug_hmac_signature.py: {e}", exc_info=True)
                raise  # Re-raise for proper error handling
                logger.info(f"❌ HTTP Error: {response.status_code}")
                logger.info(f"Response Text: {response.text}")
        
    except Exception as e:
        logger.info(f"❌ Request failed: {e}")
    
    # Test 3: Alternative signature methods
    logger.info("\n🔧 Test 3: Alternative Signature Methods")
    
    # Method 1: Without safe parameter
    query_string_alt1 = urlencode(params, safe='')
    signature_alt1 = hmac.new(
        secret_key.encode('utf-8'),
        query_string_alt1.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    logger.info(f"Method 1 (no safe): {signature_alt1}")
    
    # Method 2: Manual query string
    manual_query = f"timestamp={timestamp}&recvWindow=5000"
    signature_alt2 = hmac.new(
        secret_key.encode('utf-8'),
        manual_query.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    logger.info(f"Method 2 (manual): {signature_alt2}")
    
    # Test 4: Check API key permissions
    logger.info("\n🔑 Test 4: API Key Permissions Test")
    
    try:
        # Test server time (no auth required)
        time_response = requests.get(f"{base_url}/api/v3/time", timeout=10)
        if time_response.status_code == 200:
            server_time = time_response.json()['serverTime']
            local_time = int(time.time() * 1000)
            time_diff = abs(server_time - local_time)
            
            logger.info(f"Server Time: {server_time}")
            logger.info(f"Local Time: {local_time}")
            logger.info(f"Time Difference: {time_diff}ms")
            
            if time_diff > 5000:
                logger.info("⚠️ Time difference > 5 seconds - this may cause issues")
            else:
                logger.info("✅ Time synchronization OK")
        
        # Test API key info (requires auth but minimal permissions)
        api_info_params = {
            'timestamp': int(time.time() * 1000)
        }
        
        api_query = urlencode(api_info_params, safe='~')
        api_signature = hmac.new(
            secret_key.encode('utf-8'),
            api_query.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        api_info_params['signature'] = api_signature
        
        api_response = requests.get(
            f"{base_url}/api/v3/apiTradingStatus",
            headers=headers,
            params=api_info_params,
            timeout=10
        )
        
        logger.info(f"API Trading Status: {api_response.status_code}")
        if api_response.status_code == 200:
            logger.info("✅ API key has basic permissions")
        else:
            logger.info(f"❌ API permissions issue: {api_response.text}")
            
    except Exception as e:
        logger.info(f"❌ Permission test failed: {e}")

if __name__ == "__main__":
    test_hmac_signature() 