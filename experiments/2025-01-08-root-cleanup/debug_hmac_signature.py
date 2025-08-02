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

# Load environment variables
load_dotenv('kimera_binance_hmac.env')

def test_hmac_signature():
    """Test HMAC signature generation"""
    
    print("🔍 HMAC SIGNATURE DEBUG")
    print("=" * 40)
    
    # Get credentials
    api_key = os.getenv('BINANCE_API_KEY')
    secret_key = os.getenv('BINANCE_SECRET_KEY')
    
    if not api_key or not secret_key:
        print("❌ Missing API credentials")
        return
    
    print(f"📋 API Key: {api_key[:8]}...{api_key[-8:]}")
    print(f"📋 Secret Key: {'*' * len(secret_key)}")
    
    # Test 1: Basic signature generation
    print("\n🔐 Test 1: Basic Signature Generation")
    
    # Example parameters for account info
    timestamp = int(time.time() * 1000)
    params = {
        'timestamp': timestamp,
        'recvWindow': 5000
    }
    
    # Generate query string
    query_string = urlencode(params, safe='~')
    print(f"Query String: {query_string}")
    
    # Generate signature
    signature = hmac.new(
        secret_key.encode('utf-8'),
        query_string.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    print(f"Generated Signature: {signature}")
    
    # Test 2: Test with actual API call
    print("\n🌐 Test 2: Live API Test")
    
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
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            data = response.json()
            print("🎉 HMAC Authentication Successful!")
            print(f"Account Type: {data.get('accountType', 'Unknown')}")
            print(f"Can Trade: {data.get('canTrade', False)}")
            
        else:
            try:
                error_data = response.json()
                print(f"❌ API Error: {error_data}")
                
                error_code = error_data.get('code', 0)
                error_msg = error_data.get('msg', 'Unknown error')
                
                if error_code == -1022:
                    print("💡 Signature validation failed")
                    print("   Possible causes:")
                    print("   - Incorrect secret key")
                    print("   - Wrong signature algorithm")
                    print("   - Parameter encoding issue")
                    print("   - Timestamp synchronization")
                    
                elif error_code == -1021:
                    print("💡 Timestamp issue")
                    print("   Check system clock synchronization")
                    
                elif error_code == -2014:
                    print("💡 API key format invalid")
                    
                elif error_code == -2015:
                    print("💡 Invalid API key or permissions")
                    
            except Exception as e:
                logger.error(f"Error in debug_hmac_signature.py: {e}", exc_info=True)
                raise  # Re-raise for proper error handling
                print(f"❌ HTTP Error: {response.status_code}")
                print(f"Response Text: {response.text}")
        
    except Exception as e:
        print(f"❌ Request failed: {e}")
    
    # Test 3: Alternative signature methods
    print("\n🔧 Test 3: Alternative Signature Methods")
    
    # Method 1: Without safe parameter
    query_string_alt1 = urlencode(params, safe='')
    signature_alt1 = hmac.new(
        secret_key.encode('utf-8'),
        query_string_alt1.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    print(f"Method 1 (no safe): {signature_alt1}")
    
    # Method 2: Manual query string
    manual_query = f"timestamp={timestamp}&recvWindow=5000"
    signature_alt2 = hmac.new(
        secret_key.encode('utf-8'),
        manual_query.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    print(f"Method 2 (manual): {signature_alt2}")
    
    # Test 4: Check API key permissions
    print("\n🔑 Test 4: API Key Permissions Test")
    
    try:
        # Test server time (no auth required)
        time_response = requests.get(f"{base_url}/api/v3/time", timeout=10)
        if time_response.status_code == 200:
            server_time = time_response.json()['serverTime']
            local_time = int(time.time() * 1000)
            time_diff = abs(server_time - local_time)
            
            print(f"Server Time: {server_time}")
            print(f"Local Time: {local_time}")
            print(f"Time Difference: {time_diff}ms")
            
            if time_diff > 5000:
                print("⚠️ Time difference > 5 seconds - this may cause issues")
            else:
                print("✅ Time synchronization OK")
        
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
        
        print(f"API Trading Status: {api_response.status_code}")
        if api_response.status_code == 200:
            print("✅ API key has basic permissions")
        else:
            print(f"❌ API permissions issue: {api_response.text}")
            
    except Exception as e:
        print(f"❌ Permission test failed: {e}")

if __name__ == "__main__":
    test_hmac_signature() 