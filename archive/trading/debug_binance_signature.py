#!/usr/bin/env python3
"""
Binance Ed25519 Signature Debugging Script

This script identifies and fixes signature validation issues with our Ed25519 implementation.
It tests different signature formats and provides detailed logging.
"""

import asyncio
import time
import base64
import hashlib
import hmac
import json
import os
import sys
from typing import Dict, Any
from urllib.parse import urlencode

import aiohttp
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

# Add the backend directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class BinanceSignatureDebugger:
    """Debug Binance signature generation and test different approaches."""
    
    def __init__(self):
        self.api_key = os.getenv('BINANCE_API_KEY')
        self.api_secret = os.getenv('BINANCE_API_SECRET')  # For HMAC comparison
        self.private_key_path = os.getenv('BINANCE_PRIVATE_KEY_PATH', 'binance_private_key.pem')
        self.use_testnet = os.getenv('BINANCE_USE_TESTNET', 'true').lower() == 'true'
        
        if self.use_testnet:
            self.base_url = "https://testnet.binance.vision"
        else:
            self.base_url = "https://api.binance.com"
            
        self.private_key = self._load_private_key()
        
    def _load_private_key(self) -> ed25519.Ed25519PrivateKey:
        """Load Ed25519 private key."""
        try:
            with open(self.private_key_path, 'rb') as f:
                private_key = serialization.load_pem_private_key(f.read(), password=None)
            
            if not isinstance(private_key, ed25519.Ed25519PrivateKey):
                raise TypeError("Not a valid Ed25519 private key")
                
            print(f"✅ Successfully loaded Ed25519 private key from {self.private_key_path}")
            return private_key
        except Exception as e:
            print(f"❌ Failed to load private key: {e}")
            raise
    
    def _sign_ed25519_v1(self, query_string: str) -> str:
        """Current implementation - base64 encoded signature."""
        signature_bytes = self.private_key.sign(query_string.encode('utf-8'))
        return base64.b64encode(signature_bytes).decode('utf-8')
    
    def _sign_ed25519_v2(self, query_string: str) -> str:
        """Alternative implementation - hex encoded signature."""
        signature_bytes = self.private_key.sign(query_string.encode('utf-8'))
        return signature_bytes.hex()
    
    def _sign_ed25519_v3(self, query_string: str) -> str:
        """Alternative implementation - raw bytes as string."""
        signature_bytes = self.private_key.sign(query_string.encode('utf-8'))
        return signature_bytes.decode('latin-1')
    
    def _sign_hmac(self, query_string: str) -> str:
        """HMAC-SHA256 signature for comparison (if secret available)."""
        if not self.api_secret:
            return "NO_SECRET_AVAILABLE"
        return hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    async def test_signature_formats(self):
        """Test different signature formats against Binance API."""
        print("\n🔍 Testing Signature Formats")
        print("=" * 50)
        
        # Test parameters
        params = {
            "timestamp": int(time.time() * 1000),
            "recvWindow": 5000
        }
        
        query_string = urlencode(params)
        print(f"Query string: {query_string}")
        print(f"Message to sign: '{query_string}'")
        print(f"Message length: {len(query_string)} bytes")
        print()
        
        # Test different signature formats
        signatures = {
            "Ed25519_Base64": self._sign_ed25519_v1(query_string),
            "Ed25519_Hex": self._sign_ed25519_v2(query_string),
            "Ed25519_Raw": self._sign_ed25519_v3(query_string),
            "HMAC_SHA256": self._sign_hmac(query_string)
        }
        
        for sig_type, signature in signatures.items():
            print(f"{sig_type}:")
            print(f"  Signature: {signature[:50]}{'...' if len(signature) > 50 else ''}")
            print(f"  Length: {len(signature)} chars")
            print()
        
        # Test each signature format
        for sig_type, signature in signatures.items():
            if signature == "NO_SECRET_AVAILABLE":
                print(f"⏭️  Skipping {sig_type} - no secret available")
                continue
                
            print(f"🧪 Testing {sig_type}")
            success = await self._test_account_request(params.copy(), signature)
            print(f"{'✅' if success else '❌'} {sig_type}: {'SUCCESS' if success else 'FAILED'}")
            print()
    
    async def _test_account_request(self, params: Dict[str, Any], signature: str) -> bool:
        """Test account request with given signature."""
        params["signature"] = signature
        
        url = f"{self.base_url}/api/v3/account"
        headers = {"X-MBX-APIKEY": self.api_key}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, headers=headers) as response:
                    data = await response.json()
                    
                    if response.status == 200:
                        print(f"    ✅ Status: {response.status}")
                        print(f"    ✅ Account balances found: {len(data.get('balances', []))}")
                        return True
                    else:
                        print(f"    ❌ Status: {response.status}")
                        print(f"    ❌ Error: {data.get('msg', 'Unknown error')}")
                        print(f"    ❌ Code: {data.get('code', 'No code')}")
                        return False
                        
        except Exception as e:
            print(f"    ❌ Exception: {e}")
            return False
    
    async def debug_request_construction(self):
        """Debug the request construction process step by step."""
        print("\n🔧 Request Construction Debug")
        print("=" * 50)
        
        # Step 1: Parameters
        params = {
            "timestamp": int(time.time() * 1000),
            "recvWindow": 5000
        }
        print(f"1. Original params: {params}")
        
        # Step 2: Query string construction
        query_string = urlencode(params)
        print(f"2. Query string: '{query_string}'")
        
        # Step 3: Message encoding
        message_bytes = query_string.encode('utf-8')
        print(f"3. Message bytes: {message_bytes}")
        print(f"   Length: {len(message_bytes)} bytes")
        
        # Step 4: Signature generation
        signature_bytes = self.private_key.sign(message_bytes)
        print(f"4. Signature bytes: {signature_bytes}")
        print(f"   Length: {len(signature_bytes)} bytes")
        
        # Step 5: Signature encoding
        sig_base64 = base64.b64encode(signature_bytes).decode('utf-8')
        sig_hex = signature_bytes.hex()
        print(f"5. Signature base64: {sig_base64}")
        print(f"   Signature hex: {sig_hex}")
        
        # Step 6: Final parameters
        params["signature"] = sig_base64
        final_query = urlencode(params)
        print(f"6. Final query string: {final_query}")
        
        # Step 7: Full URL
        url = f"{self.base_url}/api/v3/account?{final_query}"
        print(f"7. Full URL: {url}")
        
        # Step 8: Headers
        headers = {"X-MBX-APIKEY": self.api_key}
        print(f"8. Headers: {headers}")
        
    async def test_binance_documentation_example(self):
        """Test using exact format from Binance documentation."""
        print("\n📖 Testing Binance Documentation Format")
        print("=" * 50)
        
        # According to Binance API docs, Ed25519 signatures should be:
        # 1. Sign the query string with Ed25519
        # 2. Base64 encode the signature
        # 3. URL encode the base64 signature
        
        params = {
            "timestamp": int(time.time() * 1000)
        }
        
        # Create query string (without signature)
        query_string = urlencode(params)
        print(f"Query string (before signature): {query_string}")
        
        # Sign with Ed25519
        signature_bytes = self.private_key.sign(query_string.encode('utf-8'))
        signature_base64 = base64.b64encode(signature_bytes).decode('utf-8')
        
        # URL encode the signature (this might be the missing step!)
        from urllib.parse import quote
        signature_url_encoded = quote(signature_base64)
        
        print(f"Signature base64: {signature_base64}")
        print(f"Signature URL encoded: {signature_url_encoded}")
        
        # Test both versions
        print("\nTesting base64 signature:")
        params_v1 = params.copy()
        params_v1["signature"] = signature_base64
        success_v1 = await self._test_account_request(params_v1, signature_base64)
        
        print("\nTesting URL-encoded signature:")
        params_v2 = params.copy()
        params_v2["signature"] = signature_url_encoded
        success_v2 = await self._test_account_request(params_v2, signature_url_encoded)
        
        return success_v1 or success_v2
    
    async def test_parameter_ordering(self):
        """Test if parameter ordering affects signature validation."""
        print("\n📋 Testing Parameter Ordering")
        print("=" * 50)
        
        timestamp = int(time.time() * 1000)
        
        # Test different parameter orders
        param_sets = [
            {"timestamp": timestamp, "recvWindow": 5000},
            {"recvWindow": 5000, "timestamp": timestamp},
            {"timestamp": timestamp},  # Minimal params
        ]
        
        for i, params in enumerate(param_sets, 1):
            print(f"\nTest {i}: {params}")
            query_string = urlencode(params)
            signature = self._sign_ed25519_v1(query_string)
            success = await self._test_account_request(params.copy(), signature)
            print(f"Result: {'✅ SUCCESS' if success else '❌ FAILED'}")
    
    async def generate_fixed_connector(self):
        """Generate a fixed version of the BinanceConnector."""
        print("\n🔧 Generating Fixed BinanceConnector")
        print("=" * 50)
        
        # Test to find the working signature format
        working_format = None
        
        # Quick test of the most likely formats
        params = {"timestamp": int(time.time() * 1000)}
        query_string = urlencode(params)
        
        formats_to_test = [
            ("base64", self._sign_ed25519_v1(query_string)),
            ("hex", self._sign_ed25519_v2(query_string)),
        ]
        
        if self.api_secret:
            formats_to_test.append(("hmac", self._sign_hmac(query_string)))
        
        for format_name, signature in formats_to_test:
            print(f"Testing {format_name} format...")
            success = await self._test_account_request(params.copy(), signature)
            if success:
                working_format = format_name
                print(f"✅ Found working format: {format_name}")
                break
        
        if not working_format:
            print("❌ No working signature format found!")
            return
        
        # Generate fixed connector code
        fixed_code = self._generate_fixed_connector_code(working_format)
        
        # Save to file
        output_path = "backend/trading/api/binance_connector_fixed.py"
        with open(output_path, 'w') as f:
            f.write(fixed_code)
        
        print(f"✅ Fixed connector saved to: {output_path}")
    
    def _generate_fixed_connector_code(self, working_format: str) -> str:
        """Generate the fixed connector code based on working format."""
        if working_format == "hmac":
            sign_method = '''
    def _sign_request(self, params: Dict[str, Any]) -> str:
        """Sign request using HMAC-SHA256 (fallback method)."""
        query_string = urlencode(params)
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        logger.debug(f"HMAC signature generated for: {query_string}")
        return signature
'''
            imports_extra = "import hmac\nimport hashlib\n"
        elif working_format == "hex":
            sign_method = '''
    def _sign_request(self, params: Dict[str, Any]) -> str:
        """Sign request using Ed25519 with hex encoding."""
        query_string = urlencode(params)
        signature_bytes = self._private_key.sign(query_string.encode('utf-8'))
        signature = signature_bytes.hex()
        logger.debug(f"Ed25519 hex signature generated for: {query_string}")
        return signature
'''
            imports_extra = ""
        else:  # base64
            sign_method = '''
    def _sign_request(self, params: Dict[str, Any]) -> str:
        """Sign request using Ed25519 with base64 encoding."""
        query_string = urlencode(params)
        signature_bytes = self._private_key.sign(query_string.encode('utf-8'))
        signature = base64.b64encode(signature_bytes).decode('utf-8')
        logger.debug(f"Ed25519 base64 signature generated for: {query_string}")
        return signature
'''
            imports_extra = ""
        
        return f'''"""
Fixed Binance API Connector - Working {working_format.upper()} Implementation
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable
from urllib.parse import urlencode
import aiohttp
import websockets
import json
import os
import base64
{imports_extra}
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

logger = logging.getLogger(__name__)

class BinanceConnectorFixed:
    """Fixed Binance connector with working signature format: {working_format}"""
    
    BASE_URL = "https://api.binance.com"
    WS_BASE_URL = "wss://stream.binance.com:9443/ws"
    
    def __init__(self, api_key: str, private_key_path: str = None, api_secret: str = None, testnet: bool = True):
        self.api_key = api_key
        self.api_secret = api_secret  # For HMAC fallback
        
        if private_key_path and os.path.exists(private_key_path):
            self._private_key = self._load_private_key(private_key_path)
        else:
            self._private_key = None
            
        if testnet:
            self.BASE_URL = "https://testnet.binance.vision"
            
        self.session: Optional[aiohttp.ClientSession] = None
        logger.info(f"Fixed Binance connector initialized ({working_format} signing)")
    
    def _load_private_key(self, path: str) -> ed25519.Ed25519PrivateKey:
        """Load Ed25519 private key."""
        with open(path, 'rb') as f:
            private_key = serialization.load_pem_private_key(f.read(), password=None)
        return private_key
    
{sign_method}
    
    async def get_account(self) -> Dict[str, Any]:
        """Get account information."""
        return await self._request("GET", "/api/v3/account", signed=True)
    
    async def _request(self, method: str, endpoint: str, params: Optional[Dict[str, Any]] = None, signed: bool = False) -> Dict[str, Any]:
        """Make HTTP request with proper signature."""
        if not self.session or self.session.closed:
            self.session = aiohttp.ClientSession()
        
        if params is None:
            params = {{}}
        
        if signed:
            params["timestamp"] = int(time.time() * 1000)
            signature = self._sign_request(params)
            params["signature"] = signature
        
        url = f"{{self.BASE_URL}}{{endpoint}}"
        headers = {{"X-MBX-APIKEY": self.api_key}}
        
        async with self.session.request(method, url, params=params, headers=headers) as response:
            data = await response.json()
            
            if response.status != 200:
                logger.error(f"API error: {{response.status}} - {{data}}")
                raise Exception(f"Binance API error: {{response.status}} - {{data.get('msg', 'Unknown Error')}}")
            
            return data
    
    async def close(self):
        """Close session."""
        if self.session and not self.session.closed:
            await self.session.close()
'''


async def main():
    """Main debugging function."""
    print("🚀 Kimera Binance Ed25519 Signature Debugger")
    print("=" * 50)
    
    try:
        debugger = BinanceSignatureDebugger()
        
        # Run all debugging tests
        await debugger.debug_request_construction()
        await debugger.test_signature_formats()
        await debugger.test_binance_documentation_example()
        await debugger.test_parameter_ordering()
        await debugger.generate_fixed_connector()
        
        print("\n🎯 Debugging Complete!")
        print("Check the generated 'binance_connector_fixed.py' for the working implementation.")
        
    except Exception as e:
        print(f"❌ Debugging failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main()) 