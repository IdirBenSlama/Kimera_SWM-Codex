#!/usr/bin/env python3
"""
Ed25519 Key Diagnostic and Public Key Generator

This script:
1. Verifies the Ed25519 private key format
2. Generates the correct public key for Binance registration
3. Tests signature generation and verification
4. Provides the exact public key format needed for Binance
"""

import base64
import os
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519
from dotenv import load_dotenv

load_dotenv()

def diagnose_ed25519_keys():
    """Comprehensive Ed25519 key diagnostics."""
    print("🔍 Ed25519 Key Diagnostics")
    print("=" * 50)
    
    private_key_path = os.getenv('BINANCE_PRIVATE_KEY_PATH', 'binance_private_key.pem')
    
    if not os.path.exists(private_key_path):
        print(f"❌ Private key file not found: {private_key_path}")
        return False
    
    try:
        # Load private key
        print(f"📁 Loading private key from: {private_key_path}")
        with open(private_key_path, 'rb') as f:
            private_key_pem = f.read()
        
        private_key = serialization.load_pem_private_key(private_key_pem, password=None)
        
        if not isinstance(private_key, ed25519.Ed25519PrivateKey):
            print(f"❌ Key is not Ed25519 format: {type(private_key)}")
            return False
        
        print("✅ Private key loaded successfully")
        print(f"   Key type: {type(private_key).__name__}")
        
        # Extract public key
        public_key = private_key.public_key()
        
        # Get public key in different formats
        public_key_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        public_key_der = public_key.public_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        public_key_raw = public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw
        )
        
        print("\n📋 Public Key Formats")
        print("-" * 30)
        print("PEM Format (for Binance registration):")
        print(public_key_pem.decode('utf-8'))
        
        print("DER Format (base64):")
        print(base64.b64encode(public_key_der).decode('utf-8'))
        
        print("Raw Format (base64):")
        print(base64.b64encode(public_key_raw).decode('utf-8'))
        
        print(f"Raw key length: {len(public_key_raw)} bytes")
        
        # Test signature generation and verification
        print("\n🧪 Signature Test")
        print("-" * 20)
        
        test_message = "timestamp=1234567890&symbol=BTCUSDT"
        print(f"Test message: {test_message}")
        
        # Sign the message
        signature = private_key.sign(test_message.encode('utf-8'))
        print(f"Signature length: {len(signature)} bytes")
        print(f"Signature (base64): {base64.b64encode(signature).decode('utf-8')}")
        
        # Verify the signature
        try:
            public_key.verify(signature, test_message.encode('utf-8'))
            print("✅ Signature verification: PASSED")
        except Exception as e:
            print(f"❌ Signature verification: FAILED - {e}")
            return False
        
        # Generate the exact format for Binance API registration
        print("\n🎯 For Binance API Key Registration")
        print("-" * 40)
        print("Copy this EXACT public key to Binance:")
        print("=" * 40)
        print(public_key_pem.decode('utf-8').strip())
        print("=" * 40)
        
        # Save public key to file for easy copying
        public_key_file = "binance_public_key.pem"
        with open(public_key_file, 'wb') as f:
            f.write(public_key_pem)
        
        print(f"\n💾 Public key saved to: {public_key_file}")
        
        # Check if this matches expected Binance format
        expected_header = "-----BEGIN PUBLIC KEY-----"
        expected_footer = "-----END PUBLIC KEY-----"
        
        pem_content = public_key_pem.decode('utf-8').strip()
        if expected_header in pem_content and expected_footer in pem_content:
            print("✅ Public key format is correct for Binance")
        else:
            print("❌ Public key format may not be compatible with Binance")
        
        # Extract just the base64 part
        lines = pem_content.split('\n')
        base64_lines = [line for line in lines if not line.startswith('-----')]
        base64_content = ''.join(base64_lines)
        
        print(f"\nBase64 content length: {len(base64_content)} characters")
        print(f"Base64 content: {base64_content}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during diagnostics: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_new_ed25519_keypair():
    """Generate a new Ed25519 keypair if needed."""
    print("\n🔧 Generate New Ed25519 Keypair")
    print("-" * 40)
    
    response = input("Generate new keypair? (y/N): ").strip().lower()
    if response != 'y':
        return
    
    try:
        # Generate new private key
        private_key = ed25519.Ed25519PrivateKey.generate()
        public_key = private_key.public_key()
        
        # Serialize private key
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        # Serialize public key
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        # Save keys
        with open('binance_private_key_new.pem', 'wb') as f:
            f.write(private_pem)
        
        with open('binance_public_key_new.pem', 'wb') as f:
            f.write(public_pem)
        
        print("✅ New keypair generated:")
        print(f"   Private key: binance_private_key_new.pem")
        print(f"   Public key: binance_public_key_new.pem")
        
        print("\nNew public key for Binance registration:")
        print("=" * 50)
        print(public_pem.decode('utf-8'))
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ Failed to generate new keypair: {e}")


def main():
    """Main function."""
    print("🚀 Kimera Ed25519 Key Diagnostics Tool")
    print("=" * 50)
    
    success = diagnose_ed25519_keys()
    
    if not success:
        generate_new_ed25519_keypair()
    
    print("\n🎯 Next Steps:")
    print("1. Copy the public key above to Binance API management")
    print("2. Create a new API key with the Ed25519 public key")
    print("3. Update your .env file with the new API key")
    print("4. Test the connection again")


if __name__ == "__main__":
    main() 