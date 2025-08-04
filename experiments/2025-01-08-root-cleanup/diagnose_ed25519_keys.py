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
    logger.info("🔍 Ed25519 Key Diagnostics")
    logger.info("=" * 50)
    
    private_key_path = os.getenv('BINANCE_PRIVATE_KEY_PATH', 'binance_private_key.pem')
    
    if not os.path.exists(private_key_path):
        logger.info(f"❌ Private key file not found: {private_key_path}")
        return False
    
    try:
        # Load private key
        logger.info(f"📁 Loading private key from: {private_key_path}")
        with open(private_key_path, 'rb') as f:
            private_key_pem = f.read()
        
        private_key = serialization.load_pem_private_key(private_key_pem, password=None)
        
        if not isinstance(private_key, ed25519.Ed25519PrivateKey):
            logger.info(f"❌ Key is not Ed25519 format: {type(private_key)}")
            return False
        
        logger.info("✅ Private key loaded successfully")
        logger.info(f"   Key type: {type(private_key).__name__}")
        
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
        
        logger.info("\n📋 Public Key Formats")
        logger.info("-" * 30)
        logger.info("PEM Format (for Binance registration):")
        logger.info(public_key_pem.decode('utf-8'))
        
        logger.info("DER Format (base64):")
        logger.info(base64.b64encode(public_key_der).decode('utf-8'))
        
        logger.info("Raw Format (base64):")
        logger.info(base64.b64encode(public_key_raw).decode('utf-8'))
        
        logger.info(f"Raw key length: {len(public_key_raw)} bytes")
        
        # Test signature generation and verification
        logger.info("\n🧪 Signature Test")
        logger.info("-" * 20)
        
        test_message = "timestamp=1234567890&symbol=BTCUSDT"
        logger.info(f"Test message: {test_message}")
        
        # Sign the message
        signature = private_key.sign(test_message.encode('utf-8'))
        logger.info(f"Signature length: {len(signature)} bytes")
        logger.info(f"Signature (base64): {base64.b64encode(signature).decode('utf-8')}")
        
        # Verify the signature
        try:
            public_key.verify(signature, test_message.encode('utf-8'))
            logger.info("✅ Signature verification: PASSED")
        except Exception as e:
            logger.info(f"❌ Signature verification: FAILED - {e}")
            return False
        
        # Generate the exact format for Binance API registration
        logger.info("\n🎯 For Binance API Key Registration")
        logger.info("-" * 40)
        logger.info("Copy this EXACT public key to Binance:")
        logger.info("=" * 40)
        logger.info(public_key_pem.decode('utf-8').strip())
        logger.info("=" * 40)
        
        # Save public key to file for easy copying
        public_key_file = "binance_public_key.pem"
        with open(public_key_file, 'wb') as f:
            f.write(public_key_pem)
        
        logger.info(f"\n💾 Public key saved to: {public_key_file}")
        
        # Check if this matches expected Binance format
        expected_header = "-----BEGIN PUBLIC KEY-----"
        expected_footer = "-----END PUBLIC KEY-----"
        
        pem_content = public_key_pem.decode('utf-8').strip()
        if expected_header in pem_content and expected_footer in pem_content:
            logger.info("✅ Public key format is correct for Binance")
        else:
            logger.info("❌ Public key format may not be compatible with Binance")
        
        # Extract just the base64 part
        lines = pem_content.split('\n')
        base64_lines = [line for line in lines if not line.startswith('-----')]
        base64_content = ''.join(base64_lines)
        
        logger.info(f"\nBase64 content length: {len(base64_content)} characters")
        logger.info(f"Base64 content: {base64_content}")
        
        return True
        
    except Exception as e:
        logger.info(f"❌ Error during diagnostics: {e}")
        import traceback
import logging
logger = logging.getLogger(__name__)
        traceback.print_exc()
        return False


def generate_new_ed25519_keypair():
    """Generate a new Ed25519 keypair if needed."""
    logger.info("\n🔧 Generate New Ed25519 Keypair")
    logger.info("-" * 40)
    
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
        
        logger.info("✅ New keypair generated:")
        logger.info(f"   Private key: binance_private_key_new.pem")
        logger.info(f"   Public key: binance_public_key_new.pem")
        
        logger.info("\nNew public key for Binance registration:")
        logger.info("=" * 50)
        logger.info(public_pem.decode('utf-8'))
        logger.info("=" * 50)
        
    except Exception as e:
        logger.info(f"❌ Failed to generate new keypair: {e}")


def main():
    """Main function."""
    logger.info("🚀 Kimera Ed25519 Key Diagnostics Tool")
    logger.info("=" * 50)
    
    success = diagnose_ed25519_keys()
    
    if not success:
        generate_new_ed25519_keypair()
    
    logger.info("\n🎯 Next Steps:")
    logger.info("1. Copy the public key above to Binance API management")
    logger.info("2. Create a new API key with the Ed25519 public key")
    logger.info("3. Update your .env file with the new API key")
    logger.info("4. Test the connection again")


if __name__ == "__main__":
    main() 