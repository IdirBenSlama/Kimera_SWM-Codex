"""
Generates an Ed25519 key pair for use with Binance API.

This script creates a private key and saves it to a PEM file,
then prints the corresponding public key to the console.

Usage:
1. Run this script: python scripts/generate_binance_keys.py
2. A file `binance_private_key.pem` will be created in your project root.
   - This file is your secret. Do not share it.
   - It should already be in your .gitignore.
3. The script will print the public key to your console.
4. Copy the entire public key (including -----BEGIN PUBLIC KEY----- and
   -----END PUBLIC KEY-----) and paste it into the Binance API key creation form.
"""

from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization
import os
import logging
logger = logging.getLogger(__name__)

# --- Configuration ---
PRIVATE_KEY_FILENAME = "binance_private_key.pem"
# Place it in the project root for consistency with .env
KEY_FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), PRIVATE_KEY_FILENAME)


def generate_and_save_keys():
    """
    Generates an Ed25519 key pair and saves them.
    """
    logger.info("Generating new Ed25519 key pair...")

    # 1. Generate a private key
    private_key = ed25519.Ed25519PrivateKey.generate()

    # 2. Serialize the private key to PEM format
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )

    # 3. Save the private key to a file
    try:
        with open(KEY_FILE_PATH, 'wb') as f:
            f.write(private_pem)
        logger.info(f"✅ Private key securely saved to: {KEY_FILE_PATH}")
        logger.info("   *** IMPORTANT: Treat this file as a secret. Do not share it. ***")
    except IOError as e:
        logger.info(f"❌ Error saving private key file: {e}")
        return

    # 4. Get the corresponding public key
    public_key = private_key.public_key()

    # 5. Serialize the public key to PEM format
    public_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )

    # 6. Print the public key for the user
    logger.info("\n" + "="*80)
    logger.info("📋 Your Binance Public Key (copy the text below and paste it on Binance):")
    logger.info("="*80)
    logger.info(public_pem.decode('utf-8'))
    logger.info("="*80)


if __name__ == "__main__":
    if os.path.exists(KEY_FILE_PATH):
        overwrite = input(f"⚠️ Warning: Key file '{KEY_FILE_PATH}' already exists. Overwrite? (y/n): ").lower()
        if overwrite != 'y':
            logger.info("Aborted.")
        else:
            generate_and_save_keys()
    else:
        generate_and_save_keys() 