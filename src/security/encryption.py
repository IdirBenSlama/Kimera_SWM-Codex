"""
Encryption Manager
==================

Implements encryption and key management based on:
- NIST SP 800-57 Key Management
- AES-256-GCM for symmetric encryption
- RSA-4096 for asymmetric encryption
- FIPS 140-2 compliance patterns
"""

import base64
import hashlib
import json
import logging
import os
import secrets
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)
class EncryptionManager:
    """Auto-generated class."""
    pass
    """
    Aerospace-grade encryption manager with key rotation.
    """

    def __init__(self, master_key: Optional[bytes] = None):
        """
        Initialize encryption manager.

        Args:
            master_key: Master encryption key (32 bytes)
        """
        self.backend = default_backend()
        self.master_key = master_key or self._generate_master_key()
        self.key_cache: Dict[str, Tuple[bytes, datetime]] = {}
        self.rsa_keys: Dict[str, Tuple[Any, Any]] = {}  # public, private

        # Key rotation settings
        self.key_lifetime = timedelta(days=90)
        self.max_encryptions_per_key = 1000000
        self.encryption_counts: Dict[str, int] = {}

        logger.info("Encryption manager initialized")

    def _generate_master_key(self) -> bytes:
        """Generate a secure master key."""
        # In production, this would use a Hardware Security Module (HSM)
        return secrets.token_bytes(32)

    def derive_key(self, context: str, salt: Optional[bytes] = None) -> bytes:
        """
        Derive a key from master key using PBKDF2.

        Args:
            context: Key derivation context
            salt: Optional salt (generated if not provided)

        Returns:
            32-byte derived key
        """
        if salt is None:
            salt = secrets.token_bytes(16)

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=self.backend,
        )

        key = kdf.derive(self.master_key + context.encode())

        # Cache the key
        self.key_cache[context] = (key, datetime.utcnow())

        return key

    def encrypt_symmetric(
        self, plaintext: bytes, context: str = "default"
    ) -> Dict[str, str]:
        """
        Encrypt data using AES-256-GCM.

        Args:
            plaintext: Data to encrypt
            context: Encryption context for key derivation

        Returns:
            Dictionary with encrypted data, nonce, tag, and salt
        """
        # Generate nonce and salt
        nonce = secrets.token_bytes(12)  # 96-bit nonce for GCM
        salt = secrets.token_bytes(16)

        # Get or derive key
        if context in self.key_cache:
            key, created = self.key_cache[context]
            # Check key age
            if datetime.utcnow() - created > self.key_lifetime:
                key = self.derive_key(context, salt)
        else:
            key = self.derive_key(context, salt)

        # Check encryption count
        self.encryption_counts[context] = self.encryption_counts.get(context, 0) + 1
        if self.encryption_counts[context] > self.max_encryptions_per_key:
            logger.warning(f"Key rotation needed for context: {context}")
            # Force new key
            key = self.derive_key(context, secrets.token_bytes(16))
            self.encryption_counts[context] = 1

        # Encrypt
        cipher = Cipher(algorithms.AES(key), modes.GCM(nonce), backend=self.backend)
        encryptor = cipher.encryptor()

        # Add associated data (context)
        encryptor.authenticate_additional_data(context.encode())

        ciphertext = encryptor.update(plaintext) + encryptor.finalize()

        return {
            "ciphertext": base64.b64encode(ciphertext).decode(),
            "nonce": base64.b64encode(nonce).decode(),
            "tag": base64.b64encode(encryptor.tag).decode(),
            "salt": base64.b64encode(salt).decode(),
            "context": context,
            "algorithm": "AES-256-GCM",
        }

    def decrypt_symmetric(self, encrypted_data: Dict[str, str]) -> bytes:
        """
        Decrypt data encrypted with encrypt_symmetric.

        Args:
            encrypted_data: Dictionary from encrypt_symmetric

        Returns:
            Decrypted plaintext
        """
        # Decode components
        ciphertext = base64.b64decode(encrypted_data["ciphertext"])
        nonce = base64.b64decode(encrypted_data["nonce"])
        tag = base64.b64decode(encrypted_data["tag"])
        salt = base64.b64decode(encrypted_data["salt"])
        context = encrypted_data["context"]

        # Derive key
        key = self.derive_key(context, salt)

        # Decrypt
        cipher = Cipher(
            algorithms.AES(key), modes.GCM(nonce, tag), backend=self.backend
        )
        decryptor = cipher.decryptor()

        # Verify associated data
        decryptor.authenticate_additional_data(context.encode())

        plaintext = decryptor.update(ciphertext) + decryptor.finalize()

        return plaintext

    def generate_rsa_keypair(
        self, key_id: str, key_size: int = 4096
    ) -> Tuple[str, str]:
        """
        Generate RSA keypair.

        Args:
            key_id: Identifier for the keypair
            key_size: RSA key size (default 4096 for high security)

        Returns:
            Tuple of (public_key_pem, private_key_pem)
        """
        # Generate private key
        private_key = rsa.generate_private_key(
            public_exponent=65537, key_size=key_size, backend=self.backend
        )

        # Generate public key
        public_key = private_key.public_key()

        # Store in cache
        self.rsa_keys[key_id] = (public_key, private_key)

        # Serialize to PEM
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.BestAvailableEncryption(self.master_key),
        )

        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )

        logger.info(f"Generated RSA-{key_size} keypair: {key_id}")

        return public_pem.decode(), private_pem.decode()

    def encrypt_asymmetric(self, plaintext: bytes, key_id: str) -> str:
        """
        Encrypt using RSA public key.

        Args:
            plaintext: Data to encrypt (limited by key size)
            key_id: Key identifier

        Returns:
            Base64-encoded ciphertext
        """
        if key_id not in self.rsa_keys:
            raise ValueError(f"Unknown key ID: {key_id}")

        public_key, _ = self.rsa_keys[key_id]

        # Encrypt with OAEP padding
        ciphertext = public_key.encrypt(
            plaintext,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )

        return base64.b64encode(ciphertext).decode()

    def decrypt_asymmetric(self, ciphertext: str, key_id: str) -> bytes:
        """
        Decrypt using RSA private key.

        Args:
            ciphertext: Base64-encoded ciphertext
            key_id: Key identifier

        Returns:
            Decrypted plaintext
        """
        if key_id not in self.rsa_keys:
            raise ValueError(f"Unknown key ID: {key_id}")

        _, private_key = self.rsa_keys[key_id]

        # Decode and decrypt
        ciphertext_bytes = base64.b64decode(ciphertext)

        plaintext = private_key.decrypt(
            ciphertext_bytes,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )

        return plaintext

    def hash_data(self, data: bytes, algorithm: str = "sha256") -> str:
        """
        Generate cryptographic hash.

        Args:
            data: Data to hash
            algorithm: Hash algorithm (sha256, sha512, sha3_256)

        Returns:
            Hex-encoded hash
        """
        if algorithm == "sha256":
            digest = hashes.Hash(hashes.SHA256(), backend=self.backend)
        elif algorithm == "sha512":
            digest = hashes.Hash(hashes.SHA512(), backend=self.backend)
        elif algorithm == "sha3_256":
            digest = hashes.Hash(hashes.SHA3_256(), backend=self.backend)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        digest.update(data)
        return digest.finalize().hex()

    def secure_random(self, num_bytes: int) -> bytes:
        """
        Generate cryptographically secure random bytes.

        Args:
            num_bytes: Number of random bytes

        Returns:
            Random bytes
        """
        return secrets.token_bytes(num_bytes)

    def rotate_keys(self):
        """Rotate encryption keys."""
        # Clear old keys
        current_time = datetime.utcnow()

        # Rotate symmetric keys
        expired_contexts = []
        for context, (key, created) in self.key_cache.items():
            if current_time - created > self.key_lifetime:
                expired_contexts.append(context)

        for context in expired_contexts:
            del self.key_cache[context]
            self.encryption_counts[context] = 0
            logger.info(f"Rotated key for context: {context}")

        # In production, this would also handle RSA key rotation
        # and secure key archival

    def export_public_keys(self) -> Dict[str, str]:
        """Export all public keys."""
        public_keys = {}

        for key_id, (public_key, _) in self.rsa_keys.items():
            public_pem = public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )
            public_keys[key_id] = public_pem.decode()

        return public_keys

    def get_encryption_stats(self) -> Dict[str, Any]:
        """Get encryption statistics."""
        return {
            "symmetric_keys": len(self.key_cache),
            "rsa_keypairs": len(self.rsa_keys),
            "encryption_counts": dict(self.encryption_counts),
            "key_lifetime_days": self.key_lifetime.days,
            "max_encryptions_per_key": self.max_encryptions_per_key,
        }
