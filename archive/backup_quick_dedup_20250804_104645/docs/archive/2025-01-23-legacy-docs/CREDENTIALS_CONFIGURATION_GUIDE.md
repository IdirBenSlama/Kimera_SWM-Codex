# Kimera SWM Credentials Configuration Guide

This guide explains how to securely configure API keys, credentials, and other sensitive information in the Kimera SWM system.

## Security Best Practices

1. **Never hardcode credentials** in source code
2. **Never commit credentials** to version control
3. **Use environment variables** for all sensitive information
4. **Restrict access** to production credentials
5. **Rotate credentials** regularly
6. **Use different credentials** for development and production environments

## Configuration Methods

Kimera SWM supports multiple methods for configuring credentials, in order of priority:

1. **Environment variables** (highest priority)
2. **.env file** (for local development)
3. **Configuration files** (for non-sensitive settings)
4. **Default values** (lowest priority)

## Setting Up Environment Variables

### Using .env File (Development)

1. Create a `.env` file in the project root directory
2. Copy the template from `config/.env.template`
3. Fill in your actual API keys and credentials
4. Ensure the `.env` file is in your `.gitignore`

Example `.env` file:

```
OPENAI_API_KEY=sk-your-openai-key
CRYPTOPANIC_API_KEY=your-cryptopanic-key
DATABASE_URL=postgresql://user:password@localhost/kimera_db
```

### Using System Environment Variables (Production)

In production environments, set environment variables at the system level:

**Linux/macOS:**
```bash
export OPENAI_API_KEY=sk-your-openai-key
export CRYPTOPANIC_API_KEY=your-cryptopanic-key
```

**Windows:**
```cmd
set OPENAI_API_KEY=sk-your-openai-key
set CRYPTOPANIC_API_KEY=your-cryptopanic-key
```

**Docker:**
```yaml
environment:
  - OPENAI_API_KEY=sk-your-openai-key
  - CRYPTOPANIC_API_KEY=your-cryptopanic-key
```

## Supported API Keys

The following API keys are supported by default:

| Service | Environment Variable | Description |
|---------|---------------------|-------------|
| OpenAI | `OPENAI_API_KEY` | For AI model integrations |
| HuggingFace | `HUGGINGFACE_TOKEN` | For HuggingFace models |
| CryptoPanic | `CRYPTOPANIC_API_KEY` | Financial news API |
| Alpha Vantage | `ALPHA_VANTAGE_API_KEY` | Stock market data |
| Finnhub | `FINNHUB_API_KEY` | Financial data API |
| Twelve Data | `TWELVE_DATA_API_KEY` | Market data API |

## Custom API Keys

For additional API keys not listed above, use the `KIMERA_CUSTOM_API_KEYS` environment variable with a JSON string:

```
KIMERA_CUSTOM_API_KEYS={"service1": "key1", "service2": "key2"}
```

## Accessing API Keys in Code

Use the `ConfigManager` to access API keys in your code:

```python
from src.config.config_manager import ConfigManager

# Get an API key
openai_key = ConfigManager.get_api_key("openai")
custom_key = ConfigManager.get_api_key("service1")  # From custom keys

# Get database URL
db_url = ConfigManager.get_database_url()

# Get other secrets
secret = ConfigManager.get_secret("MY_SECRET")

# Check environment
if ConfigManager.is_production():
    # Use production settings
```

## Troubleshooting

If you're experiencing issues with credentials:

1. Verify your environment variables are set correctly
2. Check that your `.env` file is in the correct location
3. Ensure you're using the correct service name in `get_api_key()`
4. Look for warning messages in the logs about missing API keys

## Security Recommendations

1. Use different API keys for development and production
2. Implement key rotation policies
3. Use the minimum required permissions for each API key
4. Monitor API key usage for unusual activity
5. Consider using a secrets manager for production deployments 