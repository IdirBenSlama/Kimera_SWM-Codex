# Modern Coinbase API Setup Guide

## 🔐 No Passphrase Required - Modern Authentication Method

This guide explains how to set up and use the **modern Coinbase Advanced Trade API** which eliminates the need for a passphrase and uses secure JWT authentication instead.

## 📋 Table of Contents

1. [Understanding the Modern API](#understanding-the-modern-api)
2. [Authentication Method](#authentication-method)
3. [Required Dependencies](#required-dependencies)
4. [API Credentials Setup](#api-credentials-setup)
5. [Testing Authentication](#testing-authentication)
6. [Running Kimera Live Trading](#running-kimera-live-trading)
7. [Troubleshooting](#troubleshooting)

---

## 🔍 Understanding the Modern API

### Legacy vs Modern Authentication

| Feature | Legacy Coinbase Pro API | Modern Coinbase Advanced Trade API |
|---------|------------------------|-----------------------------------|
| **Passphrase** | ❌ Required | ✅ Not Required |
| **Authentication** | 3-component (Key + Secret + Passphrase) | 2-component (Key + Secret) |
| **Token Type** | HMAC-SHA256 | JWT (JSON Web Token) |
| **Security** | Good | Enhanced |
| **Status** | Deprecated (Feb 2025) | Active |

### Why No Passphrase?

The modern Coinbase API uses **JWT (JSON Web Token) authentication** which:
- Eliminates the need for a static passphrase
- Provides enhanced security through time-limited tokens
- Simplifies the authentication process
- Reduces the risk of credential exposure

---

## 🔑 Authentication Method

### JWT Token Structure

The modern API generates JWT tokens with the following payload:

```json
{
  "sub": "your-api-key",
  "iss": "coinbase-cloud",
  "nbf": 1234567890,
  "exp": 1234567890,
  "aud": ["public_websocket_api"],
  "uri": "GET https://api.coinbase.com/api/v3/brokerage/accounts"
}
```

### Authentication Headers

```http
Authorization: Bearer <JWT_TOKEN>
Content-Type: application/json
User-Agent: Your-App-Name/1.0
```

---

## 📦 Required Dependencies

Install the necessary Python packages:

```bash
# Core dependencies
pip install requests
pip install PyJWT
pip install cryptography
pip install numpy

# Optional for advanced features
pip install asyncio
pip install aiohttp
```

Or install all at once:

```bash
pip install requests PyJWT cryptography numpy asyncio aiohttp
```

---

## 🔧 API Credentials Setup

### Step 1: Create Coinbase Developer Account

1. Go to [Coinbase Developer Platform](https://portal.cdp.coinbase.com)
2. Sign in with your Coinbase account
3. Create a new project or select existing one

### Step 2: Generate API Keys

1. Click **"Create API Key"**
2. Select **"Trading Key"** (not CDP key)
3. Configure permissions:
   - ✅ **View** (required for account info)
   - ✅ **Trade** (required for trading)
   - ⚠️ **Transfer** (optional, use with caution)

### Step 3: Configure Security

1. **IP Whitelisting**: Add your IP address for enhanced security
2. **Scopes**: Select only necessary permissions
3. **Expiration**: Set appropriate expiration date

### Step 4: Download Credentials

You'll receive:
- **API Key**: Format `organizations/{org_id}/apiKeys/{key_id}`
- **API Secret**: Private key in base64 format

**Important**: No passphrase is generated or required!

---

## 🧪 Testing Authentication

### Quick Test Script

Use our test script to verify your credentials:

```bash
python test_modern_coinbase_auth.py
```

### Expected Output

```
🧪 MODERN COINBASE API AUTHENTICATION TEST
============================================================
🔐 JWT Authentication (No Passphrase Required)
⚡ Testing Modern Coinbase Advanced Trade API

🔑 API Credentials:
   Key: 9268de76...2c19503
   Secret: ********************
   Passphrase: NOT REQUIRED (Modern API)

🔧 Testing Modern API Connection...
🔗 Making GET request to: /api/v3/brokerage/accounts
🔑 Using JWT authentication (no passphrase)
📡 Response status: 200
✅ Modern Coinbase API connection successful!

🎉 ALL TESTS PASSED!
✅ Modern Coinbase API authentication working perfectly
🔐 JWT authentication confirmed (no passphrase needed)
🚀 Ready for live trading with Kimera system
```

---

## 🚀 Running Kimera Live Trading

### Step 1: Update Credentials

Edit the credentials in `kimera_modern_coinbase_trader.py`:

```python
# Your modern API credentials (no passphrase needed)
API_KEY = "your-api-key-here"
API_SECRET = "your-api-secret-here"
```

### Step 2: Choose Environment

```bash
python kimera_modern_coinbase_trader.py
```

Select environment:
- **1**: 🧪 Sandbox (safe testing)
- **2**: 💰 Production (real money)

### Step 3: Start Trading

For production trading, type `TRADE LIVE` when prompted.

### Sample Trading Session

```
🚀 KIMERA MODERN COINBASE TRADER
==================================================
💰 Real Money Trading with Modern API
🧠 Advanced Cognitive AI Analysis
⚡ JWT Authentication (No Passphrase Required)

🔑 Modern API Credentials:
   API Key: 9268de76...2c19503
   Authentication: JWT (No passphrase required)

🚀 STARTING MODERN TRADING SESSION
   Environment: PRODUCTION
   API: Modern Coinbase Advanced Trade
   Starting Balance: $1,250.00
   Max Daily Trades: 25

🔄 COGNITIVE TRADING CYCLE #1
🧠 COGNITIVE ANALYSIS:
   Confidence: 0.847
   Quantum Coherence: 0.923
   Signal: strong_buy
   Optimal Position: $47.50

✅ BUY executed: $47.50 of BTC-USD
📈 Cognitive Trade #1 executed successfully
```

---

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. "JWT generation error"

**Problem**: Error generating JWT token
**Solution**: 
- Verify API secret is correct
- Install `PyJWT` and `cryptography` packages
- Check API key format

```bash
pip install PyJWT cryptography
```

#### 2. "HTTP 401 Unauthorized"

**Problem**: Authentication failed
**Solutions**:
- Verify API key and secret are correct
- Check API key permissions (View + Trade)
- Ensure IP whitelisting is configured correctly
- Confirm API key hasn't expired

#### 3. "HTTP 403 Forbidden"

**Problem**: Insufficient permissions
**Solutions**:
- Enable "Trade" permission in API key settings
- Check account verification status
- Verify sufficient balance for trading

#### 4. "Connection timeout"

**Problem**: Network connectivity issues
**Solutions**:
- Check internet connection
- Verify firewall settings
- Try different network if using VPN

### Authentication Debugging

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### API Status Check

Check Coinbase API status:
- [Coinbase Status Page](https://status.coinbase.com/)
- [API Documentation](https://docs.cloud.coinbase.com/advanced-trade-api/docs/welcome)

---

## 📊 Performance Optimization

### Best Practices

1. **Token Caching**: JWT tokens are valid for 2 minutes
2. **Request Batching**: Combine multiple API calls when possible
3. **Rate Limiting**: Respect API rate limits (10 requests/second)
4. **Error Handling**: Implement robust error handling and retries

### Security Recommendations

1. **Environment Variables**: Store credentials as environment variables
2. **IP Whitelisting**: Enable IP restrictions
3. **Minimal Permissions**: Only grant necessary permissions
4. **Regular Rotation**: Rotate API keys periodically
5. **Monitoring**: Monitor API usage and account activity

---

## 🔗 Additional Resources

### Official Documentation
- [Coinbase Advanced Trade API](https://docs.cloud.coinbase.com/advanced-trade-api/docs/welcome)
- [Authentication Guide](https://docs.cloud.coinbase.com/advanced-trade-api/docs/auth)
- [API Reference](https://docs.cloud.coinbase.com/advanced-trade-api/reference)

### Kimera Documentation
- [Trading Module Documentation](docs/02_User_Guides/crypto_trading_guide.md)
- [API Integration Guide](docs/02_User_Guides/api/)
- [Cognitive Field Metrics](docs/02_User_Guides/cognitive_field_metrics.md)

---

## 🎯 Summary

The modern Coinbase Advanced Trade API:
- ✅ **No passphrase required**
- ✅ **Enhanced JWT security**
- ✅ **Simplified authentication**
- ✅ **Better performance**
- ✅ **Future-proof design**

Your Kimera trading system is now ready for live trading with the most secure and modern authentication method available!

---

*Last updated: January 2025*
*Kimera SWM Alpha Prototype V0.1* 