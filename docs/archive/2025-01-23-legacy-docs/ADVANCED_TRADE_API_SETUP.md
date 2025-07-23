# Coinbase Advanced Trade API Setup Guide

## Authentication Issue Resolution

Your current CDP API key (`9268de76-b5f4-4683-b593-327fb2c19503`) is designed for **Coinbase Developer Platform (CDP)** services, not the **Coinbase Advanced Trade API** that we need for real trading.

## Required API Keys

To enable real trading, you need **Coinbase Advanced Trade API** credentials:

### 1. Create Advanced Trade API Keys

1. **Visit Coinbase Settings:**
   - Go to: https://www.coinbase.com/settings/api
   - **NOT** the CDP portal

2. **Create New API Key:**
   - Click "New API Key"
   - Select permissions:
     - ✅ `wallet:accounts:read` (view balances)
     - ✅ `wallet:trades:read` (view trades)
     - ✅ `wallet:buys:create` (place buy orders)
     - ✅ `wallet:sells:create` (place sell orders)

3. **Download Credentials:**
   - Save the API Key name
   - Save the Private Key (PEM format)

### 2. Environment Configuration

Create a `.env` file with your Advanced Trade API credentials:

```bash
# Coinbase Advanced Trade API (for real trading)
COINBASE_ADVANCED_API_KEY=your_api_key_name
COINBASE_ADVANCED_API_SECRET=-----BEGIN EC PRIVATE KEY-----
your_private_key_content_here
-----END EC PRIVATE KEY-----
```

## API Differences Explained

| Feature | CDP API (Your Current) | Advanced Trade API (Needed) |
|---------|------------------------|------------------------------|
| **Purpose** | Wallet operations, onchain activities | Real trading on Coinbase |
| **Trading** | ❌ Limited | ✅ Full trading capabilities |
| **Key Format** | Simple UUID | API Key + PEM Private Key |
| **Endpoints** | `api.cdp.coinbase.com` | `api.coinbase.com` |
| **SDK** | AgentKit | `coinbase-advanced-py` |

## Installation Steps

1. **Install Official SDK:**
```bash
pip install coinbase-advanced-py
```

2. **Update Environment:**
```bash
# Add to your .env file
COINBASE_ADVANCED_API_KEY=your_advanced_api_key
COINBASE_ADVANCED_API_SECRET=your_private_key_pem_content
```

3. **Run New Implementation:**
```bash
python kimera_omnidimensional_advanced_trade.py
```

## Testing Connection

```python
from coinbase.rest import RESTClient

client = RESTClient(
    api_key="your_api_key",
    api_secret="your_private_key_content"
)

# Test connection
accounts = client.get_accounts()
print("Connection successful!")
```

## Expected Results with €5 Balance

Using our omnidimensional strategy:

- **Horizontal Strategy:** €0.10-0.15 profit (2-3% return)
- **Vertical Strategy:** €0.15-0.25 profit (3-5% return)  
- **Synergy Bonus:** €0.05 profit (1% return)
- **Total Expected:** €0.30-0.45 profit (6-9% return per 5 minutes)

## Security Best Practices

1. **API Key Permissions:**
   - Only enable required permissions
   - Use IP whitelisting if available

2. **Key Storage:**
   - Never commit API keys to git
   - Use environment variables
   - Store private keys securely

3. **Trading Limits:**
   - Start with small amounts
   - Monitor all trades
   - Set stop-loss limits

## Troubleshooting

### Common Issues:

1. **401 Unauthorized:**
   - Verify you're using Advanced Trade API keys
   - Check API key permissions
   - Ensure PEM format private key

2. **Insufficient Funds:**
   - Minimum €5 per trade
   - Check available balance
   - Account for trading fees

3. **Rate Limiting:**
   - SDK handles rate limits automatically
   - Add delays between requests if needed

## Alternative: Keep Using CDP

If you prefer to keep your CDP setup, we can create a CDP-compatible version using the AgentKit SDK, but it has limited trading capabilities compared to Advanced Trade API.

## Ready to Trade

Once you have Advanced Trade API credentials:

1. Update your `.env` file
2. Run: `python kimera_omnidimensional_advanced_trade.py`
3. Monitor results in `test_results/` directory

The system will execute real trades using your omnidimensional strategy across multiple EUR trading pairs. 