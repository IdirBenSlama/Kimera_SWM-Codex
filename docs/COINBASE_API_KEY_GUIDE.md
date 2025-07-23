# 🔑 Complete Guide: Getting Coinbase Advanced Trade API Keys

## Prerequisites
- Active Coinbase account with completed KYC
- €5+ balance in your account
- 2FA enabled (recommended)

## Step-by-Step Instructions

### 1. Access Coinbase Settings
1. **Login** to your Coinbase account at https://www.coinbase.com
2. Click your **profile icon** (top right)
3. Select **Settings** from the dropdown

### 2. Navigate to API Section
1. In Settings, find **API** in the left sidebar
2. Or directly visit: https://www.coinbase.com/settings/api
3. You'll see "API Keys" section

### 3. Create New API Key
1. Click **"+ New API Key"** button
2. You'll see a permissions screen

### 4. Select Permissions
For trading, you need these permissions:

**Essential Permissions:**
- ✅ `wallet:accounts:read` - View your account balances
- ✅ `wallet:trades:read` - View your trading history
- ✅ `wallet:buys:create` - Create buy orders
- ✅ `wallet:sells:create` - Create sell orders

**Optional (but useful):**
- ✅ `wallet:transactions:read` - View transaction history
- ✅ `wallet:orders:read` - View order status
- ✅ `wallet:deposits:read` - View deposits
- ✅ `wallet:withdrawals:read` - View withdrawals

**DO NOT enable unless needed:**
- ❌ `wallet:withdrawals:create` - Security risk
- ❌ `wallet:accounts:delete` - Dangerous

### 5. Configure Security Settings
1. **Nickname**: Give it a name like "Kimera Trading Bot"
2. **IP Whitelist** (optional but recommended):
   - Add your current IP address
   - You can find it at: https://whatismyipaddress.com
3. **Notification URL**: Leave blank for now

### 6. Create and Download Key
1. Click **"Create"** button
2. **IMPORTANT**: A popup will show:
   - **API Key Name**: Copy this immediately
   - **API Secret**: Click "Download" to save the private key
3. **You will NEVER see the secret again!**

### 7. Save Your Credentials

The downloaded file contains your private key in PEM format:
```
-----BEGIN EC PRIVATE KEY-----
MHcCAQEEIAbcdefghijklmnopqrstuvwxyz1234567890abcdefghijklmnop
qr3oAoGCCqGSM49AwEHoUQDQgAEabcdefghijklmnopqrstuvwxyz1234567890
abcdefghijklmnopqrstuvwxyz1234567890abcdefghijklmnopqrstuvwxyz12
34567890abcdefghijklmnopqr==
-----END EC PRIVATE KEY-----
```

### 8. Update Your .env File

Create or update `.env` in your project root:
```bash
# Coinbase Advanced Trade API
COINBASE_ADVANCED_API_KEY=organizations/12345678-1234-1234-1234-123456789012/apiKeys/87654321-4321-4321-4321-210987654321
COINBASE_ADVANCED_API_SECRET=-----BEGIN EC PRIVATE KEY-----
MHcCAQEEIAbcdefghijklmnopqrstuvwxyz1234567890abcdefghijklmnop
qr3oAoGCCqGSM49AwEHoUQDQgAEabcdefghijklmnopqrstuvwxyz1234567890
abcdefghijklmnopqrstuvwxyz1234567890abcdefghijklmnopqrstuvwxyz12
34567890abcdefghijklmnopqr==
-----END EC PRIVATE KEY-----
```

## Common Issues & Solutions

### "Invalid API Key"
- Make sure you're using Advanced Trade API, not CDP keys
- Check that you copied the full key including `organizations/...`

### "Insufficient Permissions"
- Go back to API settings and edit permissions
- Ensure all required permissions are checked

### "Invalid Signature"
- Private key must include BEGIN/END lines
- No extra spaces or characters
- Must be exact copy from downloaded file

### "IP Not Whitelisted"
- Add your current IP in API settings
- Or disable IP whitelist (less secure)

## Security Best Practices

1. **Never share your API keys**
2. **Use environment variables** (.env file)
3. **Add .env to .gitignore**
4. **Enable IP whitelisting**
5. **Use minimal permissions**
6. **Rotate keys regularly**
7. **Monitor API usage**

## Test Your Keys

Run this command to test:
```bash
python test_advanced_trade_api.py
```

Expected output:
```
✅ Coinbase Advanced Trade SDK imported successfully
✅ API authentication successful
💰 ACCOUNT BALANCES:
   EUR: 5.000000
```

## Ready to Trade!

Once tests pass, run:
```bash
python kimera_omnidimensional_advanced_trade.py
```

## Need Help?

- Coinbase Support: https://help.coinbase.com
- API Documentation: https://docs.cdp.coinbase.com/advanced-trade/docs
- Community Forum: https://forums.coinbase.com

Remember: Start with small amounts until you're comfortable with the system! 