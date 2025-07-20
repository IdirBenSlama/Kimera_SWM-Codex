# KIMERA API SETUP GUIDE
## Get Your Trading API Ready in 5 Minutes

### 🎯 **RECOMMENDED: Coinbase Advanced Trading API**

#### **Step 1: Create Account**
1. Go to: https://cloud.coinbase.com/
2. Sign up or log in with your Coinbase account
3. Verify your identity if needed

#### **Step 2: Create API Credentials**
1. Navigate to: https://cloud.coinbase.com/access/api
2. Click "Create API Key"
3. Select "Advanced Trading" 
4. Choose permissions:
   - ✅ View accounts
   - ✅ Trade
   - ✅ View orders
5. Set IP whitelist (optional but recommended)

#### **Step 3: Download Credentials**
You'll get:
- **API Key** (UUID format)
- **API Secret** (long string)
- **Passphrase** (you set this)
- **Private Key** (PEM file - DOWNLOAD THIS!)

#### **Step 4: Save Credentials**
Create file: `coinbase_credentials.json`
```json
{
  "api_key": "your-api-key-here",
  "api_secret": "your-api-secret-here", 
  "passphrase": "your-passphrase-here"
}
```

#### **Step 5: Test with Kimera**
```bash
python test_coinbase_advanced.py
```

---

### 🔄 **ALTERNATIVE: Binance API (If you prefer)**

#### **Step 1: Create Binance Account**
1. Go to: https://www.binance.com/
2. Complete KYC verification

#### **Step 2: Create API Key**
1. Go to: Account → API Management
2. Create new API key
3. Enable "Spot & Margin Trading"
4. Set IP whitelist (recommended)

#### **Step 3: Save Credentials**
```json
{
  "api_key": "your-binance-api-key",
  "secret_key": "your-binance-secret-key"
}
```

---

### ⚡ **IMMEDIATE SETUP COMMANDS**

Once you have credentials, run:

```bash
# Test your API
python test_your_api.py

# Run Kimera with live trading
python kimera_live_trading.py

# Start with small amount
python kimera_start_trading.py --capital=100
```

---

### 💰 **RECOMMENDED STARTING AMOUNTS**

- **Testing**: $25-50
- **Initial Live**: $100-200  
- **Confident**: $500-1000
- **Scaling**: $2000+

---

### 🛡️ **SECURITY TIPS**

1. **Never share your API credentials**
2. **Use IP whitelisting**
3. **Start with small amounts**
4. **Enable 2FA on exchange account**
5. **Set trading limits in API settings**

---

### 🎯 **WHAT HAPPENS NEXT**

1. You provide API credentials
2. I integrate with Kimera (5 minutes)
3. We test with small amount ($25-50)
4. Scale up based on performance
5. Full autonomous operation

**The Kimera trading engine is ready and waiting for your API!** 