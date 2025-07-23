# 🔐 Binance API Setup Guide for Live Trading

## ⚠️ CRITICAL: Enable Trading Permissions

Your current API key has **READ-ONLY** permissions. To enable live trading, follow these steps:

### 📋 Step 1: Access Binance API Management

1. **Login to Binance:** Go to [binance.com](https://binance.com)
2. **Navigate to API:** Account → API Management
3. **Find Your API Key:** `Y9WyflPyK1tVXnET3CTMvSdCbPia3Nhtd89VYWjS9RaAbQ0KEhHezkcGSCySQ8cL`

### 🔧 Step 2: Enable Trading Permissions

**Current Status:** ❌ Read Only
**Required Status:** ✅ Spot & Margin Trading

1. **Click "Edit"** next to your API key
2. **Enable Permissions:**
   - ✅ **Enable Reading** (already enabled)
   - ✅ **Enable Spot & Margin Trading** ← **ENABLE THIS**
   - ❌ **Enable Futures** (not needed for spot trading)
   - ❌ **Enable Withdrawals** (not recommended for trading bots)

### 🌐 Step 3: IP Whitelist Configuration

**Current Issue:** API calls from your IP may be blocked

**Option A: Unrestricted Access (Easier)**
1. **Remove IP Restrictions:** Leave IP whitelist empty
2. **Higher Security Risk:** Anyone with your keys can trade

**Option B: IP Whitelist (Recommended)**
1. **Add Your Current IP:** Check your IP at [whatismyipaddress.com](https://whatismyipaddress.com)
2. **Add IP to Whitelist:** Enter your IP in the API settings
3. **Update When IP Changes:** Remember to update if your IP changes

### 🔐 Step 4: Security Best Practices

1. **API Key Security:**
   - Never share your API keys
   - Store in secure environment files
   - Use IP restrictions when possible
   - Monitor API usage regularly

2. **Trading Limits:**
   - Start with small position sizes
   - Set daily trading limits
   - Monitor all trades closely
   - Keep emergency stop procedures ready

### 🧪 Step 5: Test Configuration

After enabling permissions, test with this command:

```bash
python test_api_permissions.py
```

Expected output:
```
✅ API Key: Valid
✅ Permissions: Spot Trading Enabled
✅ IP Access: Allowed
✅ Ready for Live Trading
```

### 🚨 Step 6: Emergency Procedures

**If Something Goes Wrong:**

1. **Immediate Actions:**
   - Disable API key in Binance settings
   - Run emergency stop script: `python emergency_stop.py`
   - Check all open positions

2. **Emergency Contacts:**
   - Binance Support: [support.binance.com](https://support.binance.com)
   - Emergency Stop: Press `Ctrl+C` in trading terminal

### ✅ Verification Checklist

Before starting live trading:

- [ ] Spot & Margin Trading enabled
- [ ] IP whitelist configured (or unrestricted)
- [ ] API permissions tested successfully
- [ ] Emergency procedures understood
- [ ] Small test trade executed successfully
- [ ] Position size limits configured
- [ ] Stop loss mechanisms tested

### 🎯 Ready to Trade

Once all steps are complete, you can run:

```bash
python kimera_profit_maximizer_v2.py
```

This will start the enhanced profit maximizer with full trading capabilities.

---

## 📞 Support

If you encounter issues:

1. **Check API Status:** Binance API status page
2. **Verify Permissions:** Re-check all settings above
3. **Test Connection:** Run diagnostic scripts
4. **Contact Support:** Binance customer service

**Remember:** Always start with small amounts and gradually increase as you gain confidence in the system. 