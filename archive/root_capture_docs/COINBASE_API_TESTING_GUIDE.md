# COINBASE PRO API TESTING GUIDE

## 🎯 OBJECTIVE
Test your Coinbase Pro API connection and check available funds for autonomous trading deployment.

---

## 🚀 QUICK START

### Option 1: Simple Test (Recommended)
```bash
python simple_coinbase_test.py
```

### Option 2: Comprehensive Test
```bash
python test_coinbase_api.py
```

---

## 🔧 SETUP REQUIREMENTS

### 1. Coinbase Pro API Credentials
You need to create API credentials in your Coinbase Pro account:

1. **Login to Coinbase Pro**: https://pro.coinbase.com
2. **Navigate to**: Settings → API
3. **Create New API Key** with permissions:
   - ✅ **View** (required for checking balances)
   - ✅ **Trade** (required for autonomous trading)
   - ⚠️ **Transfer** (optional, not needed for trading)

### 2. API Key Information
When you create the API key, you'll get:
- **API Key**: Public identifier
- **API Secret**: Private key for signing requests
- **Passphrase**: Additional security phrase

### 3. Environment Variables (Optional)
You can set environment variables to avoid entering credentials each time:
```bash
export COINBASE_API_KEY="your_api_key_here"
export COINBASE_API_SECRET="your_api_secret_here"
export COINBASE_PASSPHRASE="your_passphrase_here"
```

---

## 🧪 TESTING ENVIRONMENTS

### Sandbox Environment (Recommended for Testing)
- **URL**: https://api-public.sandbox.pro.coinbase.com
- **Purpose**: Test API integration without real money
- **Funds**: Simulated balances for testing
- **Safety**: No real money at risk

### Live Environment (Real Money)
- **URL**: https://api.pro.coinbase.com
- **Purpose**: Actual trading with real funds
- **Funds**: Your actual account balances
- **Safety**: Real money trading - use with caution

---

## 📊 WHAT THE TEST CHECKS

### 1. API Connection
- ✅ Server connectivity
- ✅ Authentication verification
- ✅ Account access permissions

### 2. Available Funds
- 💰 USD balances
- 🪙 Cryptocurrency balances (BTC, ETH, LTC, BCH)
- 💵 Total USD equivalent value
- 📊 Number of accounts with balances

### 3. Trading Pairs
- 📈 Available major trading pairs
- ✅ BTC-USD, ETH-USD, LTC-USD, BCH-USD
- 🔄 Trading pair status (online/offline)

### 4. Trading Readiness Assessment
- 🎯 Capital adequacy analysis
- 📊 Risk level assessment
- 💡 Recommended trading session sizes
- ✅ Overall readiness score

---

## 💰 FUND REQUIREMENTS

### Trading Readiness Levels

#### 🏆 EXCELLENT ($1000+)
- **Status**: Ready for full autonomous trading
- **Recommended**: $100-500 per trading session
- **Risk Level**: Low
- **Strategy**: Aggressive optimization possible

#### ✅ GOOD ($500-999)
- **Status**: Ready for moderate autonomous trading
- **Recommended**: $50-200 per trading session
- **Risk Level**: Moderate
- **Strategy**: Conservative to moderate

#### ✅ ADEQUATE ($100-499)
- **Status**: Ready for conservative trading
- **Recommended**: $25-100 per trading session
- **Risk Level**: Moderate-High
- **Strategy**: Conservative only

#### ⚠️ MINIMAL ($25-99)
- **Status**: Use extreme caution
- **Recommended**: $5-25 per trading session
- **Risk Level**: High
- **Strategy**: Micro-trading only

#### ❌ INSUFFICIENT (<$25)
- **Status**: Not recommended for autonomous trading
- **Recommended**: Add more funds before trading
- **Risk Level**: Very High
- **Strategy**: Manual trading only

---

## 🔍 SAMPLE TEST OUTPUT

```
COINBASE PRO API FUNDS TEST
==================================================
🧪 Using SANDBOX environment (test mode)

🔗 Testing API connection...
✅ Server connection successful
✅ API authentication successful
   Found 8 accounts

💰 Checking available funds...

📊 Account Balances:
--------------------------------------------------
💵 USD: $1500.00 available
🪙 BTC: 0.05000000 available
     = $2100.00 USD (@ $42000.00)
🪙 ETH: 1.20000000 available
     = $3600.00 USD (@ $3000.00)
🪙 LTC: 10.00000000 available
     = $800.00 USD (@ $80.00)
--------------------------------------------------
💰 TOTAL USD VALUE: $8000.00
📊 Accounts with balance: 4

🎯 TRADING READINESS ASSESSMENT:
✅ EXCELLENT - Ready for full autonomous trading
   Recommended: $100-500 per trading session

📈 Checking trading pairs...
✅ Available major pairs: BTC-USD, ETH-USD, LTC-USD, BCH-USD

🏆 TEST COMPLETE
==================================================
Connection: ✅ SUCCESS
Total USD Value: $8000.00
Trading Pairs: 4 available
Status: ✅ READY FOR AUTONOMOUS TRADING
==================================================
```

---

## 🚨 SECURITY CONSIDERATIONS

### API Key Security
- 🔒 **Never share** your API credentials
- 🔒 **Use environment variables** when possible
- 🔒 **Limit permissions** to only what's needed
- 🔒 **Rotate keys regularly** for security

### Testing Best Practices
- 🧪 **Always test in sandbox first**
- 🧪 **Verify credentials work** before live trading
- 🧪 **Start with small amounts** in live environment
- 🧪 **Monitor closely** during initial runs

### Risk Management
- 💰 **Never risk more** than you can afford to lose
- 💰 **Start conservative** and scale up gradually
- 💰 **Use stop-losses** and position limits
- 💰 **Monitor performance** continuously

---

## 🔧 TROUBLESHOOTING

### Common Issues

#### Authentication Errors
```
❌ Connection failed: 401 Unauthorized
```
**Solution**: Check API key, secret, and passphrase are correct

#### Permission Errors
```
❌ Connection failed: 403 Forbidden
```
**Solution**: Ensure API key has "View" and "Trade" permissions

#### Network Errors
```
❌ Connection failed: Connection timeout
```
**Solution**: Check internet connection and try again

#### Invalid Credentials
```
❌ Connection failed: Invalid passphrase
```
**Solution**: Verify passphrase matches exactly (case-sensitive)

### Getting Help
1. **Check Coinbase Pro API documentation**: https://docs.pro.coinbase.com/
2. **Verify API key permissions** in Coinbase Pro settings
3. **Test with sandbox environment** first
4. **Contact Coinbase support** for account-specific issues

---

## 🚀 NEXT STEPS AFTER SUCCESSFUL TEST

### If Test Passes ✅
1. **Note your total USD value** from the test results
2. **Choose appropriate trading session size** based on assessment
3. **Proceed to autonomous trading deployment**:
   ```bash
   python run_kimera_with_autonomous_trading.py
   ```
4. **Enable trading** and enter your API credentials
5. **Start with conservative parameters** for first session

### If Test Fails ❌
1. **Review error messages** and troubleshoot
2. **Verify API credentials** are correct
3. **Check API permissions** in Coinbase Pro
4. **Try sandbox environment** first
5. **Contact support** if issues persist

---

## 📊 INTEGRATION WITH KIMERA

Once your API test passes, you can integrate with the full Kimera system:

### Deployment Options
1. **Server + Trading**: Full integrated system
2. **Trading Only**: Standalone autonomous trading
3. **Server Only**: Cognitive capabilities without trading

### Configuration
- **Professional Parameters**: 8% max position, 6 trades/day
- **Conservative Risk Management**: <5% drawdown protection
- **Cognitive Enhancement**: AI-powered market analysis
- **Real-time Monitoring**: Complete system oversight

---

**🎯 MISSION: Verify your Coinbase Pro API and available funds for autonomous trading deployment**

**✅ RESULT: Ready to deploy Kimera autonomous trading with real-world execution capability** 