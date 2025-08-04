
# Security Scan Report
Generated: 2025-07-09T01:58:47.286162

## Summary
- Total findings: 29
- High severity: 0
- Medium severity: 29
- Files with issues: 16

## Findings

### MEDIUM - backend\security\authentication.py:24
```
PASSWORD = "password"
```
Pattern: (?i)password\s*=\s*["\']([^"\']+)["\']

### MEDIUM - backend\security\authentication.py:25
```
TOKEN = "token"
```
Pattern: (?i)token\s*=\s*["\']([^"\']+)["\']

### MEDIUM - archive\broken_scripts_and_tests\test_trading_system.py:225
```
api_key="test_key"
```
Pattern: (?i)api_key\s*=\s*["\']([^"\']+)["\']

### MEDIUM - archive\broken_scripts_and_tests\test_trading_system.py:226
```
secret="test_secret"
```
Pattern: (?i)secret\s*=\s*["\']([^"\']+)["\']

### MEDIUM - archive\broken_scripts_and_tests\demonstration\coinbase_with_passphrase_demo.py:18
```
API_KEY = "9268de76-b5f4-4683-b593-327fb2c19503"
```
Pattern: (?i)api_key\s*=\s*["\']([^"\']+)["\']

### MEDIUM - archive\broken_scripts_and_tests\demonstration\coinbase_with_passphrase_demo.py:19
```
SECRET = "BiCUFOxZ4J4Fi8F6mcyzuzreXaGZeBLHxr7q8Puo6VHcSEgyqJ6mIx29RbbAJGAjq6SHBt5K4PieiymRhEWVHw=="
```
Pattern: (?i)secret\s*=\s*["\']([^"\']+)["\']

### MEDIUM - archive\broken_scripts_and_tests\development\extreme_fortress_breaker.py:106
```
password='pwned'
```
Pattern: (?i)password\s*=\s*["\']([^"\']+)["\']

### MEDIUM - archive\broken_scripts_and_tests\development\fixed_extreme_fortress_breaker.py:100
```
password='pwned'
```
Pattern: (?i)password\s*=\s*["\']([^"\']+)["\']

### MEDIUM - archive\broken_scripts_and_tests\development\kimera_agentkit_integration.py:469
```
API_KEY = "your_cdp_api_key_here"
```
Pattern: (?i)api_key\s*=\s*["\']([^"\']+)["\']

### MEDIUM - archive\broken_scripts_and_tests\development\kimera_agentkit_integration.py:470
```
SECRET = "your_cdp_api_secret_here"
```
Pattern: (?i)secret\s*=\s*["\']([^"\']+)["\']

### MEDIUM - archive\broken_scripts_and_tests\development\kimera_agentkit_integration.py:471
```
API_KEY = "your_openai_api_key_here"
```
Pattern: (?i)api_key\s*=\s*["\']([^"\']+)["\']

### MEDIUM - archive\broken_scripts_and_tests\development\validate_vault_security_system.py:137
```
password = "KIMERA_SECURE_PASSWORD_2025"
```
Pattern: (?i)password\s*=\s*["\']([^"\']+)["\']

### MEDIUM - archive\broken_scripts_and_tests\testing\quick_coinbase_test.py:14
```
API_KEY = "9268de76-b5f4-4683-b593-327fb2c19503"
```
Pattern: (?i)api_key\s*=\s*["\']([^"\']+)["\']

### MEDIUM - archive\broken_scripts_and_tests\testing\quick_coinbase_test.py:15
```
SECRET = "BiCUFOxZ4J4Fi8F6mcyzuzreXaGZeBLHxr7q8Puo6VHcSEgyqJ6mIx29RbbAJGAjq6SHBt5K4PieiymRhEWVHw=="
```
Pattern: (?i)secret\s*=\s*["\']([^"\']+)["\']

### MEDIUM - archive\broken_scripts_and_tests\testing\simple_coinbase_test.py:12
```
API_KEY = "9268de76-b5f4-4683-b593-327fb2c19503"
```
Pattern: (?i)api_key\s*=\s*["\']([^"\']+)["\']

### MEDIUM - archive\broken_scripts_and_tests\testing\simple_coinbase_test.py:13
```
SECRET = "BiCUFOxZ4J4Fi8F6mcyzuzreXaGZeBLHxr7q8Puo6VHcSEgyqJ6mIx29RbbAJGAjq6SHBt5K4PieiymRhEWVHw=="
```
Pattern: (?i)secret\s*=\s*["\']([^"\']+)["\']

### MEDIUM - archive\broken_scripts_and_tests\testing\test_coinbase_api_connection.py:148
```
API_KEY = "9268de76-b5f4-4683-b593-327fb2c19503"
```
Pattern: (?i)api_key\s*=\s*["\']([^"\']+)["\']

### MEDIUM - archive\broken_scripts_and_tests\testing\test_coinbase_api_connection.py:149
```
SECRET = "BiCUFOxZ4J4Fi8F6mcyzuzreXaGZeBLHxr7q8Puo6VHcSEgyqJ6mIx29RbbAJGAjq6SHBt5K4PieiymRhEWVHw=="
```
Pattern: (?i)secret\s*=\s*["\']([^"\']+)["\']

### MEDIUM - archive\broken_scripts_and_tests\testing\test_modern_coinbase_auth.py:161
```
API_KEY = "9268de76-b5f4-4683-b593-327fb2c19503"
```
Pattern: (?i)api_key\s*=\s*["\']([^"\']+)["\']

### MEDIUM - archive\broken_scripts_and_tests\testing\test_modern_coinbase_auth.py:162
```
SECRET = "BiCUFOxZ4J4Fi8F6mcyzuzreXaGZeBLHxr7q8Puo6VHcSEgyqJ6mIx29RbbAJGAjq6SHBt5K4PieiymRhEWVHw=="
```
Pattern: (?i)secret\s*=\s*["\']([^"\']+)["\']

### MEDIUM - archive\broken_scripts_and_tests\trading\kimera_coinbase_advanced_integration.py:478
```
API_KEY = "9268de76-b5f4-4683-b593-327fb2c19503"
```
Pattern: (?i)api_key\s*=\s*["\']([^"\']+)["\']

### MEDIUM - archive\broken_scripts_and_tests\trading\kimera_coinbase_advanced_integration.py:479
```
SECRET = "BiCUFOxZ4J4Fi8F6mcyzuzreXaGZeBLHxr7q8Puo6VHcSEgyqJ6mIx29RbbAJGAjq6SHBt5K4PieiymRhEWVHw=="
```
Pattern: (?i)secret\s*=\s*["\']([^"\']+)["\']

### MEDIUM - archive\broken_scripts_and_tests\trading\kimera_live_coinbase_trader.py:419
```
API_KEY = "9268de76-b5f4-4683-b593-327fb2c19503"
```
Pattern: (?i)api_key\s*=\s*["\']([^"\']+)["\']

### MEDIUM - archive\broken_scripts_and_tests\trading\kimera_live_coinbase_trader.py:420
```
SECRET = "BiCUFOxZ4J4Fi8F6mcyzuzreXaGZeBLHxr7q8Puo6VHcSEgyqJ6mIx29RbbAJGAjq6SHBt5K4PieiymRhEWVHw=="
```
Pattern: (?i)secret\s*=\s*["\']([^"\']+)["\']

### MEDIUM - archive\broken_scripts_and_tests\trading\kimera_modern_coinbase_trader.py:497
```
API_KEY = "9268de76-b5f4-4683-b593-327fb2c19503"
```
Pattern: (?i)api_key\s*=\s*["\']([^"\']+)["\']

### MEDIUM - archive\broken_scripts_and_tests\trading\kimera_modern_coinbase_trader.py:498
```
SECRET = "BiCUFOxZ4J4Fi8F6mcyzuzreXaGZeBLHxr7q8Puo6VHcSEgyqJ6mIx29RbbAJGAjq6SHBt5K4PieiymRhEWVHw=="
```
Pattern: (?i)secret\s*=\s*["\']([^"\']+)["\']

### MEDIUM - archive\broken_scripts_and_tests\trading\real_coinbase_cdp_connector.py:88
```
secret = "BiCUFOxZ4J4Fi8F6mcyzuzreXaGZeBLHxr7q8Puo6VHcSEgyqJ6mIx29RbbAJGAjq6SHBt5K4PieiymRhEWVHw=="
```
Pattern: (?i)secret\s*=\s*["\']([^"\']+)["\']

### MEDIUM - archive\broken_scripts_and_tests\trading\real_coinbase_pro_trader.py:458
```
API_KEY = "f7360d36-8068-4b75-8169-6d016b96d810"
```
Pattern: (?i)api_key\s*=\s*["\']([^"\']+)["\']

### MEDIUM - archive\broken_scripts_and_tests\trading\real_coinbase_pro_trader.py:459
```
SECRET = "BiCUFOxZ4J4Fi8F6mcyzuzreXaGZeBLHxr7q8Puo6VHcSEgyqJ6mIx29RbbAJGAjq6SHBt5K4PieiymRhEWVHw=="
```
Pattern: (?i)secret\s*=\s*["\']([^"\']+)["\']
