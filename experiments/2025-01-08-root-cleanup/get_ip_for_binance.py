#!/usr/bin/env python3
"""
Kimera IP Detection for Binance API Whitelisting
Detects current public IP address and provides whitelisting instructions
"""

import requests
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ip_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_public_ip():
    """Get current public IP address using multiple services for reliability"""
    
    ip_services = [
        "https://api.ipify.org?format=json",
        "https://httpbin.org/ip",
        "https://api.myip.com",
        "https://ipapi.co/json/",
        "https://api.ip.sb/jsonip"
    ]
    
    detected_ips = []
    
    for service in ip_services:
        try:
            logger.info(f"Checking IP with service: {service}")
            response = requests.get(service, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract IP from different response formats
            if 'ip' in data:
                ip = data['ip']
            elif 'origin' in data:
                ip = data['origin']
            else:
                ip = str(data).strip('"')
            
            detected_ips.append(ip)
            logger.info(f"✅ Detected IP: {ip}")
            
        except Exception as e:
            logger.warning(f"❌ Failed to get IP from {service}: {e}")
            continue
    
    return detected_ips

def display_whitelisting_instructions(ips):
    """Display comprehensive IP whitelisting instructions"""
    
    logger.info("\n" + "="*80)
    logger.info("🔒 BINANCE API IP WHITELISTING INSTRUCTIONS")
    logger.info("="*80)
    
    if ips:
        # Get most common IP (in case of multiple detections)
        most_common_ip = max(set(ips), key=ips.count)
        
        logger.info(f"\n📍 YOUR CURRENT PUBLIC IP ADDRESS: {most_common_ip}")
        
        if len(set(ips)) > 1:
            logger.info(f"⚠️  Multiple IPs detected: {list(set(ips))}")
            logger.info("   This might indicate IP rotation or different detection services")
            logger.info("   Use the most consistent IP or consider all IPs for whitelisting")
        
        logger.info(f"\n🎯 IP TO WHITELIST IN BINANCE: {most_common_ip}")
    else:
        logger.info("\n❌ Could not detect public IP address")
        logger.info("   Please check your internet connection and try again")
        return
    
    logger.info("\n📋 STEP-BY-STEP WHITELISTING PROCESS:")
    logger.info("="*50)
    
    steps = [
        "1. Login to Binance.com",
        "2. Go to Account → API Management",
        "3. Find your API Key: Y9WyflPyK1tVXnET3CTMvSdCbPia3Nhtd89VYWjS9RaAbQ0KEhHezkcGSCySQ8cL",
        "4. Click 'Edit restrictions' or 'Edit' button",
        "5. In the 'API restrictions' section:",
        "   ├── ✅ Enable 'Spot & Margin Trading'",
        "   ├── ✅ Enable 'Futures Trading' (if needed)",
        "   └── ✅ Enable 'Read Info'",
        "6. In the 'IP access restrictions' section:",
        f"   ├── Select 'Restrict access to trusted IPs only'",
        f"   ├── Add IP address: {most_common_ip if ips else 'YOUR_IP_HERE'}",
        f"   └── Click 'Confirm'",
        "7. Complete 2FA verification",
        "8. Wait 5-10 minutes for changes to take effect"
    ]
    
    for step in steps:
        logger.info(f"   {step}")
    
    logger.info("\n⚠️  IMPORTANT SECURITY NOTES:")
    logger.info("="*30)
    logger.info("• IP whitelisting significantly improves API security")
    logger.info("• Only add IPs you trust and control")
    logger.info("• If your IP changes frequently, consider using a VPN with static IP")
    logger.info("• Remove old/unused IPs from the whitelist regularly")
    logger.info("• Monitor API usage in Binance dashboard")
    
    logger.info("\n🔄 IF YOUR IP CHANGES:")
    logger.info("="*25)
    logger.info("• Run this script again to detect new IP")
    logger.info("• Update the whitelist in Binance API settings")
    logger.info("• Trading will be blocked until new IP is whitelisted")
    
    logger.info("\n🚀 AFTER WHITELISTING:")
    logger.info("="*25)
    logger.info("• Run: python test_trading_permissions.py")
    logger.info("• Verify all permissions are working")
    logger.info("• Start trading with Kimera autonomous systems")
    
    # Save IP information to file
    ip_info = {
        "timestamp": datetime.now().isoformat(),
        "detected_ips": ips,
        "recommended_ip": most_common_ip if ips else None,
        "all_unique_ips": list(set(ips)) if ips else []
    }
    
    with open('detected_ip_info.json', 'w') as f:
        json.dump(ip_info, f, indent=2)
    
    logger.info(f"\n💾 IP information saved to: detected_ip_info.json")
    logger.info("="*80)

def main():
    """Main execution function"""
    logger.info("🔍 Starting IP detection for Binance API whitelisting...")
    
    try:
        # Detect public IP addresses
        detected_ips = get_public_ip()
        
        # Display whitelisting instructions
        display_whitelisting_instructions(detected_ips)
        
        logger.info("✅ IP detection and instructions completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Error during IP detection: {e}")
        logger.info(f"\n❌ Error: {e}")
        logger.info("Please check your internet connection and try again")

if __name__ == "__main__":
    main() 