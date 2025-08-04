#!/usr/bin/env python3
"""
Get Your Current Public IP Address for Coinbase API Whitelist
"""

import requests
import logging
logger = logging.getLogger(__name__)

def get_my_ip():
    """Get current public IP address"""
    logger.info("🔍 GETTING YOUR CURRENT IP ADDRESS")
    logger.info("=" * 40)
    
    try:
        # Try multiple IP services
        ip_services = [
            "https://api.ipify.org",
            "https://ipv4.icanhazip.com",
            "https://api.my-ip.io/ip",
            "https://checkip.amazonaws.com"
        ]
        
        for service in ip_services:
            try:
                response = requests.get(service, timeout=5)
                if response.status_code == 200:
                    ip = response.text.strip()
                    logger.info(f"✅ Your current public IP: {ip}")
                    logger.info(f"📋 For Coinbase whitelist, use: {ip}")
                    logger.info(f"🔒 Or for IP range, use: {ip}/32")
                    return ip
            except Exception as e:
                logger.error(f"Error in get_my_ip.py: {e}", exc_info=True)
                raise  # Re-raise for proper error handling
                continue
        
        logger.info("❌ Could not determine IP address")
        logger.info("💡 You can:")
        logger.info("   1. Leave whitelist empty (any IP)")
        logger.info("   2. Google 'what is my ip' to find it manually")
        return None
        
    except Exception as e:
        logger.info(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    get_my_ip() 