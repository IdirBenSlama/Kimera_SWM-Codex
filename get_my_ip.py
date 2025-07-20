#!/usr/bin/env python3
"""
Get Your Current Public IP Address for Coinbase API Whitelist
"""

import requests

def get_my_ip():
    """Get current public IP address"""
    print("🔍 GETTING YOUR CURRENT IP ADDRESS")
    print("=" * 40)
    
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
                    print(f"✅ Your current public IP: {ip}")
                    print(f"📋 For Coinbase whitelist, use: {ip}")
                    print(f"🔒 Or for IP range, use: {ip}/32")
                    return ip
            except:
                continue
        
        print("❌ Could not determine IP address")
        print("💡 You can:")
        print("   1. Leave whitelist empty (any IP)")
        print("   2. Google 'what is my ip' to find it manually")
        return None
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    get_my_ip() 