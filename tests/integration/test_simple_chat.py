#!/usr/bin/env python3
"""Simple test to check KIMERA chat endpoint."""

import requests
import json

def test_chat():
    """Test basic chat functionality."""
    
    url = "http://localhost:8000/kimera/api/chat/"
    payload = {
        "message": "Hello, KIMERA!",
        "cognitive_mode": "standard",
        "session_id": "test_session"
    }
    
    print(f"Testing: {url}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        print(f"\nStatus Code: {response.status_code}")
        print(f"Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"\nResponse: {json.dumps(result, indent=2)}")
        else:
            print(f"\nError Response: {response.text}")
            
    except Exception as e:
        print(f"\nException: {e}")

if __name__ == "__main__":
    test_chat() 