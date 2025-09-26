#!/usr/bin/env python3
"""
Simple test script to verify the integration between frontend and backend
"""

import requests
import json

def test_backend_api():
    """Test the backend API endpoints"""
    base_url = "http://localhost:5000/api"
    
    print("Testing Cosmic Weather Insurance API...")
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health")
        print(f"Health check: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"Health check failed: {e}")
        return False
    
    # Test satellite operators endpoint
    try:
        response = requests.get(f"{base_url}/satellites/operators")
        if response.status_code == 200:
            operators = response.json()
            print(f"Satellite operators: {len(operators.get('operators', []))} operators found")
        else:
            print(f"Satellite operators failed: {response.status_code}")
    except Exception as e:
        print(f"Satellite operators failed: {e}")
    
    # Test database stats endpoint
    try:
        response = requests.get(f"{base_url}/database/stats")
        if response.status_code == 200:
            stats = response.json()
            print(f"Database stats: {stats.get('total_satellites', 0)} satellites")
        else:
            print(f"Database stats failed: {response.status_code}")
    except Exception as e:
        print(f"Database stats failed: {e}")
    
    # Test satellite search
    try:
        response = requests.get(f"{base_url}/satellites", params={"query": "Hubble"})
        if response.status_code == 200:
            satellites = response.json()
            print(f"Satellite search: {satellites.get('count', 0)} satellites found")
        else:
            print(f"Satellite search failed: {response.status_code}")
    except Exception as e:
        print(f"Satellite search failed: {e}")
    
    # Test risk assessment
    try:
        response = requests.post(f"{base_url}/risk-assessment", 
                               json={"satellite_name": "Hubble Space Telescope"})
        if response.status_code == 200:
            assessment = response.json()
            print(f"Risk assessment: Success")
            print(f"  Satellite: {assessment.get('satellite_info', {}).get('name', 'Unknown')}")
            print(f"  Risk Level: {assessment.get('impact_assessment', {}).get('24h', {}).get('risk_level', 'Unknown')}")
        else:
            print(f"Risk assessment failed: {response.status_code}")
    except Exception as e:
        print(f"Risk assessment failed: {e}")
    
    return True

if __name__ == "__main__":
    test_backend_api()