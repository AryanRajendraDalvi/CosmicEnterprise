#!/usr/bin/env python3
"""
Script to start the Cosmic Weather Insurance API
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing backend dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        sys.exit(1)

def start_api():
    """Start the Flask API"""
    print("Starting Cosmic Weather Insurance API...")
    try:
        # Change to the Backend directory
        backend_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(backend_dir)
        
        # Start the API
        subprocess.check_call([sys.executable, "api.py"])
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting API: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n🛑 API stopped by user")
        sys.exit(0)

if __name__ == "__main__":
    # Install requirements if needed
    if len(sys.argv) > 1 and sys.argv[1] == "--install":
        install_requirements()
    
    # Start the API
    start_api()