#!/usr/bin/env python3
"""Test if Termux:API works"""
import subprocess
import time

print("Testing Termux:API...")
print("=" * 40)

# Test 1: List sensors
print("\n1. Listing available sensors:")
try:
    result = subprocess.run(['termux-sensor', '-l'], 
                          capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print("✅ Sensor list:", result.stdout[:200])
    else:
        print("❌ Failed to list sensors")
        print("Install Termux:API app from Play Store/F-Droid!")
except Exception as e:
    print(f"❌ Error: {e}")
    print("Run: pkg install termux-api")

# Test 2: Single sensor read
print("\n2. Reading accelerometer once:")
try:
    result = subprocess.run(['termux-sensor', '-s', 'accelerometer', '-n', '1'],
                          capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print("✅ Got data:", result.stdout[:100])
    else:
        print("❌ Failed to read accelerometer")
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "=" * 40)
print("If both tests failed, you need to:")
print("1. Install 'Termux:API' app (separate app!)")
print("2. Grant permissions when prompted")
print("3. Run: pkg install termux-api")
