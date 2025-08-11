#!/usr/bin/env python3
"""Debug sensor reading on S24 Ultra"""
import subprocess
import json
import time

print("🔍 Testing sensor reads on S24 Ultra")
print("=" * 40)

# Test 1: Read accelerometer with exact name
print("\n1. Testing accelerometer read:")
try:
    cmd = ['termux-sensor', '-s', 'accelerometer', '-n', '1']
    print(f"Running: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
    
    print(f"Return code: {result.returncode}")
    print(f"Output: {result.stdout[:500] if result.stdout else 'No output'}")
    print(f"Error: {result.stderr if result.stderr else 'No error'}")
    
    if result.stdout:
        try:
            data = json.loads(result.stdout)
            print("✅ Parsed JSON successfully!")
            print(f"Data structure: {list(data.keys())}")
        except json.JSONDecodeError as e:
            print(f"❌ JSON parse error: {e}")
    
except subprocess.TimeoutExpired:
    print("❌ Command timed out after 5 seconds")
except Exception as e:
    print(f"❌ Error: {e}")

# Test 2: Try with full sensor name
print("\n2. Testing with full sensor name:")
try:
    # Try using the exact sensor name from your list
    cmd = ['termux-sensor', '-s', '"lsm6dsv LSM6DSV Accelerometer Non-wakeup"', '-n', '1']
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
    print(f"Output: {result.stdout[:200] if result.stdout else 'No output'}")
except Exception as e:
    print(f"Error: {e}")

# Test 3: Try getting ALL sensors
print("\n3. Testing ALL sensors at once:")
try:
    cmd = ['termux-sensor', '-a', '-n', '1']
    print(f"Running: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
    
    if result.stdout:
        print(f"Got output! Length: {len(result.stdout)} chars")
        print("First 300 chars:")
        print(result.stdout[:300])
    else:
        print("No output received")
        
except Exception as e:
    print(f"Error: {e}")

print("\n" + "=" * 40)
print("Check which test worked above!")
