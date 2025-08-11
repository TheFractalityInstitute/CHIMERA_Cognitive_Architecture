#!/usr/bin/env python3
"""Interactive sensor test"""
import os
import time

print("📱 Interactive Sensor Test")
print("This will prompt for permissions if needed")
print("=" * 40)

# Test with os.system (shows output directly)
print("\nTrying to read accelerometer...")
os.system('termux-sensor -s accelerometer -n 1')

time.sleep(2)

print("\nTrying to read ALL sensors...")
os.system('termux-sensor -a -n 1')

print("\nIf you see sensor data above, sensors are working!")
