#!/usr/bin/env python3
"""Test that all required components work"""

print("🧠 CHIMERA Component Test")
print("=" * 40)

# Test imports
tests = []

try:
    import numpy as np
    print("✅ NumPy:", np.__version__)
    tests.append(True)
except ImportError as e:
    print("❌ NumPy:", e)
    tests.append(False)

try:
    import rich
    print("✅ Rich installed")
    tests.append(True)
except ImportError:
    print("❌ Rich not installed - run: pip install rich")
    tests.append(False)

try:
    import subprocess
    import json
    result = subprocess.run(
        ['termux-sensor', '-l'],
        capture_output=True,
        text=True,
        timeout=2
    )
    sensors = result.stdout
    print("✅ Termux API working")
    print("\nAvailable sensors:")
    print(sensors[:500] + "..." if len(sensors) > 500 else sensors)
    tests.append(True)
except Exception as e:
    print("❌ Termux API error:", e)
    print("   Run: pkg install termux-api")
    print("   Also install Termux:API app from F-Droid")
    tests.append(False)

print("\n" + "=" * 40)
if all(tests):
    print("✅ All tests passed! Ready to run CHIMERA")
    print("\nRun: python chimera_lite.py")
else:
    print("⚠️ Some components missing. Fix errors above.")
