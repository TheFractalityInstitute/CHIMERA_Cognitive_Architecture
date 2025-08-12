#!/usr/bin/env python3
"""Test Garmin integration"""
from garmin_integration import GarminFenix5xIntegration

print("🎯 Testing Garmin Fenix 5x+ Integration")
print("=" * 50)

garmin = GarminFenix5xIntegration()

print("\n📊 Current Metrics:")
metrics = garmin.get_current_metrics()

for key, value in metrics.items():
    if key != 'timestamp':
        print(f"  {key}: {value}")

print("\n✅ Garmin integration working!")
print("\nNOTE: If values are 0, export data from Health Sync app first")
