#!/usr/bin/env python3
"""
Find where Health Sync is storing Garmin data on your S24 Ultra
"""
import os
import subprocess
from pathlib import Path
import json

print("🔍 Searching for Garmin/Health Sync data...")
print("=" * 50)

# Common app data locations
search_paths = [
    "/storage/emulated/0/",
    "/sdcard/",
    "/storage/emulated/0/Android/data/",
    "/storage/emulated/0/Documents/",
    "/storage/emulated/0/Download/",
]

found_items = []

# Search for Health Sync related files
print("\n📱 Searching for Health Sync files...")
for base_path in search_paths:
    if os.path.exists(base_path):
        for root, dirs, files in os.walk(base_path):
            # Skip system directories
            if 'Android/obb' in root or '.thumbnails' in root:
                continue
            
            # Look for Health Sync files
            for file in files:
                file_lower = file.lower()
                if any(keyword in file_lower for keyword in ['health', 'garmin', 'fitness', 'heart', 'hrv', 'stress']):
                    if file.endswith(('.csv', '.json', '.xml', '.db', '.txt')):
                        full_path = os.path.join(root, file)
                        size = os.path.getsize(full_path) / 1024  # KB
                        found_items.append((full_path, size))
                        print(f"  Found: {file} ({size:.1f} KB)")
            
            # Look for relevant directories
            for dir_name in dirs:
                dir_lower = dir_name.lower()
                if any(keyword in dir_lower for keyword in ['healthsync', 'garmin', 'fitness', 'googlefit']):
                    print(f"  📁 Directory: {os.path.join(root, dir_name)}")

# Check Google Fit integration
print("\n📱 Checking Google Fit...")
try:
    # Check if Google Fit is installed
    result = subprocess.run(['pm', 'list', 'packages'], capture_output=True, text=True)
    if 'com.google.android.apps.fitness' in result.stdout:
        print("  ✅ Google Fit is installed")
        
        # Try to find Google Fit data
        gfit_paths = [
            "/storage/emulated/0/Android/data/com.google.android.apps.fitness/",
            "/data/data/com.google.android.apps.fitness/",  # Needs root
        ]
        
        for path in gfit_paths:
            if os.path.exists(path):
                print(f"  📁 Google Fit data found at: {path}")
except:
    pass

# Check Samsung Health
print("\n📱 Checking Samsung Health...")
try:
    result = subprocess.run(['pm', 'list', 'packages'], capture_output=True, text=True)
    if 'com.sec.android.app.shealth' in result.stdout:
        print("  ✅ Samsung Health is installed")
        
        # Samsung Health data locations
        shealth_paths = [
            "/storage/emulated/0/Android/data/com.sec.android.app.shealth/",
            "/storage/emulated/0/Samsung Health/",
            "/storage/emulated/0/SHealth/",
        ]
        
        for path in shealth_paths:
            if os.path.exists(path):
                print(f"  📁 Samsung Health data found at: {path}")
except:
    pass

# Summary
print("\n" + "=" * 50)
print(f"📊 Found {len(found_items)} potential data files")

if found_items:
    print("\n🎯 Most likely data sources (largest files):")
    # Sort by size and show top 5
    found_items.sort(key=lambda x: x[1], reverse=True)
    for path, size in found_items[:5]:
        print(f"  {path}")
        print(f"    Size: {size:.1f} KB")
