#!/usr/bin/env python3
"""
Real Garmin Fenix 5x+ Integration via Health Sync
Reads actual biometric data from your watch
"""
import os
import json
import sqlite3
import csv
from datetime import datetime, timedelta
from pathlib import Path
import subprocess

class GarminFenix5xIntegration:
    def __init__(self):
        # Possible Health Sync data locations
        self.health_sync_paths = [
            "/storage/emulated/0/HealthSync/",
            "/storage/emulated/0/Download/HealthSync/",
            "/storage/emulated/0/Documents/HealthSync/",
            "/sdcard/HealthSync/",
            "/data/data/nl.appyhapps.healthsync/databases/",  # Requires root
        ]
        
        # Google Fit database (where Health Sync syncs to)
        self.google_fit_db = "/data/data/com.google.android.apps.fitness/databases/fitness.db"
        
        # Find available data sources
        self.available_sources = self.detect_data_sources()
        self.last_reading = {}
        
        print(f"🎯 Garmin Integration initialized")
        print(f"   Found {len(self.available_sources)} data sources")
        
    def detect_data_sources(self):
        """Find available Garmin/Health Sync data sources"""
        sources = []
        
        # Check for Health Sync CSV exports
        for path in self.health_sync_paths:
            if os.path.exists(path):
                csv_files = list(Path(path).glob("*.csv"))
                if csv_files:
                    sources.append({
                        'type': 'healthsync_csv',
                        'path': path,
                        'files': csv_files
                    })
                    print(f"✅ Found Health Sync CSVs at: {path}")
        
        # Check for Health Sync database (may need root)
        hs_db = "/data/data/nl.appyhapps.healthsync/databases/healthsync.db"
        if os.path.exists(hs_db):
            sources.append({
                'type': 'healthsync_db',
                'path': hs_db
            })
            print(f"✅ Found Health Sync database")
        
        # Check for Garmin Connect exports
        garmin_export = "/storage/emulated/0/Download/garmin_connect_export"
        if os.path.exists(garmin_export):
            sources.append({
                'type': 'garmin_export',
                'path': garmin_export
            })
            print(f"✅ Found Garmin Connect exports")
        
        # Check notification access (can read Garmin notifications)
        try:
            result = subprocess.run(
                ['dumpsys', 'notification'], 
                capture_output=True, text=True, timeout=2
            )
            if 'com.garmin.android.apps.connectmobile' in result.stdout:
                sources.append({
                    'type': 'notifications',
                    'source': 'garmin_connect'
                })
                print(f"✅ Can read Garmin Connect notifications")
        except:
            pass
            
        return sources
    
    def read_latest_csv_data(self):
        """Read the most recent data from Health Sync CSV exports"""
        for source in self.available_sources:
            if source['type'] == 'healthsync_csv':
                data = {}
                
                # Read each CSV type
                for csv_file in source['files']:
                    filename = csv_file.name.lower()
                    
                    try:
                        # Heart Rate CSV
                        if 'heart' in filename or 'hr' in filename:
                            with open(csv_file, 'r') as f:
                                reader = csv.DictReader(f)
                                rows = list(reader)
                                if rows:
                                    latest = rows[-1]  # Get most recent
                                    data['heart_rate'] = {
                                        'value': float(latest.get('value', latest.get('bpm', 0))),
                                        'timestamp': latest.get('timestamp', latest.get('date', ''))
                                    }
                        
                        # HRV CSV
                        elif 'hrv' in filename:
                            with open(csv_file, 'r') as f:
                                reader = csv.DictReader(f)
                                rows = list(reader)
                                if rows:
                                    latest = rows[-1]
                                    data['hrv'] = {
                                        'value': float(latest.get('value', latest.get('rmssd', 0))),
                                        'timestamp': latest.get('timestamp', '')
                                    }
                        
                        # Stress CSV
                        elif 'stress' in filename:
                            with open(csv_file, 'r') as f:
                                reader = csv.DictReader(f)
                                rows = list(reader)
                                if rows:
                                    latest = rows[-1]
                                    data['stress'] = {
                                        'value': float(latest.get('value', latest.get('level', 0))),
                                        'timestamp': latest.get('timestamp', '')
                                    }
                        
                        # Steps CSV
                        elif 'step' in filename:
                            with open(csv_file, 'r') as f:
                                reader = csv.DictReader(f)
                                rows = list(reader)
                                if rows:
                                    latest = rows[-1]
                                    data['steps'] = {
                                        'value': int(latest.get('value', latest.get('steps', 0))),
                                        'timestamp': latest.get('timestamp', '')
                                    }
                        
                        # Body Battery CSV
                        elif 'battery' in filename or 'energy' in filename:
                            with open(csv_file, 'r') as f:
                                reader = csv.DictReader(f)
                                rows = list(reader)
                                if rows:
                                    latest = rows[-1]
                                    data['body_battery'] = {
                                        'value': float(latest.get('value', latest.get('level', 0))),
                                        'timestamp': latest.get('timestamp', '')
                                    }
                        
                        # SpO2 CSV
                        elif 'spo2' in filename or 'oxygen' in filename:
                            with open(csv_file, 'r') as f:
                                reader = csv.DictReader(f)
                                rows = list(reader)
                                if rows:
                                    latest = rows[-1]
                                    data['spo2'] = {
                                        'value': float(latest.get('value', latest.get('percentage', 0))),
                                        'timestamp': latest.get('timestamp', '')
                                    }
                        
                        # Respiration CSV
                        elif 'respiration' in filename or 'breath' in filename:
                            with open(csv_file, 'r') as f:
                                reader = csv.DictReader(f)
                                rows = list(reader)
                                if rows:
                                    latest = rows[-1]
                                    data['respiration'] = {
                                        'value': float(latest.get('value', latest.get('rate', 0))),
                                        'timestamp': latest.get('timestamp', '')
                                    }
                                    
                    except Exception as e:
                        print(f"Error reading {csv_file}: {e}")
                
                return data
        
        return {}
    
    def read_realtime_notifications(self):
        """Read real-time data from Garmin Connect notifications"""
        try:
            # Use logcat to capture Garmin notifications
            result = subprocess.run(
                ['logcat', '-d', '-s', 'GarminConnect:*'],
                capture_output=True, text=True, timeout=1
            )
            
            # Parse notification data
            lines = result.stdout.split('\n')
            data = {}
            
            for line in lines[-50:]:  # Check last 50 lines
                if 'heart rate' in line.lower():
                    # Extract HR from notification
                    import re
                    hr_match = re.search(r'(\d+)\s*bpm', line, re.IGNORECASE)
                    if hr_match:
                        data['heart_rate'] = int(hr_match.group(1))
                
                elif 'stress' in line.lower():
                    stress_match = re.search(r'stress.*?(\d+)', line, re.IGNORECASE)
                    if stress_match:
                        data['stress'] = int(stress_match.group(1))
            
            return data
            
        except:
            return {}
    
    def get_current_metrics(self):
        """Get all current Garmin metrics"""
        metrics = {
            'heart_rate': 0,
            'hrv': 0,
            'stress': 0,
            'body_battery': 0,
            'steps': 0,
            'spo2': 0,
            'respiration': 0,
            'calories': 0,
            'distance': 0,
            'floors': 0,
            'intensity_minutes': 0,
            'timestamp': datetime.now().isoformat()
        }
        
        # Try CSV data first (most reliable)
        csv_data = self.read_latest_csv_data()
        if csv_data:
            for key in csv_data:
                if key in metrics and 'value' in csv_data[key]:
                    metrics[key] = csv_data[key]['value']
            
            self.last_reading = metrics
            return metrics
        
        # Try notification data for real-time updates
        notif_data = self.read_realtime_notifications()
        if notif_data:
            metrics.update(notif_data)
            self.last_reading = metrics
            return metrics
        
        # Return last known good data if available
        if self.last_reading:
            return self.last_reading
            
        return metrics
    
    def get_activity_history(self, hours=24):
        """Get activity history for the last N hours"""
        history = []
        
        for source in self.available_sources:
            if source['type'] == 'healthsync_csv':
                # Read historical data from CSVs
                for csv_file in source['files']:
                    if 'heart' in csv_file.name.lower():
                        try:
                            with open(csv_file, 'r') as f:
                                reader = csv.DictReader(f)
                                for row in reader:
                                    # Parse timestamp and filter by time range
                                    history.append({
                                        'metric': 'heart_rate',
                                        'value': float(row.get('value', 0)),
                                        'timestamp': row.get('timestamp', '')
                                    })
                        except:
                            pass
        
        return history

class GarminHealthSync:
    """Alternative method using Health Sync app's export feature"""
    
    def __init__(self):
        self.export_dir = "/storage/emulated/0/HealthSync"
        self.setup_auto_export()
    
    def setup_auto_export(self):
        """Setup automatic export from Health Sync (requires Tasker or similar)"""
        # Create export directory
        os.makedirs(self.export_dir, exist_ok=True)
        
        # Create Tasker profile (user needs to import this)
        tasker_profile = """
        Profile: Garmin Auto Export
        Trigger: Every 5 minutes
        Task:
        1. Launch App: Health Sync
        2. Wait: 2 seconds
        3. AutoInput: Click "Export"
        4. AutoInput: Click "CSV"
        5. AutoInput: Click "Last Hour"
        6. AutoInput: Click "Export"
        7. Go Home
        """
        
        profile_file = os.path.join(self.export_dir, "tasker_profile.txt")
        with open(profile_file, 'w') as f:
            f.write(tasker_profile)
        
        print(f"📝 Tasker profile saved to: {profile_file}")
        print("   Import this into Tasker for automatic exports")
