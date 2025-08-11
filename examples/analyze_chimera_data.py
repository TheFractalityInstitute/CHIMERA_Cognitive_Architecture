#!/usr/bin/env python3
"""
Analyze CHIMERA data and generate reports
"""
import sqlite3
import json
import numpy as np
from datetime import datetime, timedelta
import os

class CHIMERAAnalyzer:
    def __init__(self):
        self.data_dir = "/storage/emulated/0/Download/CHIMERA_Data"
        self.db_path = os.path.join(self.data_dir, 'chimera_memory.db')
        
        if not os.path.exists(self.db_path):
            print("❌ No CHIMERA database found. Run chimera_complete.py first!")
            return
        
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        
    def generate_report(self):
        """Generate comprehensive activity report"""
        print("📊 CHIMERA Data Analysis Report")
        print("=" * 60)
        
        # Total data points
        self.cursor.execute('SELECT COUNT(*) FROM sensor_data')
        total = self.cursor.fetchone()[0]
        print(f"\n📈 Total sensor readings: {total}")
        
        # Activity breakdown
        print("\n🏃 Activity Breakdown:")
        self.cursor.execute('''
            SELECT activity, COUNT(*) as count 
            FROM sensor_data 
            WHERE activity IS NOT NULL 
            GROUP BY activity 
            ORDER BY count DESC
        ''')
        
        for activity, count in self.cursor.fetchall():
            percentage = (count / total) * 100
            print(f"  {activity}: {count} ({percentage:.1f}%)")
        
        # Location patterns
        print("\n📍 Locations Visited:")
        self.cursor.execute('''
            SELECT location, COUNT(*) as visits 
            FROM sensor_data 
            WHERE location IS NOT NULL 
            GROUP BY location 
            ORDER BY visits DESC 
            LIMIT 5
        ''')
        
        for location, visits in self.cursor.fetchall():
            print(f"  {location}: {visits} readings")
        
        # Daily routines
        print("\n⏰ Typical Daily Routine:")
        self.cursor.execute('''
            SELECT hour, typical_activity, occurrences 
            FROM daily_routines 
            WHERE day_of_week = ? 
            ORDER BY hour
        ''', (datetime.now().weekday(),))
        
        for hour, activity, occurrences in self.cursor.fetchall():
            if occurrences > 2:  # Only show established patterns
                print(f"  {hour:02d}:00 - Usually {activity}")
        
        # Movement signatures
        print("\n🎯 Learned Movement Patterns:")
        self.cursor.execute('''
            SELECT activity, COUNT(*) as patterns 
            FROM movement_patterns 
            GROUP BY activity
        ''')
        
        for activity, patterns in self.cursor.fetchall():
            print(f"  {activity}: {patterns} unique patterns")
        
        # Insights
        print("\n💡 Recent Insights:")
        self.cursor.execute('''
            SELECT description, timestamp 
            FROM insights 
            ORDER BY timestamp DESC 
            LIMIT 5
        ''')
        
        for description, timestamp in self.cursor.fetchall():
            time_str = datetime.fromtimestamp(timestamp).strftime('%H:%M')
            print(f"  {time_str}: {description}")
        
        print("\n" + "=" * 60)
    
    def export_for_visualization(self):
        """Export data for visualization in other tools"""
        # Export activity timeline
        self.cursor.execute('''
            SELECT timestamp, activity, location 
            FROM sensor_data 
            WHERE activity IS NOT NULL 
            ORDER BY timestamp DESC 
            LIMIT 1000
        ''')
        
        timeline_file = os.path.join(self.data_dir, 'activity_timeline.csv')
        with open(timeline_file, 'w') as f:
            f.write("timestamp,activity,location\n")
            for row in self.cursor.fetchall():
                f.write(f"{row[0]},{row[1]},{row[2]}\n")
        
        print(f"📁 Timeline exported to: {timeline_file}")

if __name__ == "__main__":
    analyzer = CHIMERAAnalyzer()
    analyzer.generate_report()
    analyzer.export_for_visualization()
