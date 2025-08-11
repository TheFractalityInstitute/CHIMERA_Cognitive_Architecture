#!/usr/bin/env python3
"""
CHIMERA Complete - Full sensor integration with learning & memory
Saves data, learns patterns, integrates Garmin, tracks your life
"""
import subprocess
import json
import numpy as np
import time
import os
import sqlite3
from collections import deque, defaultdict
from datetime import datetime, timedelta
import hashlib

class CHIMERAComplete:
    def __init__(self):
        self.running = True
        self.start_time = datetime.now()
        
        # Create data directory
        self.data_dir = "/storage/emulated/0/Download/CHIMERA_Data"
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize database
        self.init_database()
        
        # Sensor buffers
        self.sensor_buffers = {
            'accel': deque(maxlen=100),
            'gyro': deque(maxlen=100),
            'mag': deque(maxlen=50),
            'pressure': deque(maxlen=20),
            'light': deque(maxlen=20),
            'heart_rate': deque(maxlen=60)
        }
        
        # Pattern learning
        self.movement_signatures = defaultdict(list)
        self.daily_patterns = defaultdict(list)
        self.location_fingerprints = {}
        
        # Current state
        self.current_state = {
            'activity': 'Unknown',
            'location': 'Unknown',
            'heart_rate': 0,
            'stress_level': 0,
            'energy_level': 100,
            'steps_today': 0,
            'calories': 0,
            'altitude': 0,
            'heading': 0,
            'environment': 'Unknown'
        }
        
        print("🧠 CHIMERA Complete System v1.0")
        print("=" * 60)
        print("✅ Database initialized")
        print("✅ Learning systems online")
        print("✅ Pattern recognition active")
        print("Press Ctrl+C to stop\n")
    
    def init_database(self):
        """Initialize SQLite database for long-term memory"""
        self.db_path = os.path.join(self.data_dir, 'chimera_memory.db')
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        
        # Create tables
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS sensor_data (
                timestamp REAL,
                sensor_type TEXT,
                values TEXT,
                activity TEXT,
                location TEXT
            )
        ''')
        
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS movement_patterns (
                activity TEXT,
                signature TEXT,
                confidence REAL,
                learned_at REAL
            )
        ''')
        
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_routines (
                hour INTEGER,
                day_of_week INTEGER,
                typical_activity TEXT,
                typical_location TEXT,
                heart_rate_avg REAL,
                occurrences INTEGER
            )
        ''')
        
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS location_fingerprints (
                location_id TEXT PRIMARY KEY,
                wifi_signature TEXT,
                magnetic_signature TEXT,
                light_pattern TEXT,
                altitude REAL,
                last_seen REAL
            )
        ''')
        
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS insights (
                timestamp REAL,
                insight_type TEXT,
                description TEXT,
                confidence REAL
            )
        ''')
        
        self.conn.commit()
    
    def save_sensor_reading(self, sensor_type, values, activity=None, location=None):
        """Save sensor data to database"""
        self.cursor.execute(
            'INSERT INTO sensor_data VALUES (?, ?, ?, ?, ?)',
            (time.time(), sensor_type, json.dumps(values), activity, location)
        )
        # Commit every 10 inserts for efficiency
        if np.random.random() < 0.1:
            self.conn.commit()
    
    def learn_movement_signature(self, activity, sensor_data):
        """Learn unique movement patterns for each activity"""
        if 'accel' in sensor_data and len(self.sensor_buffers['accel']) > 50:
            # Create signature from accelerometer pattern
            accel_array = np.array(list(self.sensor_buffers['accel']))
            
            # Extract features
            features = {
                'mean': np.mean(accel_array),
                'std': np.std(accel_array),
                'max': np.max(accel_array),
                'min': np.min(accel_array),
                'fft_dominant': self.get_dominant_frequency(accel_array)
            }
            
            # Create signature hash
            signature = hashlib.md5(str(features).encode()).hexdigest()[:8]
            
            # Store pattern
            self.movement_signatures[activity].append(features)
            
            # Save to database
            self.cursor.execute(
                'INSERT INTO movement_patterns VALUES (?, ?, ?, ?)',
                (activity, json.dumps(features), 0.8, time.time())
            )
            
            return signature
        return None
    
    def get_dominant_frequency(self, data):
        """Get dominant frequency from FFT"""
        try:
            fft = np.fft.fft(data)
            freqs = np.fft.fftfreq(len(data))
            dominant_freq = freqs[np.argmax(np.abs(fft[1:len(fft)//2]))]
            return float(dominant_freq)
        except:
            return 0.0
    
    def learn_daily_routine(self):
        """Learn typical activities by time of day"""
        hour = datetime.now().hour
        day = datetime.now().weekday()
        
        # Record current activity pattern
        self.cursor.execute('''
            INSERT OR REPLACE INTO daily_routines 
            (hour, day_of_week, typical_activity, typical_location, heart_rate_avg, occurrences)
            VALUES (?, ?, ?, ?, ?, 
                COALESCE((SELECT occurrences + 1 FROM daily_routines 
                          WHERE hour = ? AND day_of_week = ?), 1))
        ''', (hour, day, self.current_state['activity'], 
              self.current_state['location'], 
              self.current_state['heart_rate'],
              hour, day))
    
    def create_location_fingerprint(self, sensor_data):
        """Create unique fingerprint for current location"""
        fingerprint = {
            'magnetic': None,
            'light': None,
            'altitude': None,
            'pressure': None
        }
        
        # Magnetic signature (unique to each location)
        if 'magnetometer' in sensor_data:
            mag = sensor_data['magnetometer']
            fingerprint['magnetic'] = f"{mag[0]:.1f},{mag[1]:.1f},{mag[2]:.1f}"
        
        # Light pattern
        if len(self.sensor_buffers['light']) > 5:
            fingerprint['light'] = np.mean(self.sensor_buffers['light'])
        
        # Altitude
        if 'altitude' in self.current_state:
            fingerprint['altitude'] = self.current_state['altitude']
        
        # Create location ID from fingerprint
        location_id = hashlib.md5(str(fingerprint).encode()).hexdigest()[:8]
        
        # Check if we've been here before
        self.cursor.execute(
            'SELECT location_id FROM location_fingerprints WHERE location_id = ?',
            (location_id,)
        )
        
        if self.cursor.fetchone():
            location_name = f"Known Location {location_id[:4]}"
        else:
            location_name = f"New Location {location_id[:4]}"
            # Save new location
            self.cursor.execute(
                'INSERT INTO location_fingerprints VALUES (?, ?, ?, ?, ?, ?)',
                (location_id, '', str(fingerprint['magnetic']), 
                 str(fingerprint['light']), fingerprint['altitude'], time.time())
            )
        
        return location_name
    
    def get_garmin_data(self):
        """Simulate Garmin data - replace with real Health Sync integration"""
        # In real implementation, read from Health Sync database
        # For now, simulate realistic data based on activity
        base_hr = 70
        
        if self.current_state['activity'] == '🏃 Running':
            hr = base_hr + np.random.randint(40, 60)
        elif self.current_state['activity'] == '🚶 Walking':
            hr = base_hr + np.random.randint(10, 30)
        else:
            hr = base_hr + np.random.randint(-10, 10)
        
        hrv = 50 + np.random.randn() * 15
        stress = max(0, min(100, 50 + (hr - 70) * 2 + np.random.randn() * 10))
        
        return {
            'heart_rate': hr,
            'hrv': hrv,
            'stress': stress,
            'body_battery': max(5, 100 - stress + np.random.randn() * 5)
        }
    
    def generate_insights(self):
        """Generate insights from learned patterns"""
        insights = []
        
        # Check for unusual activity patterns
        hour = datetime.now().hour
        day = datetime.now().weekday()
        
        self.cursor.execute('''
            SELECT typical_activity FROM daily_routines 
            WHERE hour = ? AND day_of_week = ?
            ORDER BY occurrences DESC LIMIT 1
        ''', (hour, day))
        
        result = self.cursor.fetchone()
        if result and result[0] != self.current_state['activity']:
            insight = f"Unusual activity: You're usually {result[0]} at this time"
            insights.append(insight)
            self.save_insight('routine_deviation', insight, 0.7)
        
        # Check stress patterns
        if self.current_state['stress_level'] > 70:
            insight = "High stress detected. Consider taking a break."
            insights.append(insight)
            self.save_insight('stress_alert', insight, 0.9)
        
        # Check for new movement patterns
        if len(self.movement_signatures) > 0:
            insight = f"Learned {len(self.movement_signatures)} movement patterns"
            insights.append(insight)
        
        return insights
    
    def save_insight(self, insight_type, description, confidence):
        """Save insight to database"""
        self.cursor.execute(
            'INSERT INTO insights VALUES (?, ?, ?, ?)',
            (time.time(), insight_type, description, confidence)
        )
    
    def process_all_sensors(self, data):
        """Process all sensor data and update state"""
        sensor_values = {}
        
        for key, value in data.items():
            if 'values' not in value:
                continue
            
            values = value['values']
            
            # Accelerometer
            if "Accelerometer" in key and "Uncalibrated" not in key:
                if len(values) >= 3:
                    accel = np.array(values[:3])
                    mag = np.linalg.norm(accel)
                    self.sensor_buffers['accel'].append(mag)
                    sensor_values['accelerometer'] = values[:3]
                    
                    # Activity detection
                    if len(self.sensor_buffers['accel']) > 30:
                        std = np.std(self.sensor_buffers['accel'])
                        mean = np.mean(self.sensor_buffers['accel'])
                        
                        if std < 0.5 and mean < 10.5:
                            self.current_state['activity'] = "📱 Still"
                        elif std < 2.0:
                            self.current_state['activity'] = "🚶 Walking"
                        elif std < 5.0:
                            self.current_state['activity'] = "🏃 Running"
                        else:
                            self.current_state['activity'] = "🚗 Vehicle"
            
            # Gyroscope
            elif "Gyroscope" in key and "Uncalibrated" not in key:
                if len(values) >= 3:
                    self.sensor_buffers['gyro'].append(np.linalg.norm(values[:3]))
                    sensor_values['gyroscope'] = values[:3]
            
            # Magnetometer
            elif "Magnetometer" in key and "Uncalibrated" not in key:
                if len(values) >= 3:
                    sensor_values['magnetometer'] = values[:3]
                    # Calculate heading
                    heading = np.arctan2(values[1], values[0]) * 180 / np.pi
                    self.current_state['heading'] = (heading + 360) % 360
            
            # Light
            elif "Light" in key and "Ambient" in key:
                if len(values) >= 1:
                    self.sensor_buffers['light'].append(values[0])
                    sensor_values['light'] = values[0]
                    
                    # Environment detection
                    lux = values[0]
                    if lux < 10:
                        self.current_state['environment'] = "🌙 Dark"
                    elif lux < 100:
                        self.current_state['environment'] = "💡 Indoor"
                    elif lux < 1000:
                        self.current_state['environment'] = "☁️ Cloudy"
                    else:
                        self.current_state['environment'] = "☀️ Sunny"
            
            # Pressure
            elif "Pressure" in key:
                if len(values) >= 1:
                    self.sensor_buffers['pressure'].append(values[0])
                    # Calculate altitude
                    altitude = 44330 * (1 - (values[0]/1013.25)**0.1903)
                    self.current_state['altitude'] = altitude
        
        # Get Garmin data
        garmin = self.get_garmin_data()
        self.current_state['heart_rate'] = garmin['heart_rate']
        self.current_state['stress_level'] = garmin['stress']
        self.current_state['energy_level'] = garmin['body_battery']
        self.sensor_buffers['heart_rate'].append(garmin['heart_rate'])
        
        # Create location fingerprint
        self.current_state['location'] = self.create_location_fingerprint(sensor_values)
        
        # Learn patterns
        if self.current_state['activity'] != 'Unknown':
            self.learn_movement_signature(self.current_state['activity'], sensor_values)
        
        # Learn daily routine
        self.learn_daily_routine()
        
        # Save sensor data
        for sensor_type, values in sensor_values.items():
            self.save_sensor_reading(
                sensor_type, values, 
                self.current_state['activity'],
                self.current_state['location']
            )
        
        return sensor_values
    
    def display_status(self):
        """Display comprehensive status with insights"""
        uptime = int((datetime.now() - self.start_time).total_seconds())
        
        # Clear screen for full display
        print("\033[2J\033[H")
        
        print("🧠 CHIMERA Complete System")
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        # Activity & Movement
        print(f"\n📊 ACTIVITY")
        print(f"  Status: {self.current_state['activity']}")
        print(f"  Location: {self.current_state['location']}")
        print(f"  Environment: {self.current_state['environment']}")
        print(f"  Heading: {self.current_state['heading']:.0f}° ")
        print(f"  Altitude: {self.current_state['altitude']:.0f}m")
        
        # Biometrics
        print(f"\n❤️ BIOMETRICS")
        print(f"  Heart Rate: {self.current_state['heart_rate']} bpm")
        print(f"  Stress: {self.current_state['stress_level']:.0f}%")
        print(f"  Energy: {self.current_state['energy_level']:.0f}%")
        
        # Learning Progress
        print(f"\n🧬 LEARNING")
        print(f"  Movement Patterns: {len(self.movement_signatures)} learned")
        print(f"  Locations Known: {len(self.location_fingerprints)}")
        
        # Get row count
        self.cursor.execute('SELECT COUNT(*) FROM sensor_data')
        data_points = self.cursor.fetchone()[0]
        print(f"  Data Points: {data_points}")
        print(f"  Uptime: {uptime}s")
        
        # Insights
        insights = self.generate_insights()
        if insights:
            print(f"\n💡 INSIGHTS")
            for insight in insights[:3]:  # Show top 3
                print(f"  • {insight}")
        
        print("\n" + "=" * 60)
    
    def export_data(self):
        """Export data for analysis"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        export_file = os.path.join(self.data_dir, f'chimera_export_{timestamp}.json')
        
        # Get all sensor data from last hour
        one_hour_ago = time.time() - 3600
        self.cursor.execute(
            'SELECT * FROM sensor_data WHERE timestamp > ?',
            (one_hour_ago,)
        )
        
        data = {
            'sensor_data': self.cursor.fetchall(),
            'movement_patterns': dict(self.movement_signatures),
            'current_state': self.current_state,
            'timestamp': timestamp
        }
        
        with open(export_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"\n📁 Data exported to: {export_file}")
    
    def run(self):
        """Main execution loop"""
        error_count = 0
        last_display = time.time()
        
        while self.running:
            try:
                # Read all sensors
                result = subprocess.run(
                    ['termux-sensor', '-a', '-n', '1'],
                    capture_output=True, text=True, timeout=3
                )
                
                if result.stdout:
                    data = json.loads(result.stdout)
                    self.process_all_sensors(data)
                    
                    # Update display every 2 seconds
                    if time.time() - last_display > 2:
                        self.display_status()
                        last_display = time.time()
                    
                    error_count = 0
                else:
                    error_count += 1
                
                # Export data every 5 minutes
                if uptime % 300 == 0:
                    self.export_data()
                
                time.sleep(0.5)  # 2Hz sensor reading
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                error_count += 1
                if error_count % 10 == 0:
                    print(f"\nError #{error_count}: {e}")
                time.sleep(1)
        
        # Cleanup
        print("\n\n📊 Final Statistics:")
        self.cursor.execute('SELECT COUNT(*) FROM sensor_data')
        total_readings = self.cursor.fetchone()[0]
        print(f"  Total readings: {total_readings}")
        
        self.cursor.execute('SELECT COUNT(DISTINCT location_id) FROM location_fingerprints')
        unique_locations = self.cursor.fetchone()[0]
        print(f"  Unique locations: {unique_locations}")
        
        self.cursor.execute('SELECT COUNT(*) FROM insights')
        total_insights = self.cursor.fetchone()[0]
        print(f"  Insights generated: {total_insights}")
        
        # Export final data
        self.export_data()
        
        # Close database
        self.conn.commit()
        self.conn.close()
        
        print("\n👋 CHIMERA Complete shutdown. Data saved to:")
        print(f"   {self.data_dir}")

if __name__ == "__main__":
    try:
        chimera = CHIMERAComplete()
        chimera.run()
    except KeyboardInterrupt:
        print("\nShutdown initiated...")
