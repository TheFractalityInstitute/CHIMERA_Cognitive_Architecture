#!/usr/bin/env python3
"""
CHIMERA S24 Ultra - Full Sensor Integration
Uses ALL available sensors for complete awareness
"""
import subprocess
import json
import numpy as np
import time
from collections import deque
from datetime import datetime

class CHIMERAFullSensors:
    def __init__(self):
        self.running = True
        self.start_time = datetime.now()
        
        # Buffers for different sensors
        self.accel_buffer = deque(maxlen=50)
        self.gyro_buffer = deque(maxlen=50)
        self.mag_buffer = deque(maxlen=50)
        self.pressure_buffer = deque(maxlen=10)
        
        # State tracking
        self.current_state = {
            'activity': 'Unknown',
            'orientation': 'Unknown',
            'environment': 'Unknown',
            'movement': 'Unknown',
            'heading': 0,
            'altitude': 0,
            'steps': 0
        }
        
        print("🧠 CHIMERA S24 Ultra - Full Sensor Suite")
        print("=" * 50)
        print("Initializing all sensors...")
        print("Press Ctrl+C to stop\n")
        
    def process_sensor_data(self, data):
        """Process all sensor data from batch read"""
        if not data:
            return
        
        state_updates = {}
        
        for key, value in data.items():
            if 'values' not in value:
                continue
                
            values = value['values']
            
            # Accelerometer
            if "Accelerometer" in key and "Uncalibrated" not in key:
                if len(values) >= 3:
                    accel = np.array(values[:3])
                    mag = np.linalg.norm(accel)
                    self.accel_buffer.append(mag)
                    
                    if len(self.accel_buffer) > 10:
                        std = np.std(self.accel_buffer)
                        mean = np.mean(self.accel_buffer)
                        
                        if std < 0.5 and mean < 10.5:
                            state_updates['activity'] = "📱 Still"
                        elif std < 2.0:
                            state_updates['activity'] = "🚶 Walking"
                        elif std < 5.0:
                            state_updates['activity'] = "🏃 Running"
                        else:
                            state_updates['activity'] = "🚗 Vehicle"
            
            # Gyroscope (rotation detection)
            elif "Gyroscope" in key and "Uncalibrated" not in key:
                if len(values) >= 3:
                    gyro = np.array(values[:3])
                    gyro_mag = np.linalg.norm(gyro)
                    self.gyro_buffer.append(gyro_mag)
                    
                    if gyro_mag > 1.0:
                        state_updates['movement'] = "🔄 Rotating fast"
                    elif gyro_mag > 0.1:
                        state_updates['movement'] = "↻ Turning"
                    else:
                        state_updates['movement'] = "━ Stable"
            
            # Magnetometer (compass)
            elif "Magnetometer" in key and "Uncalibrated" not in key:
                if len(values) >= 3:
                    # Calculate heading from magnetometer
                    mag_x, mag_y = values[0], values[1]
                    heading = np.arctan2(mag_y, mag_x) * 180 / np.pi
                    heading = (heading + 360) % 360
                    
                    state_updates['heading'] = heading
                    
                    # Cardinal direction
                    if heading < 22.5 or heading > 337.5:
                        direction = "N"
                    elif heading < 67.5:
                        direction = "NE"
                    elif heading < 112.5:
                        direction = "E"
                    elif heading < 157.5:
                        direction = "SE"
                    elif heading < 202.5:
                        direction = "S"
                    elif heading < 247.5:
                        direction = "SW"
                    elif heading < 292.5:
                        direction = "W"
                    else:
                        direction = "NW"
                    
                    state_updates['orientation'] = f"🧭 {direction} ({heading:.0f}°)"
            
            # Light sensor
            elif "Light" in key and "Ambient" in key:
                if len(values) >= 1:
                    lux = values[0]
                    if lux < 10:
                        state_updates['environment'] = "🌙 Dark"
                    elif lux < 100:
                        state_updates['environment'] = "💡 Indoor"
                    elif lux < 1000:
                        state_updates['environment'] = "☁️ Cloudy"
                    else:
                        state_updates['environment'] = "☀️ Sunny"
            
            # Pressure sensor (altitude/weather)
            elif "Pressure" in key:
                if len(values) >= 1:
                    pressure = values[0]
                    self.pressure_buffer.append(pressure)
                    
                    # Estimate altitude (rough calculation)
                    altitude = 44330 * (1 - (pressure/1013.25)**0.1903)
                    state_updates['altitude'] = altitude
                    
                    # Weather prediction
                    if len(self.pressure_buffer) > 5:
                        pressure_trend = self.pressure_buffer[-1] - self.pressure_buffer[0]
                        if pressure_trend < -1:
                            weather = "📉 Falling (storm coming?)"
                        elif pressure_trend > 1:
                            weather = "📈 Rising (clearing up?)"
                        else:
                            weather = "━ Stable"
                        state_updates['pressure_trend'] = weather
            
            # Proximity sensor
            elif "Proximity" in key and "Proximity Sensor" in key:
                if len(values) >= 1:
                    distance = values[0]
                    if distance < 1:
                        state_updates['proximity'] = "🤚 Object very close!"
                    elif distance < 5:
                        state_updates['proximity'] = "👋 Object near"
                    else:
                        state_updates['proximity'] = "✓ Clear"
            
            # Step counter
            elif "Step" in key:
                if len(values) >= 1:
                    state_updates['steps'] = int(values[0])
        
        # Update current state
        self.current_state.update(state_updates)
    
    def display_status(self):
        """Display comprehensive sensor status"""
        uptime = int((datetime.now() - self.start_time).total_seconds())
        
        # Clear screen for full display (optional)
        # print("\033[2J\033[H")  # Uncomment for clear screen
        
        print("\r" + " " * 100, end="\r")  # Clear line
        
        # Compact single-line display
        status = f"{self.current_state['activity']} "
        status += f"{self.current_state['movement']} "
        status += f"{self.current_state['orientation']} "
        status += f"{self.current_state['environment']} "
        
        if 'altitude' in self.current_state:
            status += f"Alt:{self.current_state['altitude']:.0f}m "
        
        if 'proximity' in self.current_state:
            status += f"{self.current_state['proximity']} "
        
        status += f"Up:{uptime}s"
        
        print(status, end="")
    
    def run(self):
        """Main loop reading all sensors"""
        error_count = 0
        
        while self.running:
            try:
                # Read ALL sensors at once
                result = subprocess.run(
                    ['termux-sensor', '-a', '-n', '1'],
                    capture_output=True, text=True, timeout=3
                )
                
                if result.stdout:
                    data = json.loads(result.stdout)
                    self.process_sensor_data(data)
                    self.display_status()
                    error_count = 0
                else:
                    error_count += 1
                    if error_count % 10 == 0:
                        print(f"\r⚠️ Sensor timeout #{error_count}...", end="")
                
                time.sleep(0.2)  # 5Hz update
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                if error_count % 10 == 0:
                    print(f"\r❌ Error: {e}", end="")
                error_count += 1
                time.sleep(0.5)
        
        print("\n\n📊 Final State:")
        for key, value in self.current_state.items():
            print(f"  {key}: {value}")
        print("\n👋 CHIMERA shutting down...")

if __name__ == "__main__":
    try:
        chimera = CHIMERAFullSensors()
        chimera.run()
    except KeyboardInterrupt:
        print("\nGoodbye!")