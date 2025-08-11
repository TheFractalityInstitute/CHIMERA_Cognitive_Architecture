#!/usr/bin/env python3
"""
CHIMERA Lite Fixed - Better display and error handling
"""
import asyncio
import json
import subprocess
import time
import numpy as np
from collections import deque
from datetime import datetime

class CHIMERALite:
    def __init__(self):
        self.sensor_data = {}
        self.activity_history = deque(maxlen=100)
        self.running = True
        self.start_time = datetime.now()
        self.activity_state = "unknown"
        self.movement_buffer = deque(maxlen=50)
        
        print("🧠 CHIMERA Lite Initializing...")
        print("Press Ctrl+C to stop\n")
        
    async def sensor_loop(self):
        """Main sensor reading loop"""
        while self.running:
            # Read accelerometer with better error handling
            try:
                result = subprocess.run(
                    ['termux-sensor', '-s', 'accelerometer', '-n', '1'],
                    capture_output=True, text=True, timeout=2  # Increased timeout
                )
                if result.returncode == 0:
                    data = json.loads(result.stdout)
                    accel = data['accelerometer']['values']
                    
                    # Calculate magnitude using numpy
                    accel_array = np.array(accel)
                    magnitude = np.linalg.norm(accel_array)
                    
                    self.sensor_data['accelerometer'] = {
                        'x': accel[0],
                        'y': accel[1],
                        'z': accel[2],
                        'magnitude': magnitude
                    }
                    
                    # Update movement buffer
                    self.movement_buffer.append(magnitude)
                    
                    # Detect activity
                    if len(self.movement_buffer) >= 30:
                        movement_array = np.array(self.movement_buffer)
                        std_dev = np.std(movement_array)
                        mean_mag = np.mean(movement_array)
                        
                        if std_dev < 0.5 and mean_mag < 10.5:
                            self.activity_state = "📱 Still"
                        elif std_dev < 2.0:
                            self.activity_state = "🚶 Walking"
                        elif std_dev < 5.0:
                            self.activity_state = "🏃 Running"
                        else:
                            self.activity_state = "🚗 Vehicle"
                            
            except subprocess.TimeoutExpired:
                print("\r⚠️ Sensor timeout - is Termux:API app installed?", end="")
            except Exception as e:
                print(f"\r❌ Error: {e}", end="")
                
            await asyncio.sleep(0.1)  # 10Hz
            
    async def display_loop(self):
        """Simple display without Rich library"""
        while self.running:
            # Clear line and print status
            if 'accelerometer' in self.sensor_data:
                accel = self.sensor_data['accelerometer']
                uptime = int((datetime.now() - self.start_time).total_seconds())
                
                # Simple one-line display
                print(f"\r{self.activity_state} | "
                      f"Mag: {accel['magnitude']:.1f} | "
                      f"X:{accel['x']:.1f} Y:{accel['y']:.1f} Z:{accel['z']:.1f} | "
                      f"Up: {uptime}s     ", end="")
            else:
                print("\r⏳ Waiting for sensor data...     ", end="")
                
            await asyncio.sleep(0.25)
                
    async def run(self):
        """Main run method"""
        try:
            await asyncio.gather(
                self.sensor_loop(),
                self.display_loop()
            )
        except KeyboardInterrupt:
            print("\n\n👋 CHIMERA shutting down gracefully...")
            self.running = False

if __name__ == "__main__":
    chimera = CHIMERALite()
    try:
        asyncio.run(chimera.run())
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
