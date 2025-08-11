#!/usr/bin/env python3
"""
CHIMERA for S24 Ultra - Fixed sensor key names
"""
import subprocess
import json
import numpy as np
import time
from collections import deque
from datetime import datetime

class CHIMERAS24:
    def __init__(self):
        self.running = True
        self.movement_buffer = deque(maxlen=50)
        self.start_time = datetime.now()
        
        print("🧠 CHIMERA S24 Ultra Edition")
        print("=" * 40)
        print("Press Ctrl+C to stop\n")
        
    def read_sensors(self):
        """Read all sensors at once"""
        try:
            result = subprocess.run(
                ['termux-sensor', '-a', '-n', '1'],
                capture_output=True, text=True, timeout=2
            )
            
            if result.stdout:
                return json.loads(result.stdout)
            return None
            
        except Exception as e:
            print(f"\rSensor error: {e}", end="")
            return None
    
    def run(self):
        """Main loop"""
        while self.running:
            try:
                # Get ALL sensor data
                data = self.read_sensors()
                
                if data:
                    # Find accelerometer (look for any key with "Accelerometer" in it)
                    accel_data = None
                    light_data = None
                    gyro_data = None
                    
                    for key, value in data.items():
                        if "Accelerometer" in key and "Uncalibrated" not in key:
                            accel_data = value
                        elif "Light" in key and "Ambient" in key:
                            light_data = value
                        elif "Gyroscope" in key and "Uncalibrated" not in key:
                            gyro_data = value
                    
                    # Process accelerometer
                    if accel_data and 'values' in accel_data:
                        values = accel_data['values']
                        if len(values) >= 3:
                            accel = np.array(values[:3])
                            magnitude = np.linalg.norm(accel)
                            
                            self.movement_buffer.append(magnitude)
                            
                            # Activity detection
                            activity = "Unknown"
                            if len(self.movement_buffer) > 10:
                                std = np.std(self.movement_buffer)
                                mean = np.mean(self.movement_buffer)
                                
                                if std < 0.5 and mean < 10.5:
                                    activity = "📱 Still"
                                elif std < 2.0:
                                    activity = "🚶 Walking"
                                elif std < 5.0:
                                    activity = "🏃 Running"
                                else:
                                    activity = "🚗 Vehicle"
                            
                            # Display
                            uptime = int((datetime.now() - self.start_time).total_seconds())
                            
                            output = f"\r{activity} | Mag: {magnitude:.1f} | "
                            output += f"X:{accel[0]:.1f} Y:{accel[1]:.1f} Z:{accel[2]:.1f} | "
                            
                            # Add light if available
                            if light_data and 'values' in light_data:
                                lux = light_data['values'][0]
                                output += f"Light: {lux:.0f} lux | "
                            
                            output += f"Up: {uptime}s     "
                            print(output, end="")
                    else:
                        print(f"\rNo accel data in this read...     ", end="")
                
                time.sleep(0.1)  # 10Hz update
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"\rError: {e}     ", end="")
                time.sleep(1)
        
        print("\n\n👋 CHIMERA shutting down...")

if __name__ == "__main__":
    try:
        chimera = CHIMERAS24()
        chimera.run()
    except KeyboardInterrupt:
        print("\nGoodbye!")
