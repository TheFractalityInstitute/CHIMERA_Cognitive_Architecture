#!/usr/bin/env python3
"""
CHIMERA S24 Ultra - Clean version with better error handling
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
        self.error_count = 0
        self.success_count = 0
        
        print("🧠 CHIMERA S24 Ultra Edition")
        print("=" * 40)
        print("Initializing sensors...")
        print("Press Ctrl+C to stop\n")
        
    def read_sensors(self):
        """Read all sensors with better error handling"""
        try:
            result = subprocess.run(
                ['termux-sensor', '-a', '-n', '1'],
                capture_output=True, text=True, 
                timeout=3  # Increased timeout for S24 Ultra
            )
            
            if result.stdout:
                self.success_count += 1
                self.error_count = 0  # Reset error count on success
                return json.loads(result.stdout)
            return None
            
        except subprocess.TimeoutExpired:
            self.error_count += 1
            # Only show error every 10 failures
            if self.error_count % 10 == 0:
                print(f"\r⚠️ Sensor slow (retry {self.error_count})...     ", end="")
            return None
        except Exception:
            return None
    
    def run(self):
        """Main loop with cleaner display"""
        last_activity = "Initializing"
        
        while self.running:
            try:
                # Get ALL sensor data
                data = self.read_sensors()
                
                if data:
                    # Find sensors
                    accel_data = None
                    light_data = None
                    
                    for key, value in data.items():
                        if "Accelerometer" in key and "Uncalibrated" not in key:
                            accel_data = value
                        elif "Light" in key and "Ambient" in key:
                            light_data = value
                    
                    # Process accelerometer
                    if accel_data and 'values' in accel_data:
                        values = accel_data['values']
                        if len(values) >= 3:
                            accel = np.array(values[:3])
                            magnitude = np.linalg.norm(accel)
                            
                            self.movement_buffer.append(magnitude)
                            
                            # Activity detection
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
                                
                                last_activity = activity
                            else:
                                activity = last_activity
                            
                            # Get light value
                            light_str = ""
                            if light_data and 'values' in light_data:
                                lux = light_data['values'][0]
                                if lux < 10:
                                    light_str = f"🌙 Dark ({lux:.0f} lux)"
                                elif lux < 100:
                                    light_str = f"💡 Dim ({lux:.0f} lux)"
                                elif lux < 1000:
                                    light_str = f"☁️ Indoor ({lux:.0f} lux)"
                                else:
                                    light_str = f"☀️ Bright ({lux:.0f} lux)"
                            
                            # Display update
                            uptime = int((datetime.now() - self.start_time).total_seconds())
                            
                            # Build status line
                            status = f"\r{activity} | "
                            status += f"Mag: {magnitude:.1f} | "
                            status += f"X:{accel[0]:5.1f} Y:{accel[1]:5.1f} Z:{accel[2]:5.1f} | "
                            if light_str:
                                status += f"{light_str} | "
                            status += f"Up: {uptime}s | "
                            status += f"✅ {self.success_count} reads"
                            status += " " * 10  # Clear line ending
                            
                            print(status, end="")
                
                time.sleep(0.2)  # Slightly slower to reduce timeouts
                
            except KeyboardInterrupt:
                break
            except Exception:
                pass
        
        print("\n\n👋 CHIMERA shutting down...")
        print(f"📊 Statistics: {self.success_count} successful reads")

if __name__ == "__main__":
    try:
        chimera = CHIMERAS24()
        chimera.run()
    except KeyboardInterrupt:
        print("\nGoodbye!")
