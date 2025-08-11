#!/usr/bin/env python3
"""
Interactive Sensor Dashboard for S24 Ultra
Shows all sensors in organized display
"""
import subprocess
import json
import time
from datetime import datetime

def get_all_sensors():
    """Get all sensor data"""
    try:
        result = subprocess.run(
            ['termux-sensor', '-a', '-n', '1'],
            capture_output=True, text=True, timeout=3
        )
        if result.stdout:
            return json.loads(result.stdout)
    except:
        pass
    return None

def main():
    print("\033[2J\033[H")  # Clear screen
    print("🎛️ S24 Ultra Sensor Dashboard")
    print("Press Ctrl+C to exit")
    print("=" * 50)
    
    while True:
        try:
            data = get_all_sensors()
            
            if data:
                # Move cursor to home
                print("\033[H")
                print("🎛️ S24 Ultra Sensor Dashboard")
                print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
                print("=" * 50)
                
                # Motion Sensors
                print("\n📊 MOTION SENSORS:")
                for key, value in data.items():
                    if "Accelerometer" in key and "Uncalibrated" not in key:
                        v = value.get('values', [0,0,0])
                        print(f"  Accel: X={v[0]:6.2f} Y={v[1]:6.2f} Z={v[2]:6.2f}")
                    elif "Gyroscope" in key and "Uncalibrated" not in key:
                        v = value.get('values', [0,0,0])
                        print(f"  Gyro:  X={v[0]:6.2f} Y={v[1]:6.2f} Z={v[2]:6.2f}")
                
                # Position Sensors
                print("\n🧭 POSITION SENSORS:")
                for key, value in data.items():
                    if "Magnetometer" in key and "Uncalibrated" not in key:
                        v = value.get('values', [0,0,0])
                        print(f"  Mag:   X={v[0]:6.1f} Y={v[1]:6.1f} Z={v[2]:6.1f}")
                    elif "Orientation" in key:
                        v = value.get('values', [0,0,0])
                        print(f"  Orient: Az={v[0]:6.1f} Pi={v[1]:6.1f} Ro={v[2]:6.1f}")
                
                # Environment Sensors
                print("\n🌍 ENVIRONMENT SENSORS:")
                for key, value in data.items():
                    if "Light" in key and "Ambient" in key:
                        v = value.get('values', [0])
                        print(f"  Light:     {v[0]:8.1f} lux")
                    elif "Pressure" in key:
                        v = value.get('values', [0])
                        print(f"  Pressure:  {v[0]:8.1f} hPa")
                    elif "Proximity" in key:
                        v = value.get('values', [0])
                        print(f"  Proximity: {v[0]:8.1f} cm")
                
                print("\n" + "=" * 50)
                
            time.sleep(0.5)  # Update twice per second
            
        except KeyboardInterrupt:
            print("\n\nExiting dashboard...")
            break

if __name__ == "__main__":
    main()