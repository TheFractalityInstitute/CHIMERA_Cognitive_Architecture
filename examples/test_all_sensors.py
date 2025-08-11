#!/usr/bin/env python3
"""
Complete S24 Ultra Sensor Test
Tests every sensor and shows which work
"""
import subprocess
import json
import time
from datetime import datetime

class S24SensorTester:
    def __init__(self):
        # S24 Ultra sensor list from your screenshot
        self.sensors_to_test = [
            ("accelerometer", "Accelerometer", "m/s²"),
            ("gyroscope", "Gyroscope", "rad/s"),
            ("magnetometer", "Magnetometer", "μT"),
            ("light", "Light", "lux"),
            ("pressure", "Pressure", "hPa"),
            ("proximity", "Proximity", "cm"),
            ("gravity", "Gravity", "m/s²"),
            ("linear_acceleration", "Linear Accel", "m/s²"),
            ("rotation_vector", "Rotation", ""),
            ("step_detector", "Steps", "steps"),
            ("step_counter", "Step Count", "total"),
            ("orientation", "Orientation", "degrees"),
            ("temperature", "Temperature", "°C"),
            ("humidity", "Humidity", "%"),
            ("heart_rate", "Heart Rate", "bpm"),
        ]
        
        self.working_sensors = {}
        self.failed_sensors = []
        
    def test_sensor(self, sensor_name, display_name):
        """Test individual sensor"""
        try:
            print(f"Testing {display_name}...", end="")
            
            # Try reading the sensor
            result = subprocess.run(
                ['termux-sensor', '-s', sensor_name, '-n', '1'],
                capture_output=True, text=True, timeout=3
            )
            
            if result.stdout:
                try:
                    data = json.loads(result.stdout)
                    
                    # Find the sensor data (key might be complex on S24)
                    sensor_data = None
                    for key, value in data.items():
                        if display_name.lower() in key.lower() or sensor_name in key.lower():
                            sensor_data = value
                            break
                    
                    if not sensor_data and len(data) == 1:
                        # If only one sensor returned, use it
                        sensor_data = list(data.values())[0]
                    
                    if sensor_data and 'values' in sensor_data:
                        values = sensor_data['values']
                        print(f" ✅ Working! Values: {values[:3] if len(values) > 3 else values}")
                        self.working_sensors[sensor_name] = {
                            'name': display_name,
                            'sample': values,
                            'full_key': list(data.keys())[0] if data else sensor_name
                        }
                        return True
                    else:
                        print(f" ❌ No data")
                        self.failed_sensors.append(sensor_name)
                        return False
                        
                except json.JSONDecodeError:
                    print(f" ❌ Invalid response")
                    self.failed_sensors.append(sensor_name)
                    return False
            else:
                print(f" ❌ No response")
                self.failed_sensors.append(sensor_name)
                return False
                
        except subprocess.TimeoutExpired:
            print(f" ❌ Timeout")
            self.failed_sensors.append(sensor_name)
            return False
        except Exception as e:
            print(f" ❌ Error: {e}")
            self.failed_sensors.append(sensor_name)
            return False
    
    def test_all_sensors_batch(self):
        """Test using -a flag to get all at once"""
        print("\n📡 Testing batch sensor read (all at once)...")
        try:
            result = subprocess.run(
                ['termux-sensor', '-a', '-n', '1'],
                capture_output=True, text=True, timeout=5
            )
            
            if result.stdout:
                data = json.loads(result.stdout)
                print(f"✅ Batch read successful! Got {len(data)} sensors:")
                
                for key in data.keys():
                    # Extract sensor type from key
                    sensor_type = "Unknown"
                    if "Accelerometer" in key:
                        sensor_type = "Accelerometer"
                    elif "Gyroscope" in key:
                        sensor_type = "Gyroscope"
                    elif "Magnetometer" in key:
                        sensor_type = "Magnetometer"
                    elif "Light" in key:
                        sensor_type = "Light"
                    elif "Pressure" in key:
                        sensor_type = "Pressure"
                    elif "Proximity" in key:
                        sensor_type = "Proximity"
                    elif "Gravity" in key:
                        sensor_type = "Gravity"
                    elif "Rotation" in key:
                        sensor_type = "Rotation"
                    elif "Step" in key:
                        sensor_type = "Step"
                    
                    values = data[key].get('values', [])
                    print(f"  • {sensor_type}: {key[:40]}... = {values[:3] if len(values) > 3 else values}")
                    
                return data
        except Exception as e:
            print(f"❌ Batch read failed: {e}")
            return None
    
    def run_tests(self):
        """Run all sensor tests"""
        print("🧪 S24 Ultra Complete Sensor Test")
        print("=" * 60)
        print(f"Testing {len(self.sensors_to_test)} sensor types...")
        print("=" * 60)
        
        # Test individual sensors
        print("\n📱 Individual Sensor Tests:\n")
        for sensor_cmd, display_name, unit in self.sensors_to_test:
            self.test_sensor(sensor_cmd, display_name)
            time.sleep(0.5)  # Small delay between tests
        
        # Test batch reading
        batch_data = self.test_all_sensors_batch()
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        print(f"\n✅ Working Sensors ({len(self.working_sensors)}):")
        for sensor, info in self.working_sensors.items():
            print(f"  • {info['name']}: {info['sample']}")
        
        print(f"\n❌ Failed Sensors ({len(self.failed_sensors)}):")
        for sensor in self.failed_sensors:
            print(f"  • {sensor}")
        
        print("\n💡 Recommendation: Use batch mode (-a flag) for best results!")
        
        return self.working_sensors, batch_data

if __name__ == "__main__":
    tester = S24SensorTester()
    working, batch = tester.run_tests()
    
    print("\n🎯 Next step: Run chimera_full_sensors.py to use all working sensors!")