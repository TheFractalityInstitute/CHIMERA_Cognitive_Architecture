"""Biometric sensor integration via Garmin Connect and Health Sync"""
import asyncio
import json
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from collections import deque
from typing import Dict, Any, Optional
import numpy as np
from chimera.core.agent import TemporalAgent
from chimera.core.message import NeuralMessage, MessageType, MessagePriority

class GarminBiometricHub:
    """Interface to Garmin data via Health Sync database"""
    
    def __init__(self):
        # Health Sync typically stores data in Android's app data
        self.health_sync_db = Path("/data/data/com.healthsync/databases/health_sync.db")
        # Alternative: exported CSV location
        self.export_path = Path("/storage/emulated/0/HealthSync/")
        self.biometric_cache = {}
        self.historical_data = {}
        
    async def fetch_current_metrics(self) -> Dict[str, Any]:
        """Fetch latest biometric data from Garmin via Health Sync"""
        metrics = {}
        
        try:
            # Try direct database access first
            if self.health_sync_db.exists():
                conn = sqlite3.connect(str(self.health_sync_db))
                cursor = conn.cursor()
                
                # Get latest heart rate
                cursor.execute("""
                    SELECT value, timestamp FROM heart_rate 
                    ORDER BY timestamp DESC LIMIT 1
                """)
                hr_data = cursor.fetchone()
                if hr_data:
                    metrics['heart_rate'] = {
                        'bpm': hr_data[0],
                        'timestamp': hr_data[1]
                    }
                
                # Get HRV
                cursor.execute("""
                    SELECT value, timestamp FROM hrv 
                    ORDER BY timestamp DESC LIMIT 1
                """)
                hrv_data = cursor.fetchone()
                if hrv_data:
                    metrics['hrv'] = {
                        'ms': hrv_data[0],
                        'timestamp': hrv_data[1]
                    }
                
                conn.close()
                
            # Alternative: Parse exported files
            else:
                metrics = await self._parse_health_sync_exports()
                
        except Exception as e:
            print(f"Garmin data fetch error: {e}")
            # Fallback to mock data for testing
            metrics = self._get_mock_biometrics()
            
        return metrics
    
    async def _parse_health_sync_exports(self) -> Dict[str, Any]:
        """Parse Health Sync CSV exports"""
        metrics = {}
        
        # Look for recent export files
        if self.export_path.exists():
            for file in self.export_path.glob("*.csv"):
                if "heart_rate" in file.name:
                    # Parse HR data
                    with open(file, 'r') as f:
                        lines = f.readlines()[-100:]  # Last 100 entries
                        # Parse CSV format
                        pass
                        
        return metrics
    
    def _get_mock_biometrics(self) -> Dict[str, Any]:
        """Generate realistic mock biometric data for testing"""
        base_hr = 70 + np.random.randn() * 10
        return {
            'heart_rate': {'bpm': base_hr, 'timestamp': datetime.now()},
            'hrv': {'ms': 50 + np.random.randn() * 15, 'timestamp': datetime.now()},
            'stress': {'level': np.random.randint(1, 100), 'timestamp': datetime.now()},
            'body_battery': {'level': np.random.randint(5, 100), 'timestamp': datetime.now()},
            'spo2': {'percent': 95 + np.random.randn() * 2, 'timestamp': datetime.now()},
            'respiration': {'rate': 14 + np.random.randn() * 2, 'timestamp': datetime.now()}
        }

class PhysiologicalStateAgent(TemporalAgent):
    """Integrates biometric data to understand internal state"""
    
    def __init__(self):
        super().__init__("physiological", "integration", tick_rate=1.0)
        self.state_history = deque(maxlen=3600)  # 1 hour of second-by-second data
        self.current_state = "baseline"
        self.stress_level = 0.5
        self.energy_level = 0.5
        self.recovery_score = 0.5
        
    async def process(self, inputs: Dict[str, Any], timestamp: float) -> Optional[Dict[str, Any]]:
        if 'biometrics' not in inputs:
            return None
            
        bio = inputs['biometrics']
        
        # Calculate stress from HRV and HR
        if 'hrv' in bio and 'heart_rate' in bio:
            hrv_ms = bio['hrv']['ms']
            hr_bpm = bio['heart_rate']['bpm']
            
            # Lower HRV + Higher HR = More stress
            hrv_stress = 1.0 - (hrv_ms / 100.0)  # Normalize
            hr_stress = (hr_bpm - 60) / 60.0  # Normalize
            self.stress_level = 0.7 * hrv_stress + 0.3 * hr_stress
            self.stress_level = np.clip(self.stress_level, 0, 1)
        
        # Calculate energy from body battery and stress
        if 'body_battery' in bio:
            battery = bio['body_battery']['level'] / 100.0
            self.energy_level = battery * (1 - self.stress_level * 0.3)
        
        # Determine overall state
        if self.stress_level > 0.7:
            self.current_state = "stressed"
        elif self.stress_level > 0.5 and self.energy_level < 0.3:
            self.current_state = "exhausted"
        elif self.energy_level > 0.7 and self.stress_level < 0.3:
            self.current_state = "optimal"
        elif self.energy_level < 0.3:
            self.current_state = "fatigued"
        else:
            self.current_state = "baseline"
        
        # Store state
        state_snapshot = {
            'state': self.current_state,
            'stress': self.stress_level,
            'energy': self.energy_level,
            'timestamp': timestamp
        }
        self.state_history.append(state_snapshot)
        
        return state_snapshot

class S24UltraSensorHub(MobileSensorHub):
    """Enhanced sensor hub for S24 Ultra's advanced capabilities"""
    
    def __init__(self):
        super().__init__()
        self.advanced_sensors = {
            'magnetometer': {'x': 0, 'y': 0, 'z': 0},
            'barometer': {'pressure': 1013.25, 'altitude': 0},
            'temperature': {'ambient': 20, 'device': 25},
            'proximity': {'distance': 100},
            'gyroscope': {'x': 0, 'y': 0, 'z': 0}
        }
        
    async def _stream_barometer(self):
        """Stream barometric pressure for altitude and weather sensing"""
        while self.running:
            try:
                result = subprocess.run(
                    ['termux-sensor', '-s', 'pressure', '-n', '1'],
                    capture_output=True, text=True, timeout=0.5
                )
                data = json.loads(result.stdout)
                
                pressure = data['pressure']['values'][0]
                # Calculate altitude from pressure
                altitude = 44330 * (1 - (pressure/1013.25)**0.1903)
                
                self.sensor_cache['barometer'] = {
                    'pressure': pressure,
                    'altitude': altitude,
                    'timestamp': datetime.now().isoformat()
                }
            except Exception as e:
                print(f"Barometer error: {e}")
            await asyncio.sleep(1.0)  # 1Hz
    
    async def capture_camera_frame(self):
        """Use S24 Ultra's advanced camera for vision input"""
        try:
            # Use main 200MP sensor in binned mode for efficiency
            subprocess.run([
                'termux-camera-photo',
                '-c', '0',  # Main camera
                '/tmp/chimera_vision.jpg'
            ])
            # Process with lightweight vision model
            return await self._process_image('/tmp/chimera_vision.jpg')
        except Exception as e:
            print(f"Camera error: {e}")
            return None

class ContextualAwarenessAgent(TemporalAgent):
    """Fuses all sensors for complete contextual understanding"""
    
    def __init__(self):
        super().__init__("context", "executive", tick_rate=0.5)
        self.context_model = {
            'location': 'unknown',
            'activity': 'unknown',
            'physiological': 'baseline',
            'environmental': 'unknown',
            'social': 'unknown'
        }
        
    async def process(self, inputs: Dict[str, Any], timestamp: float) -> Optional[Dict[str, Any]]:
        # Integrate all available sensor data
        
        # Physical activity from accelerometer + Garmin
        if 'activity' in inputs and 'physiological_state' in inputs:
            activity = inputs['activity']
            physio = inputs['physiological_state']
            
            # Refined activity detection using heart rate
            if activity == 'walking' and physio['stress'] > 0.6:
                self.context_model['activity'] = 'exercising'
            elif activity == 'still' and physio['energy'] < 0.2:
                self.context_model['activity'] = 'resting'
            else:
                self.context_model['activity'] = activity
        
        # Environmental context from barometer + light
        if 'barometer' in inputs and 'light' in inputs:
            pressure = inputs['barometer']['pressure']
            light = inputs['light']['lux']
            
            # Weather inference
            if pressure < 1009:
                weather = 'stormy'
            elif pressure > 1020:
                weather = 'clear'
            else:
                weather = 'variable'
            
            # Indoor/outdoor detection
            if light > 10000:
                location = 'outdoor_sunny'
            elif light > 1000:
                location = 'outdoor_cloudy' if weather != 'clear' else 'indoor_bright'
            else:
                location = 'indoor_dim'
            
            self.context_model['environmental'] = f"{location}_{weather}"
        
        # Social context from microphone levels + calendar
        # (Add calendar integration later)
        
        return {
            'context': self.context_model,
            'confidence': self._calculate_context_confidence(),
            'timestamp': timestamp
        }
    
    def _calculate_context_confidence(self):
        known_fields = sum(1 for v in self.context_model.values() if v != 'unknown')
        return known_fields / len(self.context_model)
