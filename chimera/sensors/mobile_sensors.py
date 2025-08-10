"""Mobile sensor integration for embodied CHIMERA"""
import asyncio
import json
import subprocess
from datetime import datetime
from collections import deque
from typing import Dict, Any, Optional
import numpy as np
from chimera.core.agent import TemporalAgent
from chimera.core.message import NeuralMessage, MessageType, MessagePriority

class MobileSensorHub:
    """Central hub for all mobile sensor access via Termux:API"""
    
    def __init__(self):
        self.sensor_cache = {}
        self.sensor_history = deque(maxlen=1000)
        self.running = False
        
    async def start_streaming(self):
        """Start continuous sensor streaming"""
        self.running = True
        tasks = [
            self._stream_accelerometer(),
            self._stream_gyroscope(),
            self._stream_light(),
            self._stream_battery(),
            self._stream_location(),
        ]
        await asyncio.gather(*tasks)
    
    async def _stream_accelerometer(self):
        """Stream accelerometer data at 100Hz"""
        while self.running:
            try:
                result = subprocess.run(
                    ['termux-sensor', '-s', 'accelerometer', '-n', '1'],
                    capture_output=True, text=True, timeout=0.1
                )
                data = json.loads(result.stdout)
                self.sensor_cache['accelerometer'] = {
                    'x': data['accelerometer']['values'][0],
                    'y': data['accelerometer']['values'][1],
                    'z': data['accelerometer']['values'][2],
                    'timestamp': datetime.now().isoformat()
                }
            except Exception as e:
                print(f"Accelerometer error: {e}")
            await asyncio.sleep(0.01)  # 100Hz
    
    async def _stream_light(self):
        """Stream ambient light sensor at 10Hz"""
        while self.running:
            try:
                result = subprocess.run(
                    ['termux-sensor', '-s', 'light', '-n', '1'],
                    capture_output=True, text=True, timeout=0.5
                )
                data = json.loads(result.stdout)
                self.sensor_cache['light'] = {
                    'lux': data['light']['values'][0],
                    'timestamp': datetime.now().isoformat()
                }
            except Exception as e:
                print(f"Light sensor error: {e}")
            await asyncio.sleep(0.1)  # 10Hz

class AccelerometerAgent(TemporalAgent):
    """Processes raw accelerometer data into movement patterns"""
    
    def __init__(self):
        super().__init__("accelerometer", "sensory", tick_rate=100.0)
        self.movement_buffer = deque(maxlen=100)
        self.activity_state = "unknown"
        
    async def process(self, inputs: Dict[str, Any], timestamp: float) -> Optional[Dict[str, Any]]:
        if 'accelerometer' not in inputs:
            return None
            
        accel = inputs['accelerometer']
        magnitude = np.sqrt(accel['x']**2 + accel['y']**2 + accel['z']**2)
        self.movement_buffer.append(magnitude)
        
        # Simple activity detection
        if len(self.movement_buffer) >= 50:
            std_dev = np.std(self.movement_buffer)
            mean_mag = np.mean(self.movement_buffer)
            
            if std_dev < 0.5 and mean_mag < 10.5:
                self.activity_state = "still"
            elif std_dev < 2.0:
                self.activity_state = "walking"
            elif std_dev < 5.0:
                self.activity_state = "running"
            else:
                self.activity_state = "vehicle"
        
        return {
            'activity': self.activity_state,
            'magnitude': magnitude,
            'pattern': list(self.movement_buffer)[-10:]  # Last 10 samples
        }

class CircadianAgent(TemporalAgent):
    """Learns daily rhythms from light and activity patterns"""
    
    def __init__(self):
        super().__init__("circadian", "integration", tick_rate=0.1)  # Every 10 seconds
        self.hourly_patterns = {hour: {'light': [], 'activity': []} for hour in range(24)}
        self.current_phase = "unknown"
        
    async def process(self, inputs: Dict[str, Any], timestamp: float) -> Optional[Dict[str, Any]]:
        hour = datetime.now().hour
        
        if 'light' in inputs and 'activity' in inputs:
            # Store patterns
            self.hourly_patterns[hour]['light'].append(inputs['light']['lux'])
            self.hourly_patterns[hour]['activity'].append(inputs['activity'])
            
            # Keep only last 100 samples per hour
            if len(self.hourly_patterns[hour]['light']) > 100:
                self.hourly_patterns[hour]['light'].pop(0)
                self.hourly_patterns[hour]['activity'].pop(0)
            
            # Determine phase
            avg_light = np.mean(self.hourly_patterns[hour]['light'][-10:]) if self.hourly_patterns[hour]['light'] else 0
            
            if avg_light < 10:
                if inputs['activity'] == 'still':
                    self.current_phase = "sleeping"
                else:
                    self.current_phase = "night_active"
            elif avg_light < 100:
                self.current_phase = "indoor"
            else:
                self.current_phase = "outdoor"
        
        return {
            'phase': self.current_phase,
            'hour': hour,
            'learned_pattern': f"Usually {self.current_phase} at {hour}:00"
        }

class ProprioceptionAgent(TemporalAgent):
    """Develops sense of 'self' as phone in relation to user"""
    
    def __init__(self):
        super().__init__("proprioception", "integration", tick_rate=10.0)
        self.position_states = deque(maxlen=1000)
        self.relationship_to_user = "unknown"
        
    async def process(self, inputs: Dict[str, Any], timestamp: float) -> Optional[Dict[str, Any]]:
        state = {
            'in_pocket': False,
            'in_hand': False,
            'on_surface': False,
            'being_carried': False
        }
        
        if 'accelerometer' in inputs and 'light' in inputs:
            accel = inputs['accelerometer']
            light = inputs['light']['lux']
            activity = inputs.get('activity', 'unknown')
            
            # Infer position based on sensor fusion
            if light < 5 and activity in ['walking', 'running']:
                state['in_pocket'] = True
                self.relationship_to_user = "carried_hidden"
            elif light > 100 and abs(accel['z'] - 9.8) < 1:
                state['on_surface'] = True
                self.relationship_to_user = "resting_nearby"
            elif activity == 'still' and light > 50:
                state['in_hand'] = True
                self.relationship_to_user = "actively_used"
            elif activity in ['walking', 'vehicle']:
                state['being_carried'] = True
                self.relationship_to_user = "traveling_with"
        
        self.position_states.append(state)
        
        return {
            'self_state': state,
            'relationship': self.relationship_to_user,
            'confidence': self._calculate_confidence()
        }
    
    def _calculate_confidence(self):
        if len(self.position_states) < 10:
            return 0.0
        # Check consistency of recent states
        recent = list(self.position_states)[-10:]
        consistency = sum(1 for s in recent if s == recent[0]) / len(recent)
        return consistency
