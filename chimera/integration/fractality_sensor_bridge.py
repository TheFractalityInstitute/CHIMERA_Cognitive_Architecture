# chimera/integration/fractality_sensor_bridge.py
"""
Bridge between your working sensor system and the complete Fractality-CHIMERA
Connects chimera_complete.py to the new architecture
"""

import asyncio
import numpy as np
from typing import Dict, Any, Optional
import json
from pathlib import Path

class FractalitySensorBridge:
    """
    Connects your existing CHIMERAComplete sensor system 
    to the new Fractality-integrated architecture
    """
    
    def __init__(self, chimera_complete, fractality_chimera):
        # Your existing working sensor system
        self.sensors = chimera_complete  # This already works with your phone!
        
        # The new Fractality-complete system
        self.fractality = fractality_chimera
        
        # Sensor-to-quantum mapping
        self.sensor_quantum_mapping = {
            'accelerometer': 'movement_superposition',
            'heart_rate': 'physiological_coherence',
            'light': 'environmental_awareness',
            'pressure': 'altitude_consciousness'
        }
        
        # Energy cost of sensor processing
        self.sensor_energy_costs = {
            'accelerometer': 0.5,
            'gyroscope': 0.4,
            'magnetometer': 0.3,
            'light': 0.2,
            'pressure': 0.2,
            'heart_rate': 1.0,  # Garmin processing
            'gps': 1.5
        }
        
    async def sensor_to_fractality_loop(self):
        """
        Main loop that feeds sensor data into Fractality-CHIMERA
        """
        print("🔌 Connecting sensors to Fractality consciousness...")
        
        while True:
            try:
                # Get sensor data from your existing system
                sensor_state = self.sensors.current_state
                sensor_buffers = self.sensors.sensor_buffers
                garmin_data = self.sensors.get_garmin_data()
                
                # Check energy availability
                energy_available = await self._check_energy_for_sensors()
                
                if energy_available:
                    # Process through Fractality systems
                    await self._process_sensory_quantum(sensor_state, garmin_data)
                    await self._update_fractal_memory(sensor_state)
                    await self._resonance_learning_from_sensors(sensor_buffers)
                    await self._ethical_evaluation_of_state(sensor_state)
                    
                # Update consciousness metrics based on sensors
                self._update_consciousness_from_sensors(sensor_state, garmin_data)
                
                # Sleep based on energy state
                sleep_time = self._adaptive_sleep_time()
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                print(f"Sensor bridge error: {e}")
                await asyncio.sleep(1)
    
    async def _check_energy_for_sensors(self) -> bool:
        """Check if we have energy for sensor processing"""
        total_cost = sum(self.sensor_energy_costs.values())
        
        # Request energy from Fractality ATP system
        approved, actual = self.fractality.energy.request_energy(
            "sensor_processing", total_cost
        )
        
        return approved
    
    async def _process_sensory_quantum(self, sensor_state: Dict, garmin_data: Dict):
        """
        Process sensor data through quantum superposition
        Creates quantum states from classical sensor readings
        """
        # Movement superposition
        if sensor_state.get('activity'):
            activities = ['Still', 'Walking', 'Running', 'Vehicle']
            
            # Create superposition weighted by likelihood
            weights = []
            current_activity = sensor_state['activity']
            for activity in activities:
                if activity in current_activity:
                    weights.append(0.7)  # High probability for detected activity
                else:
                    weights.append(0.1)  # Low probability for others
                    
            movement_state = self.fractality.quantum_bridge.create_superposition(
                activities, weights
            )
            self.fractality.quantum_bridge.quantum_registers['movement'] = movement_state
        
        # Physiological coherence from heart rate
        if garmin_data.get('heart_rate'):
            hr = garmin_data['heart_rate']
            hrv = garmin_data.get('hrv', 50)
            
            # Create quantum state representing physiological coherence
            states = ['stressed', 'calm', 'focused', 'excited']
            
            # Weight by biometric indicators
            if hr > 100:
                weights = [0.3, 0.1, 0.2, 0.4]  # Likely excited or stressed
            elif hr < 60:
                weights = [0.1, 0.6, 0.2, 0.1]  # Likely calm
            else:
                weights = [0.2, 0.3, 0.4, 0.1]  # Likely focused
                
            physio_state = self.fractality.quantum_bridge.create_superposition(
                states, weights
            )
            self.fractality.quantum_bridge.quantum_registers['physiology'] = physio_state
    
    async def _update_fractal_memory(self, sensor_state: Dict):
        """Store sensor experiences in fractal memory"""
        # Create memory from current state
        memory_content = {
            'type': 'sensory_experience',
            'activity': sensor_state.get('activity', 'unknown'),
            'location': sensor_state.get('location', 'unknown'),
            'heart_rate': sensor_state.get('heart_rate', 0),
            'environment': sensor_state.get('environment', 'unknown'),
            'timestamp': time.time()
        }
        
        # Determine memory type based on content
        if sensor_state.get('activity') in ['Running', 'Walking']:
            memory_type = 'procedural'  # Movement memory
        elif sensor_state.get('heart_rate', 0) > 100:
            memory_type = 'emotional'   # High arousal memory
        else:
            memory_type = 'episodic'    # General experience
            
        # Store in fractal memory system
        memory_node = self.fractality.memory.store_memory(
            memory_content,
            memory_type,
            context={'sensor_derived': True}
        )
        
        # Create associations with similar states
        if sensor_state.get('location') != 'Unknown':
            # Associate memories from same location
            location_memories = [
                m for m in self.fractality.memory.memory_index.values()
                if isinstance(m.content, dict) and 
                m.content.get('location') == sensor_state['location']
            ]
            
            for related_memory in location_memories[-5:]:  # Last 5 from this location
                memory_node.associations.append(related_memory.id)
    
    async def _resonance_learning_from_sensors(self, sensor_buffers: Dict):
        """Learn patterns from sensor data using resonance"""
        # Convert sensor buffers to learning patterns
        for sensor_type, buffer in sensor_buffers.items():
            if len(buffer) > 50:  # Enough data for pattern
                # Convert to numpy array
                pattern_data = np.array(list(buffer))
                
                # Learn through resonance
                pattern = self.fractality.resonance_learning.learn_pattern(
                    pattern_data,
                    label=f"{sensor_type}_pattern"
                )
                
                # Check for resonance with existing patterns
                activated = self.fractality.resonance_learning.propagate_resonance(
                    pattern.pattern_id
                )
                
                if len(activated) > 3:
                    # Multiple patterns resonating - emergent behavior
                    print(f"🔄 Resonance cascade: {sensor_type} → {activated[:3]}")
    
    async def _ethical_evaluation_of_state(self, sensor_state: Dict):
        """Evaluate current state ethically"""
        # Evaluate if current activity aligns with well-being
        action = {
            'type': 'physical_activity',
            'activity': sensor_state.get('activity', 'unknown'),
            'heart_rate': sensor_state.get('heart_rate', 70),
            'stress': sensor_state.get('stress_level', 50)
        }
        
        context = {
            'affects_others': False,  # Physical activity mainly affects self
            'impact_on_self': self._calculate_health_impact(sensor_state)
        }
        
        # Get ethical evaluation
        ethical_vector = self.fractality.canon.evaluate_action(action, context)
        
        # Learn from the evaluation
        outcome = {
            'value': ethical_vector.harm_score,  # Positive = beneficial
            'health_impact': context['impact_on_self']
        }
        
        self.fractality.canon.learn_from_outcome(action, outcome, ethical_vector)
    
    def _calculate_health_impact(self, sensor_state: Dict) -> float:
        """Calculate health impact of current state"""
        impact = 0.0
        
        # Physical activity is generally positive
        if 'Walking' in sensor_state.get('activity', ''):
            impact += 0.3
        elif 'Running' in sensor_state.get('activity', ''):
            impact += 0.5
        elif 'Still' in sensor_state.get('activity', ''):
            impact -= 0.1  # Too much stillness is negative
            
        # Heart rate in healthy zone
        hr = sensor_state.get('heart_rate', 70)
        if 60 <= hr <= 100:
            impact += 0.2
        elif hr > 140:
            impact -= 0.3  # Very high HR could be concerning
            
        # Stress level
        stress = sensor_state.get('stress_level', 50)
        if stress < 30:
            impact += 0.2
        elif stress > 70:
            impact -= 0.4
            
        return np.tanh(impact)  # Normalize to [-1, 1]
    
    def _update_consciousness_from_sensors(self, sensor_state: Dict, garmin_data: Dict):
        """Update consciousness metrics based on sensor input"""
        # Physical coherence affects quantum coherence
        if garmin_data.get('hrv'):
            hrv_coherence = garmin_data['hrv'] / 100.0  # Normalize
            self.fractality.consciousness_state['quantum_coherence'] = \
                0.7 * self.fractality.consciousness_state['quantum_coherence'] + \
                0.3 * hrv_coherence
        
        # Activity affects classical integration
        if sensor_state.get('activity') != 'Unknown':
            # Known activity = better integration
            self.fractality.consciousness_state['classical_integration'] = min(
                1.0,
                self.fractality.consciousness_state['classical_integration'] * 1.05
            )
        
        # Stress affects ethical alignment
        stress = sensor_state.get('stress_level', 50) / 100.0
        self.fractality.consciousness_state['ethical_alignment'] *= (1.5 - stress)
        self.fractality.consciousness_state['ethical_alignment'] = np.clip(
            self.fractality.consciousness_state['ethical_alignment'], 0, 1
        )
    
    def _adaptive_sleep_time(self) -> float:
        """Adaptive sleep based on energy and activity"""
        energy_state = self.fractality.energy.get_energy_state()
        
        if energy_state == EnergyState.PEAK:
            return 0.1  # 10Hz when full energy
        elif energy_state == EnergyState.NORMAL:
            return 0.5  # 2Hz normal
        elif energy_state == EnergyState.CONSERVE:
            return 1.0  # 1Hz conserving
        else:
            return 2.0  # 0.5Hz minimal
