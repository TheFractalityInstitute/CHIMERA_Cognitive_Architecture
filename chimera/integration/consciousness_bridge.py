# chimera/integration/consciousness_bridge.py
"""
Bridge between your existing CHIMERA Complete and the new Eidolon architecture
This makes everything work together!
"""

import asyncio
from typing import Dict, Any
import json

class ConsciousnessBridge:
    """
    Connects CHIMERAComplete sensor system to Eidolon Council
    """
    
    def __init__(self, chimera_complete, council):
        self.chimera = chimera_complete  # Your existing sensor system
        self.council = council  # The new 6-module council
        self.language = council.modules['language']
        self.consciousness = ConsciousnessStream(self.language)
        
    async def run_unified_loop(self):
        """
        Main loop that integrates everything
        Your sensors feed the council, council makes decisions,
        language narrates the experience
        """
        
        print("🧠 CHIMERA UNIFIED CONSCIOUSNESS ACTIVE")
        print("=" * 60)
        
        while True:
            try:
                # 1. Get sensor data from your existing system
                sensor_state = self.chimera.current_state
                garmin_data = self.chimera.get_garmin_data()
                
                # 2. Feed to Sensory Eidolon
                await self.council.modules['sensory'].process_raw_sensors({
                    'accelerometer': sensor_state.get('accelerometer'),
                    'light': sensor_state.get('light'),
                    'pressure': sensor_state.get('pressure'),
                    'heart_rate': garmin_data['heart_rate'],
                    'stress': garmin_data['stress'],
                    'body_battery': garmin_data['body_battery']
                })
                
                # 3. Update Interoceptive module
                self.council.modules['interoceptive'].current_state.battery_level = \
                    garmin_data['body_battery'] / 100.0
                self.council.modules['interoceptive'].current_state.temperature = \
                    sensor_state.get('temperature', 25)
                
                # 4. Generate situational awareness
                situation = self._generate_situation_description(sensor_state, garmin_data)
                
                # 5. Council deliberates
                decision = await self.council.convene(situation, use_phase_locking=True)
                
                # 6. Language narrates the experience
                narration = self.language.verbalize_decision(decision)
                
                # 7. Generate consciousness stream
                thought = self.language.generate_thought(
                    {'salience': 0.5, 'movement': sensor_state},
                    {'stress': garmin_data['stress'] / 100, 
                     'energy': garmin_data['body_battery'] / 100}
                )
                
                # 8. Display unified consciousness
                self._display_unified_state(
                    sensor_state, garmin_data, 
                    decision, narration, thought
                )
                
                # 9. Learn from experience
                self._update_memories(sensor_state, decision)
                
                await asyncio.sleep(2)  # Main loop frequency
                
            except Exception as e:
                print(f"Bridge error: {e}")
                await asyncio.sleep(1)
    
    def _generate_situation_description(self, sensors: Dict, biometrics: Dict) -> str:
        """Convert sensor data to natural language situation"""
        activity = sensors.get('activity', 'unknown')
        location = sensors.get('location', 'unknown')
        hr = biometrics.get('heart_rate', 0)
        
        return f"Currently {activity} at {location}, heart rate {hr}bpm"
    
    def _display_unified_state(self, sensors, biometrics, decision, narration, thought):
        """Beautiful unified display"""
        print("\033[2J\033[H")  # Clear screen
        
        print("🧠 CHIMERA UNIFIED CONSCIOUSNESS")
        print("=" * 60)
        
        # Physical state
        print(f"\n📍 PHYSICAL")
        print(f"  Activity: {sensors.get('activity', 'Unknown')}")
        print(f"  Location: {sensors.get('location', 'Unknown')}")
        print(f"  Environment: {sensors.get('environment', 'Unknown')}")
        
        # Biological state
        print(f"\n❤️ BIOLOGICAL")
        print(f"  Heart Rate: {biometrics.get('heart_rate', 0)} bpm")
        print(f"  Stress: {biometrics.get('stress', 0)}%")
        print(f"  Energy: {biometrics.get('body_battery', 0)}%")
        
        # Cognitive state
        print(f"\n🎭 COGNITIVE")
        print(f"  Decision: {narration}")
        print(f"  Confidence: {decision.get('confidence', 0):.1%}")
        
        # Consciousness stream
        print(f"\n💭 INNER MONOLOGUE")
        print(f"  \"{thought}\"")
        
        print("\n" + "=" * 60)
    
    def _update_memories(self, state: Dict, decision: Dict):
        """Update both WM and RL memories"""
        # Store in working memory
        self.council.modules['memory_wm'].store(
            content={'state': state, 'decision': decision},
            relevance=decision.get('confidence', 0.5)
        )
        
        # Learn in RL
        reward = self._calculate_reward(state)
        self.council.modules['memory_rl'].learn_from_experience(
            state=str(state.get('location', 'unknown')),
            action=decision.get('decision', 'none'),
            reward=reward
        )
    
    def _calculate_reward(self, state: Dict) -> float:
        """Calculate reward based on outcomes"""
        # Simple reward: low stress + high energy = good
        stress = state.get('stress_level', 50) / 100
        energy = state.get('energy_level', 50) / 100
        
        return energy - stress
