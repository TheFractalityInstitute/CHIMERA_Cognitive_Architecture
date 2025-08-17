"""Embodied CHIMERA with real sensor grounding"""
import asyncio
from typing import Dict, Any
from chimera.core.chimera_core import CHIMERACore
from chimera.sensors.mobile_sensors import (
    MobileSensorHub, AccelerometerAgent, CircadianAgent, ProprioceptionAgent
)

class EmbodiedCHIMERA(CHIMERACore):
    """CHIMERA with real-world sensor embodiment"""
    
    def __init__(self):
        super().__init__()
        self.sensor_hub = MobileSensorHub()
        self.sensor_agents = {}
        
    async def initialize_embodiment(self):
        """Initialize sensor integration"""
        print("🧠 CHIMERA: Initializing embodied cognition...")
        
        # Create sensor agents
        self.sensor_agents = {
            'accelerometer': AccelerometerAgent(),
            'circadian': CircadianAgent(),
            'proprioception': ProprioceptionAgent(),
        }
        
        # Add to agent pool
        for agent in self.sensor_agents.values():
            self.add_agent(agent)
        
        # Start sensor streaming
        asyncio.create_task(self.sensor_hub.start_streaming())
        
        print("👁️ CHIMERA: Sensory systems online")
        print("🦾 CHIMERA: Body schema initializing...")
        
    async def run_embodied(self, duration=None):
        """Run with continuous sensor input"""
        await self.initialize_embodiment()
        
        # Sensor fusion loop
        async def sensor_fusion():
            while self.running:
                # Inject sensor data into message bus
                sensor_data = self.sensor_hub.sensor_cache.copy()
                
                for sensor_type, data in sensor_data.items():
                    message = NeuralMessage(
                        sender="sensor_hub",
                        content=data,
                        msg_type=MessageType.EXCITATORY,
                        priority=MessagePriority.HIGH,
                        timestamp=self.clock.get_sync_time(),
                        strength=1.0
                    )
                    await self.bus.publish(message)
                
                await asyncio.sleep(0.01)  # 100Hz fusion rate
        
        # Start fusion
        asyncio.create_task(sensor_fusion())
        
        # Run main loop
        await self.run(duration)
