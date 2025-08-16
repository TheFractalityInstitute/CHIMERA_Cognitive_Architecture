# chimera_core/__init__.py
"""
CHIMERA Cognitive Architecture
A distributed consciousness platform
"""
__version__ = "1.0.0"

from .sensors.chimera_complete import CHIMERAComplete
from .cognition.council import BiologicallyGroundedCouncil
from .collective.server import CHIMERACollectiveServer
from chimera.core.bus import DualBusSystem
from chimera.core.clock import PhaseLockedClock
from chimera.agents.crystallization import CrystallizationEngine
from chimera.agents.sensory import SensoryAgent
# ... import other agents

import asyncio

class CHIMERA:
    """Main orchestrator maintaining your multi-agent ecosystem"""
    
    def __init__(self):
        # Core infrastructure
        self.bus = DualBusSystem()
        self.clock = PhaseLockedClock()
        
        # Agent ecosystem - preserving your original design!
        self.agents = {
            'sensory_visual': SensoryAgent('visual'),
            'sensory_auditory': SensoryAgent('auditory'),
            'sensory_tactile': SensoryAgent('tactile'),
            'crystallization': CrystallizationEngine(),
            # ... all your other agents
        }
        
        self.running = False
        
    async def run(self):
        """Run the complete cognitive ecosystem"""
        self.running = True
        
        # Start all agents
        tasks = []
        for agent in self.agents.values():
            task = asyncio.create_task(
                agent.run(self.bus, self.clock)
            )
            tasks.append(task)
            
        # Let them run
        await asyncio.gather(*tasks)

__all__ = [
    'CHIMERAComplete',
    'BiologicallyGroundedCouncil', 
    'CHIMERACollectiveServer'
]
