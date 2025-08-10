"""Base classes for all CHIMERA modules"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass
import asyncio

class CognitiveAgent(ABC):
    """Base class for all agents in the ecosystem"""
    
    def __init__(self, agent_id: str, agent_type: str, tick_rate: float = 10.0):
        self.id = agent_id
        self.agent_type = agent_type
        self.tick_rate = tick_rate
        self.tick_period = 1.0 / tick_rate
        self.running = True
        
    @abstractmethod
    async def process(self, bus: 'DualBusSystem', context: Dict) -> None:
        """Process one tick - must be implemented by each agent"""
        pass
        
    async def run(self, bus: 'DualBusSystem', clock: 'PhaseLockedClock'):
        """Main run loop for the agent"""
        while self.running:
            try:
                # Sync with global clock
                phase = clock.get_current_phase(self.tick_rate)
                clock.register_agent_phase(self.id, phase)
                
                # Process this tick
                await self.process(bus, {'phase': phase})
                
                # Wait for next tick
                await asyncio.sleep(self.tick_period)
                
            except Exception as e:
                print(f"[{self.id}] Error: {e}")

@dataclass
class NeuralMessage:
    """Message passed between agents"""
    sender: str
    content: Any
    msg_type: str
    priority: float = 1.0
    timestamp: float = 0.0
    phase: float = 0.0