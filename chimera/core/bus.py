"""Communication bus for agent ecosystem"""

from collections import defaultdict, deque
from typing import List
import asyncio

class DualBusSystem:
    """Dual-bus architecture for agent communication"""
    
    def __init__(self):
        self.fast_bus = defaultdict(lambda: deque(maxlen=100))  # Urgent
        self.slow_bus = defaultdict(lambda: deque(maxlen=1000))  # Normal
        self.broadcast_channel = deque(maxlen=100)
        self.lock = asyncio.Lock()
        
    async def send(self, message, target: str = None, urgent: bool = False):
        """Send message to specific agent or broadcast"""
        async with self.lock:
            if target:
                bus = self.fast_bus if urgent else self.slow_bus
                bus[target].append(message)
            else:
                self.broadcast_channel.append(message)
                
    async def broadcast(self, message):
        """Broadcast to all agents"""
        async with self.lock:
            self.broadcast_channel.append(message)
            
    def get_messages_for(self, agent_id: str) -> List:
        """Get all messages for an agent"""
        messages = []
        
        # Check direct messages
        messages.extend(list(self.fast_bus[agent_id]))
        messages.extend(list(self.slow_bus[agent_id]))
        
        # Check broadcasts
        messages.extend(list(self.broadcast_channel))
        
        # Clear after reading
        self.fast_bus[agent_id].clear()
        self.slow_bus[agent_id].clear()
        
        return messages