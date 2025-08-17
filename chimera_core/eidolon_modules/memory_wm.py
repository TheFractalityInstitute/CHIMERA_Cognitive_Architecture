# chimera/eidolon_modules/memory_wm.py
"""
CHIMERA Working Memory Eidolon Module v1.0
Fast, flexible, but limited capacity memory
Based on Westbrook et al., 2025 findings on dopamine and WM
"""

# ============= IMPORTS AT THE TOP =============
import asyncio
import numpy as np
from typing import Dict, List, Optional, Any
from collections import deque
from dataclasses import dataclass, field

# Import from other CHIMERA modules
from chimera.core.message_bus import (
    NeuralMessage, 
    Neurotransmitter,
    MessagePriority,
    ModuleConnector
)

# ============= HELPER CLASSES/DATACLASSES =============
@dataclass
class MemoryItem:
    """Single item in working memory"""
    content: Any
    timestamp: float
    relevance: float = 1.0
    decay_rate: float = 0.1
    
    def get_activation(self, current_time: float) -> float:
        """Calculate current activation level"""
        age = current_time - self.timestamp
        return self.relevance * np.exp(-self.decay_rate * age)

# ============= MAIN MODULE CLASS =============
class WorkingMemoryEidolon:
    """
    Working Memory Module - The 'RAM' of CHIMERA
    Fast access but limited capacity, enhanced by dopamine
    """
    
    def __init__(self, name: str = "WorkingMemory"):
        # Basic properties
        self.name = name
        self.role = "fast_flexible_memory"
        
        # WM-specific parameters from Westbrook paper
        self.capacity = 7  # Miller's magical number
        self.decay_rate = 0.1  # How fast memories fade
        self.effort_cost = 0.7  # High effort to maintain
        
        # The actual memory store
        self.memory_buffer = deque(maxlen=self.capacity)
        
        # Dopamine modulation parameters
        self.dopamine_sensitivity = 1.5
        self.current_dopamine = 1.0
        
        # Bus connection (will be initialized later)
        self.connector = None
        
    # ============= CORE INTERFACE METHODS =============
    async def initialize(self, bus_url: str = "ws://127.0.0.1:7860"):
        """Connect to the message bus"""
        self.connector = ModuleConnector(self.name, bus_url)
        await self.connector.connect()
        
    async def deliberate(self, topic: str) -> Dict[str, Any]:
        """
        Form opinion on topic from WM perspective
        This is called by the Council during democratic decision-making
        """
        # Search working memory for relevant items
        relevant_items = self._search_memory(topic)
        
        if relevant_items:
            opinion = f"I have {len(relevant_items)} relevant memories. "
            opinion += f"Most relevant: {relevant_items[0].content}"
            confidence = min(1.0, len(relevant_items) / 3)  # More memories = higher confidence
        else:
            opinion = "No relevant information in working memory."
            confidence = 0.1
            
        return {
            'module': self.name,
            'opinion': opinion,
            'confidence': confidence * self._get_effective_capacity(),
            'reasoning': f"Working at {len(self.memory_buffer)}/{self.capacity} capacity",
            'memory_items': relevant_items
        }
        
    # ============= INTERNAL METHODS =============
    def _search_memory(self, query: str) -> List[MemoryItem]:
        """Search WM for relevant items"""
        relevant = []
        current_time = asyncio.get_event_loop().time()
        
        for item in self.memory_buffer:
            # Check relevance and activation
            if query.lower() in str(item.content).lower():
                if item.get_activation(current_time) > 0.3:
                    relevant.append(item)
                    
        return sorted(relevant, key=lambda x: x.get_activation(current_time), reverse=True)
        
    def _get_effective_capacity(self) -> float:
        """
        Compute effective WM capacity based on dopamine
        High dopamine = higher effective capacity (Westbrook et al., 2025)
        """
        base_capacity = len(self.memory_buffer) / self.capacity
        dopamine_boost = 1 + (self.current_dopamine - 1.0) * self.dopamine_sensitivity
        return min(1.0, base_capacity * dopamine_boost)
        
    # ============= EXTERNAL INTERFACE =============
    def store(self, content: Any, relevance: float = 1.0):
        """Store item in working memory"""
        item = MemoryItem(
            content=content,
            timestamp=asyncio.get_event_loop().time(),
            relevance=relevance,
            decay_rate=self.decay_rate / self.current_dopamine  # Dopamine slows decay
        )
        self.memory_buffer.append(item)
        
    def clear(self):
        """Clear working memory"""
        self.memory_buffer.clear()
        
    def set_dopamine_level(self, level: float):
        """Modulate by dopamine (from Executive module)"""
        self.current_dopamine = np.clip(level, 0.5, 2.0)
        
        # Dopamine affects capacity and decay
        if level > 1.2:
            self.decay_rate = 0.05  # Slower decay when high dopamine
        else:
            self.decay_rate = 0.15  # Faster decay when low dopamine

# ============= STANDALONE TEST =============
if __name__ == "__main__":
    # This only runs if you execute this file directly
    # Good for testing the module in isolation
    
    async def test_working_memory():
        wm = WorkingMemoryEidolon()
        
        # Store some items
        wm.store("The cat is on the mat", relevance=0.8)
        wm.store("Paris is the capital of France", relevance=1.0)
        wm.store("2 + 2 = 4", relevance=0.5)
        
        # Test deliberation
        opinion = await wm.deliberate("What do you know about France?")
        print(f"Opinion: {opinion}")
        
        # Test dopamine modulation
        wm.set_dopamine_level(1.5)
        print(f"With high dopamine: {wm._get_effective_capacity()}")
        
    # Run the test
    asyncio.run(test_working_memory())
