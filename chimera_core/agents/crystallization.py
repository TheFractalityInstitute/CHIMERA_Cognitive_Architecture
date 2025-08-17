"""The Crystallization Engine - preserves insights!"""

from chimera.core.base import CognitiveAgent, NeuralMessage
from dataclasses import dataclass
from typing import Dict, List, Optional
import time
import hashlib

@dataclass
class CrystallizedInsight:
    """Represents a crystallized moment of understanding"""
    id: str
    timestamp: float
    content: Dict
    linguistic_expression: str
    confidence: float
    verification_count: int = 0
    groundings: List[Dict] = None

class CrystallizationEngine(CognitiveAgent):
    """Captures and verifies insights across the system"""
    
    def __init__(self):
        super().__init__("crystallization_engine", "meta", tick_rate=0.5)
        self.insights = {}
        self.resonance_threshold = 0.85
        self.pending_crystallizations = []
        
    async def process(self, bus, context: Dict):
        """Check for crystallization opportunities"""
        
        # Monitor bus for high-resonance patterns
        messages = bus.get_messages_for(self.id)
        
        for msg in messages:
            if self._check_resonance(msg):
                insight = self._crystallize(msg)
                if insight:
                    self.insights[insight.id] = insight
                    
                    # Broadcast crystallized insight
                    await bus.broadcast(NeuralMessage(
                        sender=self.id,
                        content={'insight': insight},
                        msg_type='crystallized',
                        priority=0.9
                    ))
                    
    def _check_resonance(self, msg: NeuralMessage) -> bool:
        """Check if message has high enough resonance to crystallize"""
        # Check phase coherence across agents
        if hasattr(msg, 'phase'):
            # High coherence = potential insight
            return msg.phase > self.resonance_threshold
        return False
        
    def _crystallize(self, msg: NeuralMessage) -> Optional[CrystallizedInsight]:
        """Transform high-resonance pattern into crystallized insight"""
        
        insight_id = hashlib.md5(
            f"{msg.content}{time.time()}".encode()
        ).hexdigest()[:12]
        
        return CrystallizedInsight(
            id=insight_id,
            timestamp=time.time(),
            content=msg.content,
            linguistic_expression=self._generate_expression(msg.content),
            confidence=0.7,
            groundings=[msg.content]
        )
        
    def _generate_expression(self, content: Dict) -> str:
        """Generate linguistic expression of insight"""
        # This would be enhanced by the LanguageAgent
        return f"Pattern recognized: {content.get('type', 'unknown')}"