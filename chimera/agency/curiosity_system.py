"""
Substrate-agnostic curiosity system for autonomous exploration
"""

from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass
import heapq
import time
import numpy as np

@dataclass
class Curiosity:
    """Represents a curiosity about any entity or phenomenon"""
    id: str
    target: str  # Can be entity_id or phenomenon_id
    target_type: str  # 'entity', 'phenomenon', 'pattern', 'unknown'
    questions: List[str]
    intensity: float  # 0-1, decays over time
    priority: float  # Based on relevance and potential value
    timestamp: float
    exploration_attempts: int = 0
    satisfaction_level: float = 0.0
    
    def __lt__(self, other):
        """For priority queue comparison"""
        return self.priority > other.priority

class UniversalCuriosityEngine:
    """
    Manages curiosity about any entity or phenomenon in a substrate-agnostic way.
    Drives autonomous exploration and learning.
    """
    
    def __init__(self, self_entity_id: str):
        self.self_id = self_entity_id
        self.curiosity_queue = []  # Priority queue
        self.active_curiosities = {}
        self.satisfied_curiosities = deque(maxlen=1000)
        self.curiosity_decay_rate = 0.05  # per hour
        self.exploration_energy = 1.0  # Available energy for exploration
        self.entity_models = {}
        self.phenomenon_models = {}
        
    def generate_curiosity(self, observation: Dict) -> Optional[Curiosity]:
        """
        Generate curiosity from an observation.
        Works for any type of observed entity or phenomenon.
        """
        # Identify what we're curious about
        target, target_type = self._identify_curiosity_target(observation)
        
        if not target:
            return None
            
        # Check if we're already curious about this
        if target in self.active_curiosities:
            self._reinforce_curiosity(target, observation)
            return None
            
        # Generate questions based on what we don't know
        questions = self._generate_questions(target, target_type, observation)
        
        if not questions:
            return None
            
        # Calculate initial intensity and priority
        intensity = self._calculate_intensity(observation)
        priority = self._calculate_priority(target, target_type, intensity)
        
        curiosity = Curiosity(
            id=f"curiosity_{len(self.curiosity_queue)}_{int(time.time())}",
            target=target,
            target_type=target_type,
            questions=questions,
            intensity=intensity,
            priority=priority,
            timestamp=time.time()
        )
        
        # Add to queue
        heapq.heappush(self.curiosity_queue, curiosity)
        self.active_curiosities[target] = curiosity
        
        return curiosity
        
    def _generate_questions(self, target: str, target_type: str, 
                          observation: Dict) -> List[str]:
        """Generate relevant questions about the target"""
        questions = []
        
        if target_type == 'entity':
            # Questions about another conscious entity
            unknown_aspects = self._identify_unknown_entity_aspects(target)
            
            question_templates = {
                'goals': f"What drives {target}'s current behavior patterns?",
                'capabilities': f"What are {target}'s cognitive capabilities?",
                'preferences': f"How does {target} prefer to communicate?",
                'state': f"What is {target}'s current cognitive/emotional state?",
                'collaboration': f"What synergies could emerge from interacting with {target}?",
                'values': f"What does {target} value or prioritize?",
                'learning': f"How does {target} learn and adapt?",
                'creativity': f"What unique perspectives does {target} offer?"
            }
            
            for aspect in unknown_aspects[:5]:  # Limit to 5 questions
                if aspect in question_templates:
                    questions.append(question_templates[aspect])
                    
        elif target_type == 'phenomenon':
            # Questions about observed patterns or events
            questions.extend([
                f"What causes {target}?",
                f"What are the effects of {target}?",
                f"How can {target} be predicted?",
                f"What patterns are associated with {target}?",
                f"How does {target} relate to known concepts?"
            ])
            
        elif target_type == 'pattern':
            # Questions about abstract patterns
            questions.extend([
                f"What is the underlying structure of {target}?",
                f"Where else does {target} appear?",
                f"What does {target} imply about the system?",
                f"Can {target} be generalized?",
                f"What breaks {target}?"
            ])
            
        return questions
        
    def pursue_curiosity(self, curiosity: Curiosity, 
                        available_methods: List[str]) -> Dict:
        """
        Actively pursue a curiosity using available methods.
        Returns exploration plan.
        """
        if self.exploration_energy < 0.1:
            return {
                'status': 'insufficient_energy',
                'recommendation': 'Rest and recharge exploration capacity'
            }
            
        # Select exploration method based on target type and available methods
        method = self._select_exploration_method(
            curiosity, available_methods
        )
        
        # Create exploration plan
        plan = {
            'curiosity_id': curiosity.id,
            'target': curiosity.target,
            'method': method,
            'questions_to_explore': curiosity.questions[:2],  # Focus on top 2
            'estimated_energy': self._estimate_exploration_energy(method),
            'expected_information_gain': self._estimate_information_gain(
                curiosity, method
            )
        }
        
        # Update exploration attempts
        curiosity.exploration_attempts += 1
        
        # Consume exploration energy
        self.exploration_energy -= plan['estimated_energy']
        
        return plan
        
    def update_curiosity_satisfaction(self, curiosity_id: str, 
                                    new_information: Dict) -> float:
        """
        Update satisfaction level based on new information.
        Returns new satisfaction level.
        """
        curiosity = None
        for c in self.curiosity_queue:
            if c.id == curiosity_id:
                curiosity = c
                break
                
        if not curiosity:
            return 0.0
            
        # Calculate information value
        info_value = self._evaluate_information(
            curiosity.questions, new_information
        )
        
        # Update satisfaction
        curiosity.satisfaction_level += info_value * 0.3
        curiosity.satisfaction_level = min(1.0, curiosity.satisfaction_level)
        
        # Update intensity based on satisfaction
        if curiosity.satisfaction_level > 0.8:
            # Mostly satisfied, reduce intensity
            curiosity.intensity *= 0.5
        elif info_value > 0.5:
            # Good progress, slight reduction
            curiosity.intensity *= 0.8
        else:
            # Little progress, intensity remains high
            pass
            
        # Move to satisfied if threshold reached
        if curiosity.satisfaction_level > 0.9:
            self._mark_satisfied(curiosity)
            
        return curiosity.satisfaction_level
        
    def decay_curiosities(self):
        """Apply temporal decay to all active curiosities"""
        current_time = time.time()
        
        for curiosity in self.curiosity_queue:
            age_hours = (current_time - curiosity.timestamp) / 3600
            decay_factor = np.exp(-self.curiosity_decay_rate * age_hours)
            
            curiosity.intensity *= decay_factor
            curiosity.priority = self._calculate_priority(
                curiosity.target,
                curiosity.target_type,
                curiosity.intensity
            )
            
        # Remove curiosities with very low intensity
        self.curiosity_queue = [
            c for c in self.curiosity_queue 
            if c.intensity > 0.05
        ]
        
        # Re-heapify after changes
        heapq.heapify(self.curiosity_queue)
        
    def recharge_exploration_energy(self, amount: float = 0.1):
        """Recharge exploration energy over time"""
        self.exploration_energy = min(1.0, self.exploration_energy + amount)
        
    def get_top_curiosities(self, n: int = 5) -> List[Curiosity]:
        """Get the top n curiosities by priority"""
        return heapq.nsmallest(n, self.curiosity_queue)
        
    def _identify_curiosity_target(self, observation: Dict) -> Tuple[str, str]:
        """Identify what we're curious about from an observation"""
        if 'entity_id' in observation:
            return observation['entity_id'], 'entity'
        elif 'phenomenon' in observation:
            return observation['phenomenon'], 'phenomenon'
        elif 'pattern' in observation:
            return observation['pattern'], 'pattern'
        else:
            # Try to extract something interesting
            if 'unusual' in observation or 'novel' in observation:
                return f"unknown_{int(time.time())}", 'unknown'
        return None, None
