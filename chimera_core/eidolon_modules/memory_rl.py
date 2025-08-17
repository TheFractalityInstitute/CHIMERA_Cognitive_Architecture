# chimera/eidolon_modules/memory_rl.py
"""
CHIMERA Reinforcement Learning Memory Eidolon Module v1.0
Slow, robust, unlimited capacity memory based on value learning
Based on Westbrook et al., 2025 findings on dopamine and RL
"""

import asyncio
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
from dataclasses import dataclass, field
import time

# Import from other CHIMERA modules
from chimera.core.message_bus import (
    NeuralMessage, 
    Neurotransmitter,
    MessagePriority,
    ModuleConnector
)

# ============= HELPER CLASSES =============
@dataclass
class StateActionValue:
    """Q-value for state-action pairs"""
    state: str
    action: str
    value: float = 0.0
    visits: int = 0
    last_updated: float = field(default_factory=time.time)
    
    def update(self, reward: float, learning_rate: float, discount: float = 0.95):
        """Update Q-value using TD learning"""
        self.value = self.value + learning_rate * (reward + discount * self.value - self.value)
        self.visits += 1
        self.last_updated = time.time()

@dataclass
class Episode:
    """Single episode of experience"""
    states: List[str]
    actions: List[str]
    rewards: List[float]
    timestamp: float = field(default_factory=time.time)
    
    def get_return(self, discount: float = 0.95) -> float:
        """Calculate discounted return"""
        total = 0
        for i, reward in enumerate(reversed(self.rewards)):
            total += reward * (discount ** i)
        return total

# ============= MAIN MODULE CLASS =============
class ReinforcementLearningEidolon:
    """
    Reinforcement Learning Memory Module - The 'Hard Drive' of CHIMERA
    Slow learning but unlimited capacity, reduced by low dopamine
    """
    
    def __init__(self, name: str = "RLMemory"):
        # Basic properties
        self.name = name
        self.role = "slow_robust_memory"
        
        # RL-specific parameters from Westbrook et al., 2025
        self.base_learning_rate = 0.1  # α_RL from paper
        self.discount_factor = 0.95    # γ for future rewards
        self.exploration_rate = 0.1    # ε for exploration
        self.effort_cost = 0.2          # Low effort (vs WM's high effort)
        
        # Unlimited capacity Q-table
        self.q_values = defaultdict(lambda: defaultdict(float))
        
        # Episode buffer for replay
        self.episode_buffer = []
        self.max_episodes = 1000  # Keep last 1000 episodes
        
        # Value estimates for objects/locations (like in the paper's task)
        self.stimulus_values = defaultdict(float)
        
        # Dopamine modulation
        self.dopamine_sensitivity = 2.0  # Stronger effect than on WM
        self.current_dopamine = 1.0
        
        # Track learning progress
        self.total_updates = 0
        self.recent_errors = []
        
        # Bus connection
        self.connector = None
        
    # ============= CORE INTERFACE METHODS =============
    async def initialize(self, bus_url: str = "ws://127.0.0.1:7860"):
        """Connect to the message bus"""
        self.connector = ModuleConnector(self.name, bus_url)
        await self.connector.connect()
        
    async def deliberate(self, topic: str) -> Dict[str, Any]:
        """
        Form opinion on topic from RL perspective
        Uses learned values rather than recent memories
        """
        # Look for learned values related to the topic
        relevant_values = self._search_values(topic)
        
        if relevant_values:
            best_option = max(relevant_values.items(), key=lambda x: x[1])
            opinion = f"Based on past experience, '{best_option[0]}' has value {best_option[1]:.2f}. "
            
            # Add exploration suggestion if values are low
            if best_option[1] < 0.5:
                opinion += "Values are low - suggest exploration."
                
            confidence = min(1.0, abs(best_option[1]))  # Higher values = higher confidence
        else:
            opinion = "No learned values for this situation. Need more experience."
            confidence = 0.2  # Low confidence without experience
            
        return {
            'module': self.name,
            'opinion': opinion,
            'confidence': confidence * self._get_learning_efficiency(),
            'reasoning': f"Based on {self.total_updates} learning updates",
            'learned_values': dict(relevant_values)
        }
        
    # ============= RL LEARNING METHODS =============
    def learn_from_experience(self, 
                             state: str, 
                             action: str, 
                             reward: float,
                             next_state: Optional[str] = None):
        """
        Core RL update using TD learning
        This is where slow, robust learning happens
        """
        # Adjust learning rate based on dopamine (Westbrook et al., 2025)
        effective_lr = self._get_effective_learning_rate()
        
        if next_state:
            # TD learning with bootstrapping
            max_next_value = max(self.q_values[next_state].values()) if self.q_values[next_state] else 0
            td_target = reward + self.discount_factor * max_next_value
            td_error = td_target - self.q_values[state][action]
            
            # Update Q-value
            self.q_values[state][action] += effective_lr * td_error
            
            # Track error for monitoring
            self.recent_errors.append(abs(td_error))
            if len(self.recent_errors) > 100:
                self.recent_errors.pop(0)
        else:
            # Terminal state - just use reward
            self.q_values[state][action] += effective_lr * (reward - self.q_values[state][action])
            
        self.total_updates += 1
        
    def learn_stimulus_value(self, stimulus: str, reward: float):
        """
        Learn values for specific stimuli (like objects in the paper's task)
        This is separate from state-action values
        """
        effective_lr = self._get_effective_learning_rate()
        
        # Simple running average update
        old_value = self.stimulus_values[stimulus]
        self.stimulus_values[stimulus] += effective_lr * (reward - old_value)
        
    def add_episode(self, states: List[str], actions: List[str], rewards: List[float]):
        """Store complete episode for replay learning"""
        episode = Episode(states, actions, rewards)
        self.episode_buffer.append(episode)
        
        # Limit buffer size
        if len(self.episode_buffer) > self.max_episodes:
            self.episode_buffer.pop(0)
            
        # Optionally trigger replay learning
        if len(self.episode_buffer) % 10 == 0:
            self._replay_learning()
            
    # ============= INTERNAL METHODS =============
    def _search_values(self, query: str) -> Dict[str, float]:
        """Search for learned values related to query"""
        relevant = {}
        query_lower = query.lower()
        
        # Search in Q-values
        for state, actions in self.q_values.items():
            if query_lower in state.lower():
                for action, value in actions.items():
                    relevant[f"{state}->{action}"] = value
                    
        # Search in stimulus values
        for stimulus, value in self.stimulus_values.items():
            if query_lower in stimulus.lower():
                relevant[stimulus] = value
                
        return relevant
        
    def _get_effective_learning_rate(self) -> float:
        """
        Calculate learning rate based on dopamine
        Key finding from Westbrook et al., 2025:
        Methylphenidate (dopamine boost) increases RL rate
        """
        base_rate = self.base_learning_rate
        
        # Dopamine strongly modulates RL learning rate
        if self.current_dopamine > 1.2:
            # High dopamine = faster RL (like methylphenidate in paper)
            return base_rate * (1 + (self.current_dopamine - 1) * self.dopamine_sensitivity)
        elif self.current_dopamine < 0.8:
            # Low dopamine = slower RL (like sulpiride in paper)
            return base_rate * self.current_dopamine
        else:
            return base_rate
            
    def _get_learning_efficiency(self) -> float:
        """Estimate how well the RL system is learning"""
        if not self.recent_errors:
            return 0.5
            
        # Lower errors = better learning
        avg_error = np.mean(self.recent_errors)
        efficiency = 1.0 / (1.0 + avg_error)
        
        # Modulate by dopamine
        efficiency *= (0.5 + 0.5 * self.current_dopamine)
        
        return np.clip(efficiency, 0, 1)
        
    def _replay_learning(self, n_replays: int = 5):
        """
        Replay past episodes to consolidate learning
        Similar to hippocampal replay during rest
        """
        if len(self.episode_buffer) < n_replays:
            return
            
        # Sample random episodes
        episodes = np.random.choice(self.episode_buffer, n_replays, replace=False)
        
        for episode in episodes:
            # Re-learn from the episode with current dopamine level
            for i in range(len(episode.states) - 1):
                self.learn_from_experience(
                    episode.states[i],
                    episode.actions[i],
                    episode.rewards[i],
                    episode.states[i + 1]
                )
                
    # ============= ACTION SELECTION =============
    def select_action(self, state: str, available_actions: List[str]) -> str:
        """
        Select action using ε-greedy policy
        Balance exploration vs exploitation
        """
        # Exploration rate modulated by dopamine
        effective_exploration = self.exploration_rate * (2 - self.current_dopamine)
        
        if np.random.random() < effective_exploration:
            # Explore: random action
            return np.random.choice(available_actions)
        else:
            # Exploit: best known action
            if state in self.q_values and self.q_values[state]:
                # Get values for available actions
                action_values = {a: self.q_values[state].get(a, 0) 
                               for a in available_actions}
                return max(action_values, key=action_values.get)
            else:
                # No knowledge - random
                return np.random.choice(available_actions)
                
    # ============= EXTERNAL INTERFACE =============
    def set_dopamine_level(self, level: float):
        """
        Modulate by dopamine (from Executive module)
        Has stronger effect on RL than on WM
        """
        self.current_dopamine = np.clip(level, 0.5, 2.0)
        
        # Dopamine affects exploration
        if level > 1.5:
            self.exploration_rate = 0.05  # Less exploration when confident
        elif level < 0.7:
            self.exploration_rate = 0.2   # More exploration when uncertain
        else:
            self.exploration_rate = 0.1
            
    def get_value(self, state: str, action: Optional[str] = None) -> float:
        """Get learned value for state or state-action pair"""
        if action:
            return self.q_values[state].get(action, 0)
        else:
            # Return max value for state
            return max(self.q_values[state].values()) if self.q_values[state] else 0
            
    def reset(self):
        """Clear all learned values"""
        self.q_values.clear()
        self.stimulus_values.clear()
        self.episode_buffer.clear()
        self.total_updates = 0
        self.recent_errors.clear()

# ============= STANDALONE TEST =============
if __name__ == "__main__":
    async def test_rl_memory():
        rl = ReinforcementLearningEidolon()
        
        # Simulate learning episode
        print("Learning from experience...")
        rl.learn_from_experience("intersection", "turn_left", reward=1.0, next_state="street_A")
        rl.learn_from_experience("intersection", "turn_right", reward=-0.5, next_state="dead_end")
        rl.learn_from_experience("street_A", "continue", reward=2.0)
        
        # Test deliberation
        opinion = await rl.deliberate("Which way at the intersection?")
        print(f"\nOpinion: {opinion['opinion']}")
        print(f"Confidence: {opinion['confidence']:.2f}")
        print(f"Learned values: {opinion['learned_values']}")
        
        # Test with high dopamine (faster learning)
        print("\n--- With high dopamine ---")
        rl.set_dopamine_level(1.5)
        rl.learn_from_experience("intersection", "turn_left", reward=1.5, next_state="street_A")
        
        opinion = await rl.deliberate("Which way at the intersection?")
        print(f"Updated opinion: {opinion['opinion']}")
        
        # Test action selection
        action = rl.select_action("intersection", ["turn_left", "turn_right", "go_straight"])
        print(f"\nSelected action: {action}")
        
    asyncio.run(test_rl_memory())
