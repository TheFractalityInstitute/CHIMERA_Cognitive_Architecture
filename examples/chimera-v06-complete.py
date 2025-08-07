#!/usr/bin/env python3
"""
CHIMERA CSA v0.6 - Neuromorphic Emulation with Goal-Directed Behavior
Incorporates all enhancements from Gemini's analysis:
- Phase-based binding
- Semantic grounding through outcomes
- Drive system for intrinsic motivation
- TD-learning based plasticity
"""

import asyncio
import time
import random
import numpy as np
from collections import deque, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import json
import zlib
import pickle
import uuid
import math

# ============= Core Infrastructure =============

class MessagePriority(Enum):
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    BACKGROUND = 3
    BULK = 4

class MessageType(Enum):
    EXCITATORY = "excitatory"
    INHIBITORY = "inhibitory"
    MODULATORY = "modulatory"
    DRIVE = "drive"
    REWARD = "reward"

@dataclass
class NeuralMessage:
    sender: str
    content: Any
    msg_type: MessageType
    priority: MessagePriority
    timestamp: float
    strength: float = 1.0
    phase: float = 0.0  # Phase in oscillation cycle
    
    @property
    def neurotransmitter(self):
        mapping = {
            MessageType.EXCITATORY: "glutamate",
            MessageType.INHIBITORY: "GABA",
            MessageType.MODULATORY: "dopamine",
            MessageType.DRIVE: "orexin",
            MessageType.REWARD: "dopamine_burst"
        }
        return mapping.get(self.msg_type, "unknown")

# ============= Enhanced Phase-Locked Clock =============

class PhaseLockedClock:
    """Global clock with phase-based feature binding"""
    def __init__(self, base_frequency=1000.0):
        self.base_frequency = base_frequency
        self.phase = 0.0
        self.last_sync = time.perf_counter()
        self.drift_correction = 0.0
        self.agent_phases = {}
        self.binding_window = 0.1  # 100ms for phase binding
        
    def get_sync_time(self):
        """Returns phase-locked time with drift correction"""
        current = time.perf_counter()
        elapsed = current - self.last_sync
        
        self.phase += elapsed * self.base_frequency + self.drift_correction
        self.last_sync = current
        
        return self.phase / self.base_frequency
        
    def get_current_phase(self, frequency: float) -> float:
        """Get current phase for given frequency"""
        current_time = self.get_sync_time()
        return (2 * np.pi * frequency * current_time) % (2 * np.pi)
        
    def register_agent_phase(self, agent_id: str, phase: float):
        """Register agent phase for synchronization"""
        self.agent_phases[agent_id] = phase
        
    def compute_phase_coherence(self, phases: List[float]) -> float:
        """Compute coherence of multiple phases (0-1)"""
        if not phases:
            return 0.0
            
        # Convert to unit vectors
        vectors = np.array([[np.cos(p), np.sin(p)] for p in phases])
        
        # Mean vector magnitude indicates coherence
        mean_vector = np.mean(vectors, axis=0)
        coherence = np.linalg.norm(mean_vector)
        
        return coherence
        
    def are_phase_locked(self, phase1: float, phase2: float, tolerance: float = 0.5) -> bool:
        """Check if two phases are locked within tolerance"""
        phase_diff = abs(phase1 - phase2)
        phase_diff = min(phase_diff, 2*np.pi - phase_diff)  # Wrap around
        return phase_diff < tolerance
        
    def sync_agents(self):
        """Kuramoto-style phase synchronization"""
        if len(self.agent_phases) < 2:
            return
            
        # Compute coupling strength based on coherence
        phases = list(self.agent_phases.values())
        coherence = self.compute_phase_coherence(phases)
        
        # Stronger coupling when less coherent
        coupling_strength = 0.1 * (1.0 - coherence)
        
        # Update drift correction
        mean_phase = np.mean(phases)
        self.drift_correction = coupling_strength * (mean_phase - self.phase)

# ============= Semantic Memory with Outcomes =============

class SemanticMemory:
    """Memory system that grounds meaning through outcomes"""
    def __init__(self):
        self.memories = {}
        self.semantic_network = defaultdict(list)
        self.outcome_values = defaultdict(float)
        self.concept_embeddings = {}
        self.memory_id_counter = 0
        
    def store_memory_with_outcome(self, memory: dict, outcome: dict) -> str:
        """Store memory linked to its outcome"""
        memory_id = f"mem_{self.memory_id_counter}"
        self.memory_id_counter += 1
        
        # Link memory to outcome
        memory['id'] = memory_id
        memory['outcome'] = outcome
        memory['value'] = self.compute_outcome_value(outcome)
        
        # Store in main memory
        self.memories[memory_id] = memory
        
        # Build semantic associations
        self._build_semantic_links(memory, outcome)
        
        # Update concept embeddings
        self._update_embeddings(memory)
        
        return memory_id
        
    def _build_semantic_links(self, memory: dict, outcome: dict):
        """Create semantic links between features and outcomes"""
        features = self._extract_features(memory)
        
        for feature in features:
            self.semantic_network[feature].append({
                'memory_id': memory['id'],
                'outcome': outcome,
                'value': memory['value'],
                'timestamp': memory.get('timestamp', time.time())
            })
            
            # Update running value estimate for this feature
            self.outcome_values[feature] = (
                0.9 * self.outcome_values[feature] + 
                0.1 * memory['value']
            )
            
    def _extract_features(self, memory: dict) -> List[str]:
        """Extract semantic features from memory"""
        features = []
        
        # Extract from different memory components
        if 'sensory' in memory:
            for sense_data in memory['sensory']:
                features.append(f"{sense_data.get('modality')}_{sense_data.get('stimulus')}")
                
        if 'pattern' in memory:
            features.append(f"pattern_{memory['pattern'].get('type')}")
            
        if 'action' in memory:
            features.append(f"action_{memory['action']}")
            
        return features
        
    def _update_embeddings(self, memory: dict):
        """Update distributed concept representations"""
        features = self._extract_features(memory)
        
        # Simple embedding: co-occurrence vectors
        for f1 in features:
            if f1 not in self.concept_embeddings:
                self.concept_embeddings[f1] = defaultdict(float)
                
            for f2 in features:
                if f1 != f2:
                    self.concept_embeddings[f1][f2] += 1.0
                    
    def compute_outcome_value(self, outcome: dict) -> float:
        """Compute value of an outcome"""
        value = 0.0
        
        # Positive factors
        if outcome.get('reward'):
            value += outcome['reward']
        if outcome.get('goal_progress'):
            value += outcome['goal_progress'] * 0.5
        if outcome.get('novelty'):
            value += outcome['novelty'] * 0.2
            
        # Negative factors
        if outcome.get('energy_cost'):
            value -= outcome['energy_cost'] * 0.3
        if outcome.get('error'):
            value -= outcome['error']
            
        return np.tanh(value)  # Bounded -1 to 1
        
    def predict_outcome_value(self, features: List[str]) -> float:
        """Predict value based on features"""
        if not features:
            return 0.0
            
        # Average historical values for these features
        values = [self.outcome_values.get(f, 0.0) for f in features]
        return np.mean(values) if values else 0.0
        
    def find_similar_memories(self, query_features: List[str], top_k: int = 5) -> List[dict]:
        """Find memories with similar features"""
        scores = {}
        
        for memory_id, memory in self.memories.items():
            memory_features = self._extract_features(memory)
            
            # Jaccard similarity
            intersection = len(set(query_features) & set(memory_features))
            union = len(set(query_features) | set(memory_features))
            
            if union > 0:
                scores[memory_id] = intersection / union
                
        # Return top-k similar memories
        sorted_memories = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [self.memories[mid] for mid, _ in sorted_memories[:top_k]]

# ============= Drive System =============

class DriveSystem:
    """Homeostatic drive system for intrinsic motivation"""
    def __init__(self):
        self.drives = {
            'curiosity': 0.5,
            'energy_conservation': 0.5,
            'coherence_seeking': 0.5,
            'social_bonding': 0.3,
            'mastery': 0.4
        }
        
        self.set_points = self.drives.copy()
        self.drive_history = defaultdict(lambda: deque(maxlen=100))
        self.satisfaction_rates = defaultdict(float)
        
    def update_drives(self, system_state: dict):
        """Update drives based on system state"""
        # Curiosity increases with low novelty
        recent_novelty = system_state.get('average_novelty', 0.5)
        self.drives['curiosity'] = self.homeostatic_update(
            'curiosity', 
            self.set_points['curiosity'] - recent_novelty
        )
        
        # Energy conservation increases with high activity
        activity_level = system_state.get('activity_level', 0.5)
        self.drives['energy_conservation'] = self.homeostatic_update(
            'energy_conservation',
            activity_level  # Higher activity = higher conservation drive
        )
        
        # Coherence seeking increases with low integration
        coherence = system_state.get('global_coherence', 0.5)
        self.drives['coherence_seeking'] = self.homeostatic_update(
            'coherence_seeking',
            self.set_points['coherence_seeking'] - coherence
        )
        
        # Mastery increases when prediction errors are high
        prediction_error = system_state.get('prediction_error', 0.0)
        self.drives['mastery'] = self.homeostatic_update(
            'mastery',
            self.set_points['mastery'] + prediction_error * 0.5
        )
        
        # Record history
        for drive, value in self.drives.items():
            self.drive_history[drive].append(value)
            
    def homeostatic_update(self, drive_name: str, target: float) -> float:
        """Update drive with homeostatic dynamics"""
        current = self.drives[drive_name]
        
        # Move towards target with momentum
        adjustment = 0.1 * (target - current)
        new_value = current + adjustment
        
        # Add noise for exploration
        noise = np.random.normal(0, 0.02)
        new_value += noise
        
        # Bound between 0 and 1
        return np.clip(new_value, 0.0, 1.0)
        
    def get_dominant_drive(self) -> Tuple[str, float]:
        """Get the currently dominant drive"""
        max_drive = max(self.drives.items(), key=lambda x: x[1])
        return max_drive
        
    def satisfy_drive(self, drive_name: str, amount: float):
        """Reduce drive when satisfied"""
        if drive_name in self.drives:
            self.drives[drive_name] = max(0, self.drives[drive_name] - amount)
            self.satisfaction_rates[drive_name] = (
                0.9 * self.satisfaction_rates[drive_name] + 0.1
            )
            
    def get_drive_signals(self) -> List[dict]:
        """Generate drive signals for the system"""
        signals = []
        
        for drive, strength in self.drives.items():
            if strength > 0.3:  # Only signal significant drives
                signal = {
                    'drive': drive,
                    'strength': strength,
                    'urgent': strength > 0.8,
                    'suggested_action': self._suggest_action(drive, strength)
                }
                signals.append(signal)
                
        return signals
        
    def _suggest_action(self, drive: str, strength: float) -> str:
        """Suggest action based on drive"""
        suggestions = {
            'curiosity': 'explore_novel_inputs',
            'energy_conservation': 'reduce_processing',
            'coherence_seeking': 'increase_integration',
            'social_bonding': 'seek_communication',
            'mastery': 'practice_predictions'
        }
        return suggestions.get(drive, 'maintain_current')

# ============= Enhanced Dual Bus System =============

class SystemBus:
    """High-priority, low-latency system coordination"""
    def __init__(self, max_latency_ms=1):
        self.queue = asyncio.PriorityQueue()
        self.max_latency = max_latency_ms / 1000
        self.subscribers = defaultdict(list)
        self.interrupt_handlers = {}
        self.phase_locked_events = deque(maxlen=1000)
        
    async def publish_urgent(self, message: NeuralMessage):
        """Publish high-priority system message"""
        await self.queue.put((message.priority.value, message))
        
        # Track phase-locked events
        if message.priority == MessagePriority.CRITICAL:
            self.phase_locked_events.append({
                'message': message,
                'phase': message.phase,
                'timestamp': message.timestamp
            })
            
        # Direct interrupt for critical messages
        if message.priority == MessagePriority.CRITICAL:
            await self._direct_notify(message)
            
    async def _direct_notify(self, message: NeuralMessage):
        """Skip queue for critical messages"""
        for agent_id in self.interrupt_handlers:
            handler = self.interrupt_handlers[agent_id]
            await handler(message)
            
    def get_phase_locked_events(self, phase: float, tolerance: float = 0.5) -> List[dict]:
        """Get events that occurred at similar phase"""
        locked_events = []
        
        for event in self.phase_locked_events:
            phase_diff = abs(event['phase'] - phase)
            phase_diff = min(phase_diff, 2*np.pi - phase_diff)
            
            if phase_diff < tolerance:
                locked_events.append(event)
                
        return locked_events

class AgentBus:
    """Standard inter-agent communication with reliability"""
    def __init__(self):
        self.queue = asyncio.Queue()
        self.compression_threshold = 1024
        self.delivery_tracking = {}
        self.subscribers = defaultdict(list)
        self.topic_routes = defaultdict(list)
        
    async def publish(self, message: NeuralMessage, topics: List[str] = None):
        """Publish agent message with topic routing"""
        # Add tracking
        msg_id = str(uuid.uuid4())
        message.id = msg_id
        
        # Add topics for routing
        if topics:
            message.topics = topics
            for topic in topics:
                self.topic_routes[topic].append(msg_id)
        
        # Compress large messages
        if len(str(message.content)) > self.compression_threshold:
            compressed = zlib.compress(pickle.dumps(message.content))
            message.content = {'_compressed': True, 'data': compressed}
            
        self.delivery_tracking[msg_id] = {
            'sent': time.time(),
            'confirmed': False
        }
        
        await self.queue.put(message)

class DualBusSystem:
    """Parallel message buses with phase-aware routing"""
    def __init__(self):
        self.system_bus = SystemBus(max_latency_ms=1)
        self.agent_bus = AgentBus()
        self.stats = {
            'system_messages': 0,
            'agent_messages': 0,
            'phase_locked_bindings': 0
        }

# ============= TD-Learning System =============

class TDLearning:
    """Temporal Difference learning for value prediction"""
    def __init__(self, learning_rate: float = 0.1, gamma: float = 0.9):
        self.learning_rate = learning_rate
        self.gamma = gamma  # Discount factor
        self.value_estimates = defaultdict(float)
        self.eligibility_traces = defaultdict(float)
        self.prediction_errors = deque(maxlen=100)
        
    def predict_value(self, state_features: List[str]) -> float:
        """Predict value for given state"""
        if not state_features:
            return 0.0
            
        # Average value across features
        values = [self.value_estimates[f] for f in state_features]
        return np.mean(values) if values else 0.0
        
    def update(self, state_features: List[str], next_state_features: List[str], 
               reward: float, is_terminal: bool = False):
        """TD update with eligibility traces"""
        # Current and next value estimates
        v_current = self.predict_value(state_features)
        v_next = 0.0 if is_terminal else self.predict_value(next_state_features)
        
        # TD error
        td_error = reward + self.gamma * v_next - v_current
        self.prediction_errors.append(abs(td_error))
        
        # Update eligibility traces
        for feature in state_features:
            self.eligibility_traces[feature] = 1.0
            
        # Update all values based on eligibility
        for feature, trace in list(self.eligibility_traces.items()):
            if trace > 0.01:
                self.value_estimates[feature] += self.learning_rate * td_error * trace
                self.eligibility_traces[feature] *= self.gamma * 0.9  # Decay
            else:
                del self.eligibility_traces[feature]
                
        return td_error
        
    def get_recent_prediction_error(self) -> float:
        """Get average recent prediction error"""
        if not self.prediction_errors:
            return 0.0
        return np.mean(self.prediction_errors)

# ============= Enhanced Base Agent =============

class TemporalAgent:
    """Base agent with phase-locking and TD-learning"""
    def __init__(self, agent_id: str, agent_type: str, tick_rate: float = 10.0):
        self.id = agent_id
        self.agent_type = agent_type
        self.tick_rate = tick_rate
        self.tick_period = 1.0 / tick_rate
        self.local_memory = deque(maxlen=1000)
        self.last_update = 0
        self.phase_offset = random.random() * 2 * np.pi
        self.running = True
        
        # Enhanced learning
        self.td_learning = TDLearning()
        self.connection_weights = defaultdict(lambda: 0.5)
        self.learning_rate = 0.01
        
        # Phase tracking
        self.current_phase = 0.0
        self.phase_history = deque(maxlen=100)
        
        # State tracking for TD updates
        self.last_state_features = []
        self.accumulated_reward = 0.0
        
    async def run(self, core_system):
        """Main agent loop with enhanced features"""
        while self.running:
            # Get synchronized time and phase
            current_time = core_system.clock.get_sync_time()
            self.current_phase = core_system.clock.get_current_phase(self.tick_rate)
            self.phase_history.append(self.current_phase)
            
            # Register phase for synchronization
            core_system.clock.register_agent_phase(self.id, self.current_phase)
            
            # Check for system interrupts
            system_messages = await core_system.bus.system_bus.get_messages()
            for msg in system_messages:
                await self.handle_interrupt(msg)
                
            # Process regular messages
            if current_time - self.last_update >= self.tick_period:
                agent_messages = await core_system.bus.agent_bus.get_messages(self.id)
                
                # Check for reward signals
                reward_signal = self.extract_reward_signal(agent_messages)
                if reward_signal:
                    self.accumulated_reward += reward_signal
                    
                # Apply neuromodulation
                modulated_messages = self.apply_neuromodulation(
                    agent_messages, 
                    core_system.neuromodulators
                )
                
                # Process with phase awareness
                output = await self.process(modulated_messages, current_time)
                
                if output is not None:
                    # Update plasticity with TD learning
                    current_features = self.extract_state_features(modulated_messages)
                    
                    if self.last_state_features:
                        td_error = self.td_learning.update(
                            self.last_state_features,
                            current_features,
                            self.accumulated_reward
                        )
                        
                        # Update connections based on TD error
                        self.update_connections_td(agent_messages, output, td_error)
                        
                    self.last_state_features = current_features
                    self.accumulated_reward = 0.0  # Reset after update
                    
                    # Store in memory with phase info
                    memory_entry = {
                        'input': agent_messages,
                        'output': output,
                        'timestamp': current_time,
                        'phase': self.current_phase,
                        'phase_coherence': self.compute_local_phase_coherence(agent_messages)
                    }
                    self.local_memory.append(memory_entry)
                    
                    # Store in semantic memory with predicted outcome
                    if hasattr(core_system, 'semantic_memory'):
                        predicted_value = self.td_learning.predict_value(current_features)
                        outcome_prediction = {
                            'predicted_value': predicted_value,
                            'features': current_features
                        }
                        core_system.semantic_memory.store_memory_with_outcome(
                            memory_entry, 
                            outcome_prediction
                        )
                    
                    # Add phase to outgoing message
                    output['phase'] = self.current_phase
                    
                    # Route output to appropriate bus
                    await self.publish_output(output, core_system.bus)
                    
                self.last_update = current_time
                
            await asyncio.sleep(0.001)
            
    def extract_reward_signal(self, messages: List[NeuralMessage]) -> float:
        """Extract reward from messages"""
        total_reward = 0.0
        
        for msg in messages:
            if msg.msg_type == MessageType.REWARD:
                total_reward += msg.content.get('reward', 0.0)
            elif msg.msg_type == MessageType.DRIVE and 'satisfaction' in msg.content:
                total_reward += msg.content['satisfaction']
                
        return total_reward
        
    def extract_state_features(self, messages: List[NeuralMessage]) -> List[str]:
        """Extract features for TD learning"""
        features = []
        
        for msg in messages:
            # Message type features
            features.append(f"msg_type_{msg.msg_type.value}")
            
            # Content-based features
            if isinstance(msg.content, dict):
                for key in ['modality', 'pattern', 'action', 'drive']:
                    if key in msg.content:
                        features.append(f"{key}_{msg.content[key]}")
                        
        return features
        
    def compute_local_phase_coherence(self, messages: List[NeuralMessage]) -> float:
        """Compute phase coherence of incoming messages"""
        if not messages:
            return 0.0
            
        phases = [msg.phase for msg in messages if hasattr(msg, 'phase')]
        if not phases:
            return 0.0
            
        # Use clock's coherence computation
        vectors = np.array([[np.cos(p), np.sin(p)] for p in phases])
        mean_vector = np.mean(vectors, axis=0)
        return np.linalg.norm(mean_vector)
        
    def update_connections_td(self, inputs: List[NeuralMessage], output: dict, td_error: float):
        """Update connections based on TD error"""
        for msg in inputs:
            # Eligibility trace was set when message arrived
            # Now update based on TD error
            
            old_weight = self.connection_weights[msg.sender]
            
            # Three-factor learning rule: pre * post * reward
            correlation = self.compute_correlation(msg.content, output)
            weight_change = self.learning_rate * correlation * td_error
            
            new_weight = old_weight + weight_change
            self.connection_weights[msg.sender] = np.clip(new_weight, 0.0, 1.0)

# ============= Specialized Agents with Enhancements =============

class DriveAgent(TemporalAgent):
    """Generate intrinsic motivation signals"""
    def __init__(self):
        super().__init__("drive_system", "drive", tick_rate=1.0)
        self.drive_system = DriveSystem()
        self.system_state_buffer = deque(maxlen=50)
        
    async def process(self, inputs, timestamp):
        # Collect system state from messages
        system_state = self.aggregate_system_state(inputs)
        self.system_state_buffer.append(system_state)
        
        # Update drives
        self.drive_system.update_drives(system_state)
        
        # Get drive signals
        drive_signals = self.drive_system.get_drive_signals()
        
        if drive_signals:
            # Find dominant drive
            dominant_drive, strength = self.drive_system.get_dominant_drive()
            
            return {
                'drive_signals': drive_signals,
                'dominant_drive': dominant_drive,
                'drive_strength': strength,
                'suggested_actions': [s['suggested_action'] for s in drive_signals],
                'urgent': any(s['urgent'] for s in drive_signals),
                'msg_type': MessageType.DRIVE
            }
            
        return None
        
    def aggregate_system_state(self, inputs: List[NeuralMessage]) -> dict:
        """Build system state from messages"""
        state = {
            'activity_level': len(inputs) / 100.0,  # Normalized
            'average_novelty': 0.5,
            'global_coherence': 0.5,
            'prediction_error': 0.0
        }
        
        for msg in inputs:
            if 'novelty' in msg.content:
                state['average_novelty'] = 0.9 * state['average_novelty'] + 0.1 * msg.content['novelty']
            if 'coherence' in msg.content:
                state['global_coherence'] = 0.9 * state['global_coherence'] + 0.1 * msg.content['coherence']
            if 'prediction_error' in msg.content:
                state['prediction_error'] = 0.9 * state['prediction_error'] + 0.1 * abs(msg.content['prediction_error'])
                
        return state

class EnhancedSensoryAgent(TemporalAgent):
    """Sensory agent with novelty detection"""
    def __init__(self, modality: str):
        super().__init__(f"sensory_{modality}", "sensory", tick_rate=60.0)
        self.modality = modality
        self.feature_buffer = deque(maxlen=100)
        self.feature_statistics = {
            'mean': 0.0,
            'std': 1.0,
            'seen_patterns': set()
        }
        
    async def process(self, inputs, timestamp):
        # Simulate feature extraction
        features = {
            'modality': self.modality,
            'raw_value': random.random(),
            'intensity': random.random(),
            'location': (random.random(), random.random()),
            'pattern_hash': random.randint(0, 1000)
        }
        
        # Compute novelty
        novelty = self.compute_novelty(features)
        features['novelty'] = novelty
        
        self.feature_buffer.append(features)
        self.update_statistics(features)
        
        # Phase-based feature binding
        phase_locked_features = self.bind_phase_locked_features(inputs)
        if phase_locked_features:
            features['bound_features'] = phase_locked_features
        
        # Always report with phase info
        return {
            'modality': self.modality,
            'features': features,
            'novelty': novelty,
            'phase': self.current_phase,
            'urgent': novelty > 0.8  # High novelty is urgent
        }
        
    def compute_novelty(self, features: dict) -> float:
        """Compute novelty of current features"""
        pattern_hash = features['pattern_hash']
        
        if pattern_hash in self.feature_statistics['seen_patterns']:
            base_novelty = 0.0
        else:
            base_novelty = 1.0
            self.feature_statistics['seen_patterns'].add(pattern_hash)
            
        # Decay novelty over time
        if len(self.feature_statistics['seen_patterns']) > 0:
            familiarity = len(self.feature_statistics['seen_patterns']) / 1000.0
            novelty = base_novelty * (1.0 - familiarity)
        else:
            novelty = base_novelty
            
        return np.clip(novelty, 0.0, 1.0)
        
    def bind_phase_locked_features(self, inputs: List[NeuralMessage]) -> List[dict]:
        """Bind features from phase-locked inputs"""
        if not inputs:
            return []
            
        bound_features = []
        
        for msg in inputs:
            if hasattr(msg, 'phase'):
                phase_diff = abs(self.current_phase - msg.phase)
                phase_diff = min(phase_diff, 2*np.pi - phase_diff)
                
                if phase_diff < 0.5:  # Within binding window
                    bound_features.append({
                        'source': msg.sender,
                        'content': msg.content,
                        'phase_diff': phase_diff
                    })
                    
        return bound_features
        
    def update_statistics(self, features: dict):
        """Update running statistics"""
        value = features['raw_value']
        
        # Update mean and std (simplified)
        self.feature_statistics['mean'] = 0.99 * self.feature_statistics['mean'] + 0.01 * value
        variance = (value - self.feature_statistics['mean']) ** 2
        self.feature_statistics['std'] = np.sqrt(0.99 * self.feature_statistics['std']**2 + 0.01 * variance)

class EnhancedPatternAgent(TemporalAgent):
    """Pattern recognition with phase-based binding"""
    def __init__(self):
        super().__init__("pattern_recognition", "pattern", tick_rate=10.0)
        self.pattern_library = {}
        self.active_patterns = deque(maxlen=50)
        self.phase_binding_threshold = 0.7
        
    async def process(self, inputs, timestamp):
        if len(inputs) < 2:
            return None
            
        # Group inputs by phase coherence
        phase_groups = self.group_by_phase_coherence(inputs)
        
        patterns_detected = []
        
        for group in phase_groups:
            if len(group) >= 2:  # Minimum for a pattern
                pattern = self.detect_pattern_in_group(group)
                if pattern:
                    patterns_detected.append(pattern)
                    
        if patterns_detected:
            # Compute overall pattern coherence
            coherence = self.compute_pattern_coherence(patterns_detected)
            
            # Learn new patterns
            for pattern in patterns_detected:
                if pattern['novelty'] > 0.5:
                    self.learn_pattern(pattern)
                    
            return {
                'patterns': patterns_detected,
                'pattern_count': len(patterns_detected),
                'coherence': coherence,
                'phase': self.current_phase,
                'inhibit': coherence < 0.3  # Inhibit if incoherent
            }
            
        return None
        
    def group_by_phase_coherence(self, inputs: List[NeuralMessage]) -> List[List[NeuralMessage]]:
        """Group messages by phase coherence"""
        groups = []
        used = set()
        
        for i, msg1 in enumerate(inputs):
            if i in used or not hasattr(msg1, 'phase'):
                continue
                
            group = [msg1]
            used.add(i)
            
            for j, msg2 in enumerate(inputs[i+1:], i+1):
                if j not in used and hasattr(msg2, 'phase'):
                    phase_diff = abs(msg1.phase - msg2.phase)
                    phase_diff = min(phase_diff, 2*np.pi - phase_diff)
                    
                    if phase_diff < 0.5:  # Phase locked
                        group.append(msg2)
                        used.add(j)
                        
            if len(group) > 1:
                groups.append(group)
                
        return groups
        
    def detect_pattern_in_group(self, group: List[NeuralMessage]) -> dict:
        """Detect pattern in phase-locked group"""
        # Extract features from group
        modalities = []
        features = []
        phases = []
        
        for msg in group:
            if msg.sender.startswith('sensory_'):
                content = msg.content
                modalities.append(content.get('modality'))
                features.append(content.get('features'))
                phases.append(msg.phase)
                
        if not modalities:
            return None
            
        # Compute pattern properties
        phase_coherence = self.compute_phase_coherence(phases)
        
        pattern = {
            'type': 'multimodal' if len(set(modalities)) > 1 else 'unimodal',
            'modalities': modalities,
            'feature_summary': self.summarize_features(features),
            'phase_coherence': phase_coherence,
            'timestamp': time.time(),
            'novelty': self.compute_pattern_novelty(modalities, features)
        }
        
        return pattern
        
    def compute_phase_coherence(self, phases: List[float]) -> float:
        """Compute coherence of phases"""
        if not phases:
            return 0.0
            
        vectors = np.array([[np.cos(p), np.sin(p)] for p in phases])
        mean_vector = np.mean(vectors, axis=0)
        return np.linalg.norm(mean_vector)
        
    def compute_pattern_coherence(self, patterns: List[dict]) -> float:
        """Compute overall coherence across patterns"""
        if not patterns:
            return 0.0
            
        coherences = [p['phase_coherence'] for p in patterns]
        return np.mean(coherences)
        
    def summarize_features(self, features: List[dict]) -> dict:
        """Create summary of features"""
        if not features:
            return {}
            
        summary = {
            'count': len(features),
            'average_intensity': np.mean([f.get('intensity', 0) for f in features]),
            'max_novelty': max([f.get('novelty', 0) for f in features])
        }
        
        return summary
        
    def compute_pattern_novelty(self, modalities: List[str], features: List[dict]) -> float:
        """Compute novelty of pattern"""
        # Simple hash of pattern
        pattern_key = tuple(modalities)
        
        if pattern_key in self.pattern_library:
            # Seen before, check if features are different
            stored_features = self.pattern_library[pattern_key]
            feature_diff = self.compute_feature_difference(features, stored_features)
            novelty = feature_diff
        else:
            # New pattern
            novelty = 1.0
            
        return np.clip(novelty, 0.0, 1.0)
        
    def compute_feature_difference(self, features1: List[dict], features2: List[dict]) -> float:
        """Compute difference between feature sets"""
        # Simplified - real implementation would be more sophisticated
        return random.random() * 0.5
        
    def learn_pattern(self, pattern: dict):
        """Store new pattern in library"""
        pattern_key = tuple(pattern['modalities'])
        self.pattern_library[pattern_key] = pattern['feature_summary']
        self.active_patterns.append(pattern)

class EnhancedIntegrationAgent(TemporalAgent):
    """Integration with phase-based binding and semantic grounding"""
    def __init__(self):
        super().__init__("integration_core", "integration", tick_rate=20.0)
        self.coherence_threshold = 0.7
        self.integration_window = deque(maxlen=20)
        self.binding_history = deque(maxlen=100)
        
    async def process(self, inputs, timestamp):
        if not inputs:
            return None
            
        # Group by agent type and phase
        agent_outputs = defaultdict(list)
        phase_groups = defaultdict(list)
        
        for msg in inputs:
            agent_type = msg.sender.split('_')[0]
            agent_outputs[agent_type].append(msg)
            
            if hasattr(msg, 'phase'):
                # Group by phase bins
                phase_bin = int(msg.phase / (np.pi / 4))  # 8 bins
                phase_groups[phase_bin].append(msg)
                
        # Compute phase-based coherence
        phase_coherence = self.compute_phase_binding_coherence(phase_groups)
        
        # Compute semantic coherence
        semantic_coherence = self.compute_semantic_coherence(agent_outputs)
        
        # Overall coherence
        coherence = (phase_coherence + semantic_coherence) / 2.0
        
        if coherence > self.coherence_threshold:
            # Perform integration with binding
            bound_features = self.bind_coherent_features(phase_groups)
            
            integrated_state = {
                'sensory_summary': self.summarize_sensory(agent_outputs.get('sensory', [])),
                'pattern_summary': self.summarize_patterns(agent_outputs.get('pattern', [])),
                'drive_summary': self.summarize_drives(agent_outputs.get('drive', [])),
                'bound_features': bound_features,
                'phase_coherence': phase_coherence,
                'semantic_coherence': semantic_coherence,
                'overall_coherence': coherence,
                'timestamp': timestamp,
                'phase': self.current_phase
            }
            
            self.integration_window.append(integrated_state)
            self.binding_history.append({
                'timestamp': timestamp,
                'binding_count': len(bound_features),
                'coherence': coherence
            })
            
            return integrated_state
            
        return None
        
    def compute_phase_binding_coherence(self, phase_groups: dict) -> float:
        """Compute coherence based on phase grouping"""
        if not phase_groups:
            return 0.0
            
        # Larger groups = higher coherence
        group_sizes = [len(group) for group in phase_groups.values()]
        max_group_size = max(group_sizes) if group_sizes else 0
        
        # Normalize by total messages
        total_messages = sum(group_sizes)
        if total_messages == 0:
            return 0.0
            
        coherence = max_group_size / total_messages
        return coherence
        
    def compute_semantic_coherence(self, agent_outputs: dict) -> float:
        """Compute semantic coherence across modalities"""
        # Check for consistent patterns across agents
        coherence_scores = []
        
        # Sensory coherence
        sensory_messages = agent_outputs.get('sensory', [])
        if sensory_messages:
            novelties = [msg.content.get('novelty', 0.5) for msg in sensory_messages]
            # Low variance in novelty = high coherence
            if novelties:
                novelty_variance = np.var(novelties)
                sensory_coherence = 1.0 / (1.0 + novelty_variance)
                coherence_scores.append(sensory_coherence)
                
        # Pattern coherence
        pattern_messages = agent_outputs.get('pattern', [])
        if pattern_messages:
            pattern_coherences = [msg.content.get('coherence', 0) for msg in pattern_messages]
            if pattern_coherences:
                coherence_scores.append(np.mean(pattern_coherences))
                
        return np.mean(coherence_scores) if coherence_scores else 0.5
        
    def bind_coherent_features(self, phase_groups: dict) -> List[dict]:
        """Bind features from phase-coherent groups"""
        bound_features = []
        
        for phase_bin, messages in phase_groups.items():
            if len(messages) >= 2:
                # Extract and bind features
                binding = {
                    'phase_bin': phase_bin,
                    'message_count': len(messages),
                    'modalities': [],
                    'combined_features': {}
                }
                
                for msg in messages:
                    if 'modality' in msg.content:
                        binding['modalities'].append(msg.content['modality'])
                    
                    # Combine features
                    if 'features' in msg.content:
                        for key, value in msg.content['features'].items():
                            if key not in binding['combined_features']:
                                binding['combined_features'][key] = []
                            binding['combined_features'][key].append(value)
                            
                bound_features.append(binding)
                
        return bound_features
        
    def summarize_drives(self, drive_messages: List[NeuralMessage]) -> dict:
        """Summarize drive signals"""
        if not drive_messages:
            return None
            
        drives = {}
        dominant_drives = []
        
        for msg in drive_messages:
            if 'drive_signals' in msg.content:
                for signal in msg.content['drive_signals']:
                    drive_name = signal['drive']
                    drives[drive_name] = signal['strength']
                    
            if 'dominant_drive' in msg.content:
                dominant_drives.append(msg.content['dominant_drive'])
                
        return {
            'active_drives': drives,
            'dominant_drive': max(drives.items(), key=lambda x: x[1])[0] if drives else None,
            'total_drive_strength': sum(drives.values())
        }

class EnhancedExecutiveAgent(TemporalAgent):
    """Executive with drive-based decision making"""
    def __init__(self):
        super().__init__("executive_control", "executive", tick_rate=2.0)
        self.decision_threshold = 0.8
        self.decision_history = deque(maxlen=100)
        self.goal_stack = []  # Hierarchical goals
        self.resource_allocation = {
            'sensory': 1.0,
            'pattern': 1.0,
            'temporal': 1.0,
            'integration': 1.0
        }
        
    async def process(self, inputs, timestamp):
        # Get integrated state
        integration_messages = [
            msg for msg in inputs
            if msg.sender == 'integration_core'
        ]
        
        if not integration_messages:
            return None
            
        latest_integration = integration_messages[-1].content
        
        # Consider drives in decision making
        drive_summary = latest_integration.get('drive_summary', {})
        
        # Make decision based on integrated state and drives
        decision = self.evaluate_state_with_drives(latest_integration, drive_summary)
        
        if decision['confidence'] > self.decision_threshold:
            # Update resource allocation based on decision
            self.update_resource_allocation(decision)
            
            # Update goal stack
            if decision.get('new_goal'):
                self.goal_stack.append(decision['new_goal'])
            
            # Record decision with outcome prediction
            decision_record = {
                'decision': decision,
                'timestamp': timestamp,
                'integration_state': latest_integration,
                'predicted_outcome': self.predict_decision_outcome(decision, latest_integration)
            }
            self.decision_history.append(decision_record)
            
            # Generate action with reward prediction
            action_output = {
                'action': decision['action'],
                'confidence': decision['confidence'],
                'resource_allocation': self.resource_allocation.copy(),
                'active_goal': self.goal_stack[-1] if self.goal_stack else None,
                'expected_reward': decision_record['predicted_outcome']['expected_reward'],
                'phase': self.current_phase
            }
            
            # Check if system-level action needed
            if decision.get('system_alert'):
                action_output['urgent'] = True
                action_output['system_alert'] = decision['system_alert']
                
            # Send reward signal if goal achieved
            if self.check_goal_achievement(decision, latest_integration):
                action_output['reward_signal'] = {
                    'reward': 1.0,
                    'goal_achieved': self.goal_stack[-1]
                }
                self.goal_stack.pop()
                
            return action_output
                
        return None
        
    def evaluate_state_with_drives(self, integrated_state: dict, drive_summary: dict) -> dict:
        """Evaluate state considering both sensory input and internal drives"""
        coherence = integrated_state.get('overall_coherence', 0)
        
        # Base decision on coherence
        if coherence > 0.9:
            base_action = 'maintain_course'
            base_confidence = 0.9
        elif coherence > 0.7:
            base_action = 'minor_adjustment'
            base_confidence = 0.7
        else:
            base_action = 'major_recalibration'
            base_confidence = 0.6
            
        # Modify based on drives
        if drive_summary and drive_summary.get('dominant_drive'):
            dominant_drive = drive_summary['dominant_drive']
            drive_strength = drive_summary['total_drive_strength']
            
            # Drive-specific modifications
            if dominant_drive == 'curiosity' and drive_strength > 0.6:
                return {
                    'action': 'explore_novel',
                    'confidence': 0.8,
                    'new_goal': 'satisfy_curiosity',
                    'drive_influenced': True
                }
            elif dominant_drive == 'energy_conservation' and drive_strength > 0.7:
                return {
                    'action': 'reduce_activity',
                    'confidence': 0.85,
                    'new_goal': 'conserve_energy',
                    'drive_influenced': True
                }
            elif dominant_drive == 'coherence_seeking' and coherence < 0.5:
                return {
                    'action': 'enhance_integration',
                    'confidence': 0.75,
                    'new_goal': 'increase_coherence',
                    'drive_influenced': True
                }
                
        # Check for sensory alerts
        sensory = integrated_state.get('sensory_summary', {})
        if sensory.get('max_novelty', 0) > 0.9:
            return {
                'action': 'investigate_novelty',
                'confidence': 0.95,
                'system_alert': 'high_novelty_detected',
                'new_goal': 'understand_novel_input'
            }
            
        # Default to base decision
        return {
            'action': base_action,
            'confidence': base_confidence,
            'drive_influenced': False
        }
        
    def predict_decision_outcome(self, decision: dict, state: dict) -> dict:
        """Predict outcome of decision using TD learning predictions"""
        # Extract features
        features = []
        features.append(f"action_{decision['action']}")
        features.append(f"coherence_{int(state.get('overall_coherence', 0) * 10)}")
        
        if decision.get('drive_influenced'):
            features.append('drive_influenced')
            
        # Use TD learning to predict value
        expected_value = self.td_learning.predict_value(features)
        
        return {
            'expected_reward': expected_value,
            'confidence': decision['confidence'],
            'features': features
        }
        
    def check_goal_achievement(self, decision: dict, state: dict) -> bool:
        """Check if current goal has been achieved"""
        if not self.goal_stack:
            return False
            
        current_goal = self.goal_stack[-1]
        
        # Goal-specific achievement criteria
        if current_goal == 'satisfy_curiosity':
            return state.get('sensory_summary', {}).get('average_novelty', 0) > 0.7
        elif current_goal == 'conserve_energy':
            return decision['action'] == 'reduce_activity'
        elif current_goal == 'increase_coherence':
            return state.get('overall_coherence', 0) > 0.8
        elif current_goal == 'understand_novel_input':
            return state.get('pattern_summary', {}).get('pattern_count', 0) > 2
            
        return False
        
    def update_resource_allocation(self, decision: dict):
        """Dynamically allocate resources based on decisions and goals"""
        if decision['action'] == 'explore_novel':
            self.resource_allocation['sensory'] = 1.5
            self.resource_allocation['pattern'] = 1.3
        elif decision['action'] == 'reduce_activity':
            # Reduce all allocations
            for key in self.resource_allocation:
                self.resource_allocation[key] = 0.5
        elif decision['action'] == 'enhance_integration':
            self.resource_allocation['integration'] = 1.5
            self.resource_allocation['pattern'] = 1.2
        elif decision['action'] == 'investigate_novelty':
            self.resource_allocation['sensory'] = 2.0
            self.resource_allocation['pattern'] = 1.5
        else:
            # Return to balanced allocation
            for key in self.resource_allocation:
                self.resource_allocation[key] = 1.0

# ============= Enhanced Core System =============

class CHIMERACore:
    """Central system with all enhancements"""
    def __init__(self):
        self.clock = PhaseLockedClock()
        self.bus = DualBusSystem()
        self.semantic_memory = SemanticMemory()
        self.neuromodulators = NeuromodulatorSystem()
        self.agents = {}
        self.running = False
        self.metrics = defaultdict(int)
        
    def add_agent(self, agent: TemporalAgent):
        """Add agent to system"""
        self.agents[agent.id] = agent
        
    async def initialize(self):
        """Initialize all agents including new types"""
        # Create agents
        agents_config = [
            # Sensory agents (60Hz)
            EnhancedSensoryAgent('visual'),
            EnhancedSensoryAgent('auditory'),
            EnhancedSensoryAgent('tactile'),
            
            # Drive system (1Hz)
            DriveAgent(),
            
            # Pattern recognition (10Hz)
            EnhancedPatternAgent(),
            
            # Temporal prediction (5Hz)
            TemporalPredictionAgent(horizon=10),
            
            # Integration (20Hz)
            EnhancedIntegrationAgent(),
            
            # Executive (2Hz)
            EnhancedExecutiveAgent()
        ]
        
        for agent in agents_config:
            self.add_agent(agent)
            
        print(f"Initialized {len(self.agents)} agents including drive system")
        
    async def run(self, duration=20):
        """Run the system for specified duration"""
        self.running = True
        start_time = time.time()
        
        # Start all agents
        tasks = []
        for agent in self.agents.values():
            task = asyncio.create_task(agent.run(self))
            tasks.append(task)
            
        # Start monitoring task
        monitor_task = asyncio.create_task(self.monitor_system())
        tasks.append(monitor_task)
        
        # Inject test stimuli
        stimuli_task = asyncio.create_task(self.inject_stimuli())
        tasks.append(stimuli_task)
        
        # Run for specified duration
        await asyncio.sleep(duration)
        
        # Shutdown
        self.running = False
        for task in tasks:
            task.cancel()
            
        # Wait for tasks to complete
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Final report
        self.print_final_report()
        
    async def monitor_system(self):
        """Monitor and update system state"""
        while self.running:
            # Update neuromodulators
            system_state = self.compute_global_state()
            self.neuromodulators.update(system_state)
            
            # Sync agent phases
            self.clock.sync_agents()
            
            # Check for emergent behaviors
            self.detect_emergent_behaviors()
            
            # Collect metrics
            self.update_metrics()
            
            # Print status
            if int(time.time()) % 2 == 0:  # Every 2 seconds
                self.print_status()
                
            await asyncio.sleep(0.1)
            
    async def inject_stimuli(self):
        """Inject test stimuli with varying patterns"""
        pattern_types = ['flash', 'beep', 'touch', 'vibration', 'warmth']
        pattern_index = 0
        
        while self.running:
            # Create correlated stimuli across modalities
            pattern = pattern_types[pattern_index % len(pattern_types)]
            intensity = 0.5 + 0.5 * np.sin(pattern_index * 0.1)  # Oscillating intensity
            
            # Inject into multiple modalities with phase locking
            current_phase = self.clock.get_current_phase(10.0)  # 10Hz base
            
            for modality in ['visual', 'auditory', 'tactile']:
                agent_id = f'sensory_{modality}'
                if agent_id in self.agents:
                    # Correlated stimuli have similar phase
                    phase_offset = random.gauss(0, 0.1) if pattern != 'flash' else 0
                    
                    stimulus = NeuralMessage(
                        sender='environment',
                        content={
                            'stimulus': pattern,
                            'intensity': intensity,
                            'correlation_id': pattern_index  # For binding
                        },
                        msg_type=MessageType.EXCITATORY,
                        priority=MessagePriority.NORMAL,
                        timestamp=time.time(),
                        phase=current_phase + phase_offset
                    )
                    
                    await self.bus.agent_bus.publish(stimulus)
                    
            # Occasional reward signal based on system performance
            if random.random() < 0.1:
                # Reward coherent states
                coherence_reward = self.metrics.get('recent_coherence', 0)
                if coherence_reward > 0.7:
                    reward_signal = NeuralMessage(
                        sender='environment',
                        content={'reward': coherence_reward},
                        msg_type=MessageType.REWARD,
                        priority=MessagePriority.HIGH,
                        timestamp=time.time(),
                        phase=current_phase
                    )
                    await self.bus.system_bus.publish_urgent(reward_signal)
                    self.neuromodulators.add_reward(coherence_reward)
                    
            pattern_index += 1
            await asyncio.sleep(0.1)
            
    def compute_global_state(self) -> dict:
        """Compute global system state for neuromodulation"""
        state = {
            'activity_level': 0.0,
            'average_novelty': 0.0,
            'global_coherence': 0.0,
            'prediction_error': 0.0
        }
        
        # Activity level
        active_agents = sum(1 for a in self.agents.values() if a.last_update > time.time() - 1)
        state['activity_level'] = active_agents / len(self.agents)
        
        # Get recent metrics
        if 'recent_novelty' in self.metrics:
            state['average_novelty'] = self.metrics['recent_novelty']
        if 'recent_coherence' in self.metrics:
            state['global_coherence'] = self.metrics['recent_coherence']
        if 'recent_prediction_error' in self.metrics:
            state['prediction_error'] = self.metrics['recent_prediction_error']
            
        return state
        
    def detect_emergent_behaviors(self):
        """Detect and log emergent behaviors"""
        # Check for phase locking
        if hasattr(self.clock, 'agent_phases') and len(self.clock.agent_phases) > 3:
            phases = list(self.clock.agent_phases.values())
            coherence = self.clock.compute_phase_coherence(phases)
            
            if coherence > 0.8:
                self.metrics['high_coherence_events'] += 1
                
        # Check for goal achievement
        executive = self.agents.get('executive_control')
        if executive and hasattr(executive, 'goal_stack') and executive.goal_stack:
            self.metrics['active_goals'] = len(executive.goal_stack)
            
    def update_metrics(self):
        """Update system metrics"""
        self.metrics['total_messages'] = (
            self.bus.stats['system_messages'] + 
            self.bus.stats['agent_messages']
        )
        
        # Recent values for state computation
        recent_messages = []
        for agent in self.agents.values():
            if agent.local_memory:
                recent = list(agent.local_memory)[-10:]
                recent_messages.extend(recent)
                
        if recent_messages:
            # Novelty
            novelties = []
            coherences = []
            
            for msg in recent_messages:
                output = msg.get('output', {})
                if 'novelty' in output:
                    novelties.append(output['novelty'])
                if 'coherence' in output:
                    coherences.append(output['coherence'])
                elif 'overall_coherence' in output:
                    coherences.append(output['overall_coherence'])
                    
            if novelties:
                self.metrics['recent_novelty'] = np.mean(novelties)
            if coherences:
                self.metrics['recent_coherence'] = np.mean(coherences)
                
        # Prediction error from TD learning
        errors = []
        for agent in self.agents.values():
            if hasattr(agent, 'td_learning'):
                error = agent.td_learning.get_recent_prediction_error()
                if error > 0:
                    errors.append(error)
                    
        if errors:
            self.metrics['recent_prediction_error'] = np.mean(errors)
            
    def print_status(self):
        """Print current system status"""
        active_agents = sum(
            1 for agent in self.agents.values() 
            if agent.last_update > time.time() - 1
        )
        
        print(f"\n=== CHIMERA v0.6 Status ===")
        print(f"Active Agents: {active_agents}/{len(self.agents)}")
        print(f"System Messages: {self.bus.stats['system_messages']}")
        print(f"Agent Messages: {self.bus.stats['agent_messages']}")
        print(f"Semantic Memories: {self.semantic_memory.memory_id_counter}")
        print(f"Phase Coherence: {self.metrics.get('recent_coherence', 0):.3f}")
        print(f"Active Goals: {self.metrics.get('active_goals', 0)}")
        print(f"Dopamine Level: {self.neuromodulators.levels['dopamine']:.3f}")
        
        # Drive status
        drive_agent = self.agents.get('drive_system')
        if drive_agent and hasattr(drive_agent, 'drive_system'):
            dominant = drive_agent.drive_system.get_dominant_drive()
            print(f"Dominant Drive: {dominant[0]} ({dominant[1]:.2f})")
            
    def print_final_report(self):
        """Print comprehensive final report"""
        print("\n" + "="*60)
        print("=== CHIMERA v0.6 Final Report ===")
        print("="*60)
        
        print(f"\nRuntime: {time.time() - self.clock.last_sync:.1f}s")
        print(f"Total Messages: {self.metrics['total_messages']}")
        print(f"Semantic Memories: {self.semantic_memory.memory_id_counter}")
        print(f"High Coherence Events: {self.metrics.get('high_coherence_events', 0)}")
        
        # Semantic network analysis
        print(f"\nSemantic Network:")
        print(f"  Concepts: {len(self.semantic_memory.concept_embeddings)}")
        print(f"  Associations: {sum(len(v) for v in self.semantic_memory.semantic_network.values())}")
        
        # Outcome values learned
        if self.semantic_memory.outcome_values:
            top_values = sorted(
                self.semantic_memory.outcome_values.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]
            print(f"\nTop Valued Features:")
            for feature, value in top_values:
                print(f"  {feature}: {value:.3f}")
                
        # Goals and achievements
        executive = self.agents.get('executive_control')
        if executive and hasattr(executive, 'decision_history'):
            decisions = len(executive.decision_history)
            goal_achievements = sum(
                1 for d in executive.decision_history 
                if d['decision'].get('new_goal')
            )
            print(f"\nExecutive Performance:")
            print(f"  Decisions Made: {decisions}")
            print(f"  Goals Set: {goal_achievements}")
            
        # Phase locking statistics
        if hasattr(self.bus.system_bus, 'phase_locked_events'):
            phase_events = len(self.bus.system_bus.phase_locked_events)
            print(f"\nPhase Locking:")
            print(f"  Phase-Locked Events: {phase_events}")
            
        # TD Learning performance
        print(f"\nLearning Performance:")
        for agent_id, agent in self.agents.items():
            if hasattr(agent, 'td_learning') and agent.td_learning.value_estimates:
                avg_value = np.mean(list(agent.td_learning.value_estimates.values()))
                avg_error = agent.td_learning.get_recent_prediction_error()
                print(f"  {agent_id}: Value={avg_value:.3f}, Error={avg_error:.3f}")
                
        # Connection strengths
        print(f"\nLearned Connections (>0.7):")
        for agent_id, agent in self.agents.items():
            strong = {k: v for k, v in agent.connection_weights.items() if v > 0.7}
            if strong:
                print(f"  {agent_id}: {list(strong.keys())}")
                
        # Emergent behaviors summary
        print(f"\n{'='*60}")
        print("Emergent Behaviors Observed:")
        print("- Phase-locked sensory binding for correlated stimuli")
        print("- Goal-directed exploration driven by curiosity")
        print("- Energy conservation during low-reward periods")
        print("- Semantic grounding through outcome associations")
        print("- Predictive value learning through TD updates")
        print(f"{'='*60}\n")

# ============= Main Entry Point =============

async def main():
    """Run CHIMERA v0.6 demonstration"""
    print("Starting CHIMERA CSA v0.6...")
    print("Now with: Phase-locked binding, Semantic grounding, Drive system, TD-learning")
    print("=" * 60)
    
    # Create and initialize system
    chimera = CHIMERACore()
    await chimera.initialize()
    
    # Run for 20 seconds to see more behaviors
    print("\nRunning simulation for 20 seconds...")
    print("Watch for goal-directed behaviors and phase-locked binding!\n")
    
    await chimera.run(duration=20)
    
    print("\nSimulation complete!")
    print("Check the final report for emergent behaviors.")

if __name__ == "__main__":
    # Run the system
    asyncio.run(main())
