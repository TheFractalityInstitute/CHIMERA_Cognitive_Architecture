#!/usr/bin/env python3
"""
CHIMERA CSA v0.5 - Complete Neuromorphic Emulation System
Incorporates all fixes from DeepSeek-R1 and dual-bus architecture
"""

import asyncio
import time
import random
import numpy as np
from collections import deque, defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any
import json
import zlib
import pickle
import uuid

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

@dataclass
class NeuralMessage:
    sender: str
    content: Any
    msg_type: MessageType
    priority: MessagePriority
    timestamp: float
    strength: float = 1.0
    
    @property
    def neurotransmitter(self):
        mapping = {
            MessageType.EXCITATORY: "glutamate",
            MessageType.INHIBITORY: "GABA",
            MessageType.MODULATORY: "dopamine"
        }
        return mapping.get(self.msg_type, "unknown")

# ============= Phase-Locked Clock System =============

class PhaseLockedClock:
    """Global phase-locked clock with drift correction"""
    def __init__(self, base_frequency=1000.0):
        self.base_frequency = base_frequency
        self.phase = 0.0
        self.last_sync = time.perf_counter()
        self.drift_correction = 0.0
        self.agent_phases = {}
        
    def get_sync_time(self):
        """Returns phase-locked time with drift correction"""
        current = time.perf_counter()
        elapsed = current - self.last_sync
        
        self.phase += elapsed * self.base_frequency + self.drift_correction
        self.last_sync = current
        
        return self.phase / self.base_frequency
        
    def register_agent_phase(self, agent_id: str, phase: float):
        """Register agent phase for synchronization"""
        self.agent_phases[agent_id] = phase
        
    def sync_agents(self):
        """Kuramoto-style phase synchronization"""
        if not self.agent_phases:
            return
            
        phases = list(self.agent_phases.values())
        mean_phase = np.mean(phases)
        self.drift_correction = 0.1 * (mean_phase - self.phase)

# ============= Dual Bus System =============

class SystemBus:
    """High-priority, low-latency system coordination"""
    def __init__(self, max_latency_ms=1):
        self.queue = asyncio.PriorityQueue()
        self.max_latency = max_latency_ms / 1000
        self.subscribers = defaultdict(list)
        self.interrupt_handlers = {}
        
    async def publish_urgent(self, message: NeuralMessage):
        """Publish high-priority system message"""
        await self.queue.put((message.priority.value, message))
        
        # Direct interrupt for critical messages
        if message.priority == MessagePriority.CRITICAL:
            await self._direct_notify(message)
            
    async def _direct_notify(self, message: NeuralMessage):
        """Skip queue for critical messages"""
        for agent_id in self.interrupt_handlers:
            handler = self.interrupt_handlers[agent_id]
            await handler(message)
            
    async def subscribe_interrupts(self, agent_id: str, handler):
        """Subscribe to system interrupts"""
        self.interrupt_handlers[agent_id] = handler
        
    async def get_messages(self, timeout=0.001):
        """Get system messages with minimal latency"""
        messages = []
        deadline = time.time() + timeout
        
        while time.time() < deadline:
            try:
                priority, message = await asyncio.wait_for(
                    self.queue.get(), 
                    timeout=deadline - time.time()
                )
                messages.append(message)
            except asyncio.TimeoutError:
                break
                
        return messages

class AgentBus:
    """Standard inter-agent communication with reliability"""
    def __init__(self):
        self.queue = asyncio.Queue()
        self.compression_threshold = 1024
        self.delivery_tracking = {}
        self.subscribers = defaultdict(list)
        
    async def publish(self, message: NeuralMessage):
        """Publish agent message with reliability tracking"""
        # Add tracking
        msg_id = str(uuid.uuid4())
        message.id = msg_id
        
        # Compress large messages
        if len(str(message.content)) > self.compression_threshold:
            compressed = zlib.compress(pickle.dumps(message.content))
            message.content = {'_compressed': True, 'data': compressed}
            
        self.delivery_tracking[msg_id] = {
            'sent': time.time(),
            'confirmed': False
        }
        
        await self.queue.put(message)
        
    async def get_messages(self, agent_id: str, timeout=0.01):
        """Get messages for specific agent"""
        messages = []
        deadline = time.time() + timeout
        
        while time.time() < deadline:
            try:
                message = await asyncio.wait_for(
                    self.queue.get(),
                    timeout=deadline - time.time()
                )
                
                # Decompress if needed
                if isinstance(message.content, dict) and message.content.get('_compressed'):
                    message.content = pickle.loads(zlib.decompress(message.content['data']))
                    
                # Check if agent should receive this
                if self._should_receive(agent_id, message):
                    messages.append(message)
                    
            except asyncio.TimeoutError:
                break
                
        return messages
        
    def _should_receive(self, agent_id: str, message: NeuralMessage):
        """Determine if agent should receive message"""
        # TODO: Implement topic-based filtering
        return True

class DualBusSystem:
    """Parallel message buses for different priority levels"""
    def __init__(self):
        self.system_bus = SystemBus(max_latency_ms=1)
        self.agent_bus = AgentBus()
        self.stats = {
            'system_messages': 0,
            'agent_messages': 0
        }

# ============= Memory System =============

class SharedMemoryIndex:
    """Content-addressable memory across all agents"""
    def __init__(self):
        self.semantic_index = {}
        self.temporal_index = {}
        self.causal_graph = defaultdict(list)
        self.memory_id_counter = 0
        
    def index_memory(self, agent_id: str, memory_entry: dict):
        """Index memory for retrieval"""
        memory_id = f"{agent_id}_{self.memory_id_counter}"
        self.memory_id_counter += 1
        
        memory_entry['id'] = memory_id
        memory_entry['agent_id'] = agent_id
        
        # Temporal indexing
        timestamp = memory_entry.get('timestamp', time.time())
        self.temporal_index[timestamp] = memory_entry
        
        # Semantic indexing (simplified - real impl would use embeddings)
        if 'features' in memory_entry:
            feature_key = str(memory_entry['features'])
            self.semantic_index[feature_key] = memory_entry
            
        # Causal tracking
        if 'caused_by' in memory_entry:
            self.causal_graph[memory_entry['caused_by']].append(memory_id)
            
        return memory_id
        
    def retrieve_by_time(self, start_time: float, end_time: float):
        """Retrieve memories in time range"""
        return [
            mem for t, mem in self.temporal_index.items()
            if start_time <= t <= end_time
        ]
        
    def retrieve_by_similarity(self, features, top_k=5):
        """Retrieve similar memories (simplified)"""
        # Real implementation would use vector similarity
        return list(self.semantic_index.values())[:top_k]

# ============= Neuromodulation System =============

class NeuromodulatorSystem:
    """Global neuromodulatory dynamics"""
    def __init__(self):
        self.levels = {
            'dopamine': 0.5,      # Reward/motivation
            'serotonin': 0.5,     # Mood/inhibition
            'norepinephrine': 0.5, # Arousal/attention
            'acetylcholine': 0.5   # Learning/attention
        }
        self.circadian_phase = 0.0
        self.reward_history = deque(maxlen=100)
        
    def update(self, global_state: dict):
        """Update neuromodulator levels"""
        # Circadian rhythm
        time_of_day = (time.time() % 86400) / 86400
        self.circadian_phase = 2 * np.pi * time_of_day
        
        # Dopamine: based on recent rewards
        if self.reward_history:
            recent_reward = np.mean(list(self.reward_history)[-10:])
            self.levels['dopamine'] = 0.3 + 0.4 * np.tanh(recent_reward)
            
        # Norepinephrine: circadian + arousal
        self.levels['norepinephrine'] = 0.5 + 0.3 * np.sin(self.circadian_phase)
        
        # Maintain homeostasis
        self._maintain_homeostasis()
        
    def _maintain_homeostasis(self):
        """Keep levels in healthy range"""
        for neurotransmitter in self.levels:
            self.levels[neurotransmitter] = np.clip(
                self.levels[neurotransmitter], 0.1, 0.9
            )
            
    def add_reward(self, reward: float):
        """Register reward event"""
        self.reward_history.append(reward)

# ============= Base Agent Architecture =============

class TemporalAgent:
    """Base class for all CHIMERA agents"""
    def __init__(self, agent_id: str, agent_type: str, tick_rate: float = 10.0):
        self.id = agent_id
        self.agent_type = agent_type
        self.tick_rate = tick_rate
        self.tick_period = 1.0 / tick_rate
        self.local_memory = deque(maxlen=1000)
        self.last_update = 0
        self.phase_offset = random.random() * 2 * np.pi
        self.running = True
        
        # Plasticity
        self.connection_weights = defaultdict(lambda: 0.5)
        self.learning_rate = 0.01
        
    async def run(self, core_system):
        """Main agent loop with dual bus integration"""
        while self.running:
            # Get synchronized time
            current_time = core_system.clock.get_sync_time()
            
            # Register phase for synchronization
            phase = (current_time * self.tick_rate + self.phase_offset) % (2 * np.pi)
            core_system.clock.register_agent_phase(self.id, phase)
            
            # Check for system interrupts
            system_messages = await core_system.bus.system_bus.get_messages()
            for msg in system_messages:
                await self.handle_interrupt(msg)
                
            # Process regular messages
            if current_time - self.last_update >= self.tick_period:
                agent_messages = await core_system.bus.agent_bus.get_messages(self.id)
                
                # Apply neuromodulation
                modulated_messages = self.apply_neuromodulation(
                    agent_messages, 
                    core_system.neuromodulators
                )
                
                # Process
                output = await self.process(modulated_messages, current_time)
                
                if output is not None:
                    # Update plasticity
                    self.update_connections(agent_messages, output)
                    
                    # Store in local and shared memory
                    memory_entry = {
                        'input': agent_messages,
                        'output': output,
                        'timestamp': current_time
                    }
                    self.local_memory.append(memory_entry)
                    core_system.memory.index_memory(self.id, memory_entry)
                    
                    # Route output to appropriate bus
                    await self.publish_output(output, core_system.bus)
                    
                self.last_update = current_time
                
            await asyncio.sleep(0.001)
            
    async def handle_interrupt(self, message: NeuralMessage):
        """Handle high-priority system messages"""
        if message.content.get('command') == 'emergency_stop':
            self.running = False
            
    def apply_neuromodulation(self, messages, neuromodulators):
        """Apply global neuromodulation to messages"""
        modulated = []
        for msg in messages:
            # Copy message
            mod_msg = NeuralMessage(
                sender=msg.sender,
                content=msg.content,
                msg_type=msg.msg_type,
                priority=msg.priority,
                timestamp=msg.timestamp,
                strength=msg.strength
            )
            
            # Apply modulation based on agent type
            if self.agent_type == 'sensory':
                # Norepinephrine affects sensory gain
                mod_msg.strength *= (0.5 + neuromodulators.levels['norepinephrine'])
            elif self.agent_type == 'executive':
                # Dopamine affects decision making
                mod_msg.strength *= (0.5 + neuromodulators.levels['dopamine'])
                
            modulated.append(mod_msg)
            
        return modulated
        
    def update_connections(self, inputs, output):
        """Hebbian learning rule"""
        for msg in inputs:
            # Neurons that fire together wire together
            correlation = self.compute_correlation(msg.content, output)
            
            # Update weight
            old_weight = self.connection_weights[msg.sender]
            new_weight = old_weight + self.learning_rate * correlation
            self.connection_weights[msg.sender] = np.clip(new_weight, 0.0, 1.0)
            
    def compute_correlation(self, input_content, output):
        """Compute input-output correlation (simplified)"""
        # Real implementation would use proper similarity metrics
        return random.random() * 0.1 - 0.05
        
    async def publish_output(self, output, bus_system):
        """Route output to appropriate bus"""
        # Determine priority
        if output.get('urgent') or output.get('system_alert'):
            priority = MessagePriority.HIGH
            msg_type = MessageType.EXCITATORY
        else:
            priority = MessagePriority.NORMAL
            msg_type = MessageType.EXCITATORY
            
        # Check for inhibitory signals
        if output.get('inhibit'):
            msg_type = MessageType.INHIBITORY
            
        message = NeuralMessage(
            sender=self.id,
            content=output,
            msg_type=msg_type,
            priority=priority,
            timestamp=time.time()
        )
        
        # Route to appropriate bus
        if priority.value <= MessagePriority.HIGH.value:
            await bus_system.system_bus.publish_urgent(message)
        else:
            await bus_system.agent_bus.publish(message)
            
    async def process(self, inputs, timestamp):
        """Override in subclasses"""
        raise NotImplementedError

# ============= Specialized Agents =============

class SensoryAgent(TemporalAgent):
    """Fast perception and feature extraction"""
    def __init__(self, modality: str):
        super().__init__(f"sensory_{modality}", "sensory", tick_rate=60.0)
        self.modality = modality
        self.feature_buffer = deque(maxlen=100)
        self.change_threshold = 0.3
        
    async def process(self, inputs, timestamp):
        # Simulate feature extraction
        features = {
            'modality': self.modality,
            'raw_value': random.random(),
            'intensity': random.random(),
            'location': (random.random(), random.random())
        }
        
        self.feature_buffer.append(features)
        
        # Detect significant changes
        if len(self.feature_buffer) > 1:
            change = abs(features['intensity'] - self.feature_buffer[-2]['intensity'])
            
            if change > self.change_threshold:
                return {
                    'modality': self.modality,
                    'features': features,
                    'change_magnitude': change,
                    'urgent': change > 0.7
                }
                
        return None

class PatternAgent(TemporalAgent):
    """Pattern recognition across modalities"""
    def __init__(self):
        super().__init__("pattern_recognition", "pattern", tick_rate=10.0)
        self.pattern_library = {}
        self.active_patterns = deque(maxlen=50)
        self.min_pattern_length = 3
        
    async def process(self, inputs, timestamp):
        if len(inputs) < 2:
            return None
            
        # Aggregate features from sensory agents
        sensory_inputs = [
            msg for msg in inputs 
            if msg.sender.startswith('sensory_')
        ]
        
        if not sensory_inputs:
            return None
            
        # Simple pattern detection (real impl would be sophisticated)
        pattern_detected = len(sensory_inputs) >= self.min_pattern_length
        
        if pattern_detected:
            pattern = {
                'type': 'sequence',
                'modalities': [msg.content.get('modality') for msg in sensory_inputs],
                'confidence': random.random() * 0.5 + 0.5,
                'timestamp': timestamp
            }
            
            self.active_patterns.append(pattern)
            
            return {
                'pattern': pattern,
                'pattern_count': len(self.active_patterns),
                'inhibit': False  # Could inhibit if pattern is familiar
            }
            
        return None

class TemporalPredictionAgent(TemporalAgent):
    """Predicts future states based on temporal patterns"""
    def __init__(self, horizon: int = 10):
        super().__init__("temporal_prediction", "temporal", tick_rate=5.0)
        self.horizon = horizon
        self.prediction_buffer = deque(maxlen=100)
        self.prediction_accuracy = deque(maxlen=50)
        
    async def process(self, inputs, timestamp):
        # Get pattern inputs
        pattern_inputs = [
            msg for msg in inputs
            if msg.sender == 'pattern_recognition'
        ]
        
        if not pattern_inputs:
            return None
            
        # Generate prediction (simplified)
        prediction = {
            'future_state': 'pattern_continues',
            'confidence': np.mean([p.content.get('confidence', 0.5) for p in pattern_inputs]),
            'horizon': self.horizon,
            'timestamp': timestamp
        }
        
        self.prediction_buffer.append(prediction)
        
        # Update accuracy based on past predictions
        if len(self.prediction_buffer) > self.horizon:
            past_prediction = self.prediction_buffer[-self.horizon]
            # Simplified accuracy check
            accuracy = random.random() * 0.3 + 0.7
            self.prediction_accuracy.append(accuracy)
            
        return {
            'prediction': prediction,
            'average_accuracy': np.mean(self.prediction_accuracy) if self.prediction_accuracy else 0.5
        }

class IntegrationAgent(TemporalAgent):
    """Fuses information from multiple agents"""
    def __init__(self):
        super().__init__("integration_core", "integration", tick_rate=20.0)
        self.coherence_threshold = 0.7
        self.integration_window = deque(maxlen=20)
        
    async def process(self, inputs, timestamp):
        if not inputs:
            return None
            
        # Group by agent type
        agent_outputs = defaultdict(list)
        for msg in inputs:
            agent_type = msg.sender.split('_')[0]
            agent_outputs[agent_type].append(msg)
            
        # Compute coherence
        coherence = self.compute_coherence(agent_outputs)
        
        if coherence > self.coherence_threshold:
            # Perform integration
            integrated_state = {
                'sensory_summary': self.summarize_sensory(agent_outputs.get('sensory', [])),
                'pattern_summary': self.summarize_patterns(agent_outputs.get('pattern', [])),
                'prediction_summary': self.summarize_predictions(agent_outputs.get('temporal', [])),
                'coherence': coherence,
                'timestamp': timestamp,
                'agent_count': len(inputs)
            }
            
            self.integration_window.append(integrated_state)
            
            return integrated_state
            
        return None
        
    def compute_coherence(self, agent_outputs):
        """Measure coherence across agent outputs"""
        if not agent_outputs:
            return 0.0
            
        # Simplified coherence: more agent types = higher coherence
        type_coverage = len(agent_outputs) / 4.0  # Assuming 4 main types
        
        # Temporal coherence: messages close in time
        all_messages = [msg for msgs in agent_outputs.values() for msg in msgs]
        if all_messages:
            timestamps = [msg.timestamp for msg in all_messages]
            time_spread = max(timestamps) - min(timestamps)
            temporal_coherence = 1.0 / (1.0 + time_spread)
        else:
            temporal_coherence = 0.0
            
        return (type_coverage + temporal_coherence) / 2.0
        
    def summarize_sensory(self, sensory_messages):
        """Summarize sensory inputs"""
        if not sensory_messages:
            return None
            
        modalities = [msg.content.get('modality') for msg in sensory_messages]
        intensities = [msg.content.get('features', {}).get('intensity', 0) for msg in sensory_messages]
        
        return {
            'active_modalities': list(set(modalities)),
            'average_intensity': np.mean(intensities) if intensities else 0,
            'peak_intensity': max(intensities) if intensities else 0
        }
        
    def summarize_patterns(self, pattern_messages):
        """Summarize detected patterns"""
        if not pattern_messages:
            return None
            
        patterns = [msg.content.get('pattern') for msg in pattern_messages if msg.content.get('pattern')]
        
        return {
            'pattern_count': len(patterns),
            'average_confidence': np.mean([p.get('confidence', 0) for p in patterns]) if patterns else 0
        }
        
    def summarize_predictions(self, prediction_messages):
        """Summarize predictions"""
        if not prediction_messages:
            return None
            
        predictions = [msg.content.get('prediction') for msg in prediction_messages]
        accuracies = [msg.content.get('average_accuracy', 0.5) for msg in prediction_messages]
        
        return {
            'prediction_count': len(predictions),
            'average_accuracy': np.mean(accuracies) if accuracies else 0.5
        }

class ExecutiveAgent(TemporalAgent):
    """High-level decision making and resource allocation"""
    def __init__(self):
        super().__init__("executive_control", "executive", tick_rate=2.0)
        self.decision_threshold = 0.8
        self.decision_history = deque(maxlen=100)
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
        
        # Make decision based on integrated state
        decision = self.evaluate_state(latest_integration)
        
        if decision['confidence'] > self.decision_threshold:
            # Update resource allocation
            self.update_resource_allocation(decision)
            
            # Record decision
            self.decision_history.append({
                'decision': decision,
                'timestamp': timestamp,
                'integration_state': latest_integration
            })
            
            # Check if system-level action needed
            if decision.get('system_alert'):
                return {
                    'action': decision['action'],
                    'confidence': decision['confidence'],
                    'resource_allocation': self.resource_allocation.copy(),
                    'urgent': True,
                    'system_alert': decision['system_alert']
                }
            else:
                return {
                    'action': decision['action'],
                    'confidence': decision['confidence'],
                    'resource_allocation': self.resource_allocation.copy()
                }
                
        return None
        
    def evaluate_state(self, integrated_state):
        """Evaluate integrated state and make decision"""
        coherence = integrated_state.get('coherence', 0)
        
        # Simple decision logic (real impl would be sophisticated)
        if coherence > 0.9:
            action = 'maintain_course'
            confidence = 0.9
        elif coherence > 0.7:
            action = 'minor_adjustment'
            confidence = 0.7
        else:
            action = 'major_recalibration'
            confidence = 0.6
            
        # Check for alerts
        sensory = integrated_state.get('sensory_summary', {})
        if sensory.get('peak_intensity', 0) > 0.9:
            return {
                'action': 'emergency_response',
                'confidence': 0.95,
                'system_alert': 'high_intensity_stimulus'
            }
            
        return {
            'action': action,
            'confidence': confidence
        }
        
    def update_resource_allocation(self, decision):
        """Dynamically allocate resources based on decisions"""
        if decision['action'] == 'emergency_response':
            # Boost sensory processing
            self.resource_allocation['sensory'] = 1.5
            self.resource_allocation['pattern'] = 0.8
        elif decision['action'] == 'major_recalibration':
            # Boost pattern recognition
            self.resource_allocation['pattern'] = 1.3
            self.resource_allocation['temporal'] = 1.2
        else:
            # Return to baseline
            for key in self.resource_allocation:
                self.resource_allocation[key] = 0.9 + 0.2 * random.random()

# ============= Core System =============

class CHIMERACore:
    """Central system coordinating all components"""
    def __init__(self):
        self.clock = PhaseLockedClock()
        self.bus = DualBusSystem()
        self.memory = SharedMemoryIndex()
        self.neuromodulators = NeuromodulatorSystem()
        self.agents = {}
        self.running = False
        self.metrics = defaultdict(int)
        
    def add_agent(self, agent: TemporalAgent):
        """Add agent to system"""
        self.agents[agent.id] = agent
        
    async def initialize(self):
        """Initialize all agents"""
        # Create agents
        agents_config = [
            SensoryAgent('visual'),
            SensoryAgent('auditory'),
            SensoryAgent('tactile'),
            PatternAgent(),
            TemporalPredictionAgent(horizon=10),
            IntegrationAgent(),
            ExecutiveAgent()
        ]
        
        for agent in agents_config:
            self.add_agent(agent)
            
        print(f"Initialized {len(self.agents)} agents")
        
    async def run(self, duration=10):
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
            self.neuromodulators.update({})
            
            # Sync agent phases
            self.clock.sync_agents()
            
            # Collect metrics
            self.metrics['total_messages'] = (
                self.bus.stats['system_messages'] + 
                self.bus.stats['agent_messages']
            )
            
            # Print status
            if int(time.time()) % 2 == 0:  # Every 2 seconds
                self.print_status()
                
            await asyncio.sleep(0.1)
            
    async def inject_stimuli(self):
        """Inject test stimuli into sensory agents"""
        while self.running:
            # Random sensory input
            for modality in ['visual', 'auditory', 'tactile']:
                agent_id = f'sensory_{modality}'
                if agent_id in self.agents:
                    stimulus = NeuralMessage(
                        sender='environment',
                        content={
                            'stimulus': random.choice(['flash', 'beep', 'touch']),
                            'intensity': random.random()
                        },
                        msg_type=MessageType.EXCITATORY,
                        priority=MessagePriority.NORMAL,
                        timestamp=time.time()
                    )
                    
                    await self.bus.agent_bus.publish(stimulus)
                    
            # Occasional reward signal
            if random.random() < 0.1:
                self.neuromodulators.add_reward(random.random())
                
            await asyncio.sleep(0.1)
            
    def print_status(self):
        """Print current system status"""
        active_agents = sum(
            1 for agent in self.agents.values() 
            if agent.last_update > time.time() - 1
        )
        
        print(f"\n=== CHIMERA Status ===")
        print(f"Active Agents: {active_agents}/{len(self.agents)}")
        print(f"System Messages: {self.bus.stats['system_messages']}")
        print(f"Agent Messages: {self.bus.stats['agent_messages']}")
        print(f"Memories Stored: {self.memory.memory_id_counter}")
        print(f"Dopamine Level: {self.neuromodulators.levels['dopamine']:.3f}")
        
    def print_final_report(self):
        """Print final system report"""
        print("\n=== CHIMERA Final Report ===")
        print(f"Total Runtime: {time.time() - self.clock.last_sync:.1f}s")
        print(f"Total Messages: {self.metrics['total_messages']}")
        print(f"Memories Created: {self.memory.memory_id_counter}")
        
        # Agent-specific stats
        for agent_id, agent in self.agents.items():
            if hasattr(agent, 'decision_history') and agent.decision_history:
                print(f"\n{agent_id} Stats:")
                print(f"  Decisions Made: {len(agent.decision_history)}")
                
        # Connection weights
        print("\nLearned Connections:")
        for agent_id, agent in self.agents.items():
            strong_connections = {
                k: v for k, v in agent.connection_weights.items() 
                if v > 0.7
            }
            if strong_connections:
                print(f"  {agent_id}: {strong_connections}")

# ============= Main Entry Point =============

async def main():
    """Run CHIMERA demonstration"""
    print("Starting CHIMERA CSA v0.5...")
    print("=" * 50)
    
    # Create and initialize system
    chimera = CHIMERACore()
    await chimera.initialize()
    
    # Run for 10 seconds
    print("\nRunning simulation for 10 seconds...")
    print("Watch for emergent behaviors!\n")
    
    await chimera.run(duration=10)
    
    print("\nSimulation complete!")

if __name__ == "__main__":
    # Run the system
    asyncio.run(main())
