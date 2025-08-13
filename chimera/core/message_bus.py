"""
CHIMERA Democratic Message Bus v2.0
Implements biologically-inspired inter-module communication with:
- Neurotransmitter-like message typing
- Refractory periods and neural fatigue
- Consensus building protocols
- Global workspace theory implementation
"""

import asyncio
import time
import json
import numpy as np
from typing import Dict, List, Optional, Any, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import uvicorn
import hashlib
import uuid

# ============= Message Types & Neurotransmitters =============

class Neurotransmitter(Enum):
    """Biological message types with different dynamics"""
    GLUTAMATE = "excitatory"        # Fast excitation
    GABA = "inhibitory"             # Fast inhibition
    DOPAMINE = "reward"             # Motivation/learning
    SEROTONIN = "modulatory"        # Mood/confidence
    ACETYLCHOLINE = "attention"     # Focus/learning
    NOREPINEPHRINE = "arousal"      # Alertness/stress
    OXYTOCIN = "social"            # Bonding/trust
    ENDORPHIN = "pleasure"         # Satisfaction
    CORTISOL = "stress"            # Emergency
    MELATONIN = "circadian"        # Timing

class MessagePriority(Enum):
    EMERGENCY = 0    # Cortisol-like
    CRITICAL = 1     # Norepinephrine
    HIGH = 2         # Dopamine
    NORMAL = 3       # Glutamate/GABA
    LOW = 4          # Serotonin
    BACKGROUND = 5   # Melatonin

@dataclass
class NeuralMessage:
    """Enhanced message with biological properties"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source: str = ""
    target: Optional[str] = None  # None = broadcast
    content: Any = None
    neurotransmitter: Neurotransmitter = Neurotransmitter.GLUTAMATE
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: float = field(default_factory=time.time)
    ttl: float = 1.0  # Time to live (seconds)
    strength: float = 1.0  # Synaptic strength
    phase: float = 0.0  # Phase for binding
    
    # Biological properties
    refractory_period: float = 0.1  # Minimum time between similar messages
    decay_rate: float = 0.5  # How fast the message influence decays
    
    def is_expired(self) -> bool:
        """Check if message has expired"""
        return time.time() - self.timestamp > self.ttl
        
    def get_influence(self) -> float:
        """Calculate current influence based on decay"""
        age = time.time() - self.timestamp
        return self.strength * np.exp(-self.decay_rate * age)

# ============= Global Workspace Theory Implementation =============

class GlobalWorkspace:
    """
    Implements Global Workspace Theory for consciousness
    Only most salient information enters global broadcast
    """
    
    def __init__(self, capacity: int = 7):  # Miller's magical number
        self.capacity = capacity
        self.workspace = deque(maxlen=capacity)
        self.access_threshold = 0.5
        self.competition_rounds = 3
        
    async def compete_for_access(self, 
                                candidates: List[NeuralMessage]) -> List[NeuralMessage]:
        """
        Implements competition for global workspace access
        Similar to how only certain thoughts become conscious
        """
        if not candidates:
            return []
            
        # Calculate salience scores
        scored = []
        for msg in candidates:
            salience = self._calculate_salience(msg)
            if salience > self.access_threshold:
                scored.append((salience, msg))
                
        # Competition rounds (lateral inhibition)
        for round in range(self.competition_rounds):
            # Winners suppress similar messages
            scored = self._lateral_inhibition(scored)
            
        # Take top messages up to capacity
        scored.sort(key=lambda x: x[0], reverse=True)
        winners = [msg for _, msg in scored[:self.capacity]]
        
        # Update workspace
        self.workspace.extend(winners)
        
        return winners
        
    def _calculate_salience(self, msg: NeuralMessage) -> float:
        """Calculate message salience for consciousness"""
        
        # Priority contributes to salience
        priority_weight = 1.0 - (msg.priority.value / 5.0)
        
        # Neurotransmitter-specific weights
        nt_weights = {
            Neurotransmitter.CORTISOL: 1.0,      # Stress = high salience
            Neurotransmitter.DOPAMINE: 0.8,      # Reward = important
            Neurotransmitter.NOREPINEPHRINE: 0.7, # Arousal = notable
            Neurotransmitter.GLUTAMATE: 0.5,     # Normal excitation
            Neurotransmitter.ACETYLCHOLINE: 0.6, # Attention = relevant
            Neurotransmitter.SEROTONIN: 0.4,     # Modulation = background
            Neurotransmitter.GABA: 0.3,          # Inhibition = suppressive
            Neurotransmitter.OXYTOCIN: 0.5,      # Social = context-dependent
            Neurotransmitter.MELATONIN: 0.2,     # Circadian = low priority
        }
        nt_weight = nt_weights.get(msg.neurotransmitter, 0.5)
        
        # Recency and strength
        influence = msg.get_influence()
        
        # Combined salience
        salience = priority_weight * nt_weight * influence
        
        return salience
        
    def _lateral_inhibition(self, 
                          scored: List[tuple]) -> List[tuple]:
        """Implement lateral inhibition between competing messages"""
        if len(scored) <= 1:
            return scored
            
        # Each message inhibits similar messages
        inhibited = []
        for i, (score_i, msg_i) in enumerate(scored):
            inhibition = 0
            for j, (score_j, msg_j) in enumerate(scored):
                if i != j:
                    similarity = self._message_similarity(msg_i, msg_j)
                    if similarity > 0.5:  # Similar messages compete
                        inhibition += score_j * similarity * 0.1
                        
            new_score = max(0, score_i - inhibition)
            if new_score > 0:
                inhibited.append((new_score, msg_i))
                
        return inhibited
        
    def _message_similarity(self, msg1: NeuralMessage, msg2: NeuralMessage) -> float:
        """Calculate similarity between messages"""
        similarity = 0.0
        
        # Same source = more similar
        if msg1.source == msg2.source:
            similarity += 0.3
            
        # Same neurotransmitter = similar function
        if msg1.neurotransmitter == msg2.neurotransmitter:
            similarity += 0.4
            
        # Close in time = related
        time_diff = abs(msg1.timestamp - msg2.timestamp)
        if time_diff < 0.1:
            similarity += 0.3 * (1 - time_diff / 0.1)
            
        return min(1.0, similarity)

# ============= Democratic Consensus Protocols =============

class ConsensusProtocol:
    """
    Implements various consensus mechanisms for democratic decision-making
    """
    
    def __init__(self):
        self.voting_timeout = 5.0  # Seconds to wait for votes
        self.quorum = 0.5  # Minimum participation
        self.consensus_threshold = 0.6  # Agreement needed
        
    async def weighted_vote(self, 
                           proposal_id: str,
                           votes: Dict[str, tuple]) -> Dict[str, Any]:
        """
        Weighted voting based on confidence and expertise
        votes = {module_name: (decision, confidence, expertise)}
        """
        if not votes:
            return {'decision': None, 'consensus': 0, 'participation': 0}
            
        # Calculate participation
        participation = len(votes) / 6.0  # Assuming 6 modules
        
        if participation < self.quorum:
            return {
                'decision': None, 
                'consensus': 0, 
                'participation': participation,
                'error': 'Insufficient quorum'
            }
            
        # Weighted aggregation
        weighted_scores = defaultdict(float)
        total_weight = 0
        
        for module, (decision, confidence, expertise) in votes.items():
            weight = confidence * expertise
            weighted_scores[decision] += weight
            total_weight += weight
            
        # Find winning decision
        if total_weight == 0:
            return {'decision': None, 'consensus': 0, 'participation': participation}
            
        winner = max(weighted_scores, key=weighted_scores.get)
        consensus = weighted_scores[winner] / total_weight
        
        return {
            'decision': winner,
            'consensus': consensus,
            'participation': participation,
            'votes': dict(votes),
            'scores': dict(weighted_scores)
        }
        
    async def byzantine_consensus(self, 
                                 proposal_id: str,
                                 votes: Dict[str, Any]) -> Dict[str, Any]:
        """
        Byzantine fault-tolerant consensus for critical decisions
        Can tolerate up to 1/3 malicious/faulty modules
        """
        n = len(votes)
        f = n // 3  # Maximum Byzantine faults tolerable
        
        # Count votes
        vote_counts = defaultdict(int)
        for module, vote in votes.items():
            vote_counts[vote] += 1
            
        # Need more than 2/3 agreement
        for decision, count in vote_counts.items():
            if count > 2 * f:
                return {
                    'decision': decision,
                    'consensus': count / n,
                    'byzantine_safe': True
                }
                
        return {
            'decision': None,
            'consensus': 0,
            'byzantine_safe': False,
            'error': 'No Byzantine consensus reached'
        }

# ============= Neural Dynamics & Plasticity =============

class SynapticPlasticity:
    """
    Implements synaptic plasticity for learning connection strengths
    """
    
    def __init__(self):
        # Connection strengths between modules
        self.connections = defaultdict(lambda: defaultdict(lambda: 1.0))
        
        # Spike-Timing Dependent Plasticity parameters
        self.tau_plus = 20.0   # ms
        self.tau_minus = 20.0  # ms
        self.a_plus = 0.01     # LTP strength
        self.a_minus = 0.01    # LTD strength
        
        # History for STDP
        self.spike_history = defaultdict(deque)
        
    def record_spike(self, module: str, timestamp: float):
        """Record module activation for STDP"""
        history = self.spike_history[module]
        history.append(timestamp)
        
        # Keep only recent history (100ms window)
        cutoff = timestamp - 0.1
        while history and history[0] < cutoff:
            history.popleft()
            
    def update_connection(self, 
                         pre_module: str, 
                         post_module: str,
                         pre_time: float,
                         post_time: float):
        """
        Update connection strength based on spike timing
        Pre before post = LTP (strengthening)
        Post before pre = LTD (weakening)
        """
        dt = post_time - pre_time
        
        if dt > 0:  # Pre before post - LTP
            delta_w = self.a_plus * np.exp(-dt / self.tau_plus)
        else:  # Post before pre - LTD
            delta_w = -self.a_minus * np.exp(dt / self.tau_minus)
            
        # Update connection
        old_weight = self.connections[pre_module][post_module]
        new_weight = np.clip(old_weight + delta_w, 0.1, 5.0)
        self.connections[pre_module][post_module] = new_weight
        
        return new_weight

# ============= The Democratic Message Bus =============

class DemocraticMessageBus:
    """
    Central nervous system for CHIMERA
    Implements democratic inter-module communication with biological realism
    """
    
    def __init__(self):
        # Core components
        self.workspace = GlobalWorkspace()
        self.consensus = ConsensusProtocol()
        self.plasticity = SynapticPlasticity()
        
        # Message handling
        self.message_queue = asyncio.Queue()
        self.priority_queue = asyncio.PriorityQueue()
        self.broadcast_buffer = deque(maxlen=100)
        
        # Module registry
        self.modules = {}
        self.module_states = defaultdict(lambda: 'inactive')
        self.module_last_seen = defaultdict(float)
        
        # Refractory tracking
        self.refractory_until = defaultdict(float)
        
        # Active proposals for consensus
        self.active_proposals = {}
        self.proposal_votes = defaultdict(dict)
        
        # WebSocket connections for modules
        self.connections = {}
        
        # Statistics
        self.stats = {
            'messages_sent': 0,
            'messages_received': 0,
            'consensus_reached': 0,
            'consensus_failed': 0,
            'global_broadcasts': 0
        }
        
        # FastAPI app
        self.app = FastAPI()
        self._setup_routes()
        
    def _setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.post("/register")
        async def register_module(module_name: str, capabilities: Dict[str, Any]):
            """Register a module with the bus"""
            self.modules[module_name] = {
                'capabilities': capabilities,
                'registered_at': time.time()
            }
            self.module_states[module_name] = 'active'
            return {"status": "registered", "module": module_name}
            
        @self.app.post("/message")
        async def send_message(message: Dict[str, Any]):
            """Send a message through the bus"""
            msg = NeuralMessage(**message)
            await self.route_message(msg)
            return {"status": "sent", "id": msg.id}
            
        @self.app.post("/propose")
        async def create_proposal(proposal: Dict[str, Any]):
            """Create a proposal for democratic decision"""
            proposal_id = str(uuid.uuid4())
            self.active_proposals[proposal_id] = {
                'content': proposal,
                'created_at': time.time(),
                'votes': {}
            }
            
            # Broadcast proposal to all modules
            await self.broadcast_proposal(proposal_id, proposal)
            
            return {"proposal_id": proposal_id}
            
        @self.app.post("/vote/{proposal_id}")
        async def cast_vote(proposal_id: str, 
                          module_name: str,
                          vote: str,
                          confidence: float = 1.0,
                          expertise: float = 1.0):
            """Cast a vote on a proposal"""
            if proposal_id not in self.active_proposals:
                return {"error": "Invalid proposal"}
                
            self.proposal_votes[proposal_id][module_name] = (vote, confidence, expertise)
            
            # Check if we have enough votes
            if len(self.proposal_votes[proposal_id]) >= len(self.modules) * self.consensus.quorum:
                result = await self.consensus.weighted_vote(
                    proposal_id, 
                    self.proposal_votes[proposal_id]
                )
                
                if result['consensus'] > self.consensus.consensus_threshold:
                    self.stats['consensus_reached'] += 1
                    await self.broadcast_decision(proposal_id, result)
                else:
                    self.stats['consensus_failed'] += 1
                    
                return result
                
            return {"status": "vote_recorded"}
            
        @self.app.websocket("/ws/{module_name}")
        async def websocket_endpoint(websocket: WebSocket, module_name: str):
            """WebSocket connection for real-time module communication"""
            await websocket.accept()
            self.connections[module_name] = websocket
            
            try:
                while True:
                    # Receive messages from module
                    data = await websocket.receive_json()
                    
                    if data.get('type') == 'message':
                        msg = NeuralMessage(**data['content'])
                        msg.source = module_name
                        await self.route_message(msg)
                        
                    elif data.get('type') == 'heartbeat':
                        self.module_last_seen[module_name] = time.time()
                        
            except WebSocketDisconnect:
                del self.connections[module_name]
                self.module_states[module_name] = 'disconnected'
                
    async def route_message(self, message: NeuralMessage):
        """
        Route message based on neurotransmitter type and target
        Implements neural routing with refractory periods
        """
        
        # Check refractory period
        if self._in_refractory(message.source):
            return  # Neuron can't fire yet
            
        # Record spike for STDP
        self.plasticity.record_spike(message.source, message.timestamp)
        
        # Update statistics
        self.stats['messages_received'] += 1
        
        # Check if message should enter global workspace
        if message.priority.value <= MessagePriority.HIGH.value:
            winners = await self.workspace.compete_for_access([message])
            if winners:
                await self._global_broadcast(winners[0])
                self.stats['global_broadcasts'] += 1
                
        # Route based on target
        if message.target:
            # Direct message
            await self._send_to_module(message.target, message)
            
            # Update synaptic strength
            if message.target in self.module_last_seen:
                self.plasticity.update_connection(
                    message.source,
                    message.target,
                    message.timestamp,
                    self.module_last_seen[message.target]
                )
        else:
            # Broadcast based on neurotransmitter
            await self._neurotransmitter_broadcast(message)
            
        # Set refractory period
        self._set_refractory(message.source, message.refractory_period)
        
    def _in_refractory(self, module: str) -> bool:
        """Check if module is in refractory period"""
        return time.time() < self.refractory_until[module]
        
    def _set_refractory(self, module: str, period: float):
        """Set refractory period for module"""
        self.refractory_until[module] = time.time() + period
        
    async def _send_to_module(self, target: str, message: NeuralMessage):
        """Send message to specific module"""
        if target in self.connections:
            try:
                await self.connections[target].send_json({
                    'type': 'message',
                    'content': {
                        'id': message.id,
                        'source': message.source,
                        'content': message.content,
                        'neurotransmitter': message.neurotransmitter.value,
                        'strength': message.get_influence(),
                        'timestamp': message.timestamp
                    }
                })
                self.stats['messages_sent'] += 1
            except:
                self.module_states[target] = 'error'
                
    async def _neurotransmitter_broadcast(self, message: NeuralMessage):
        """
        Broadcast based on neurotransmitter type
        Different neurotransmitters affect different module subsets
        """
        
        # Define neurotransmitter targets
        targets = {
            Neurotransmitter.GLUTAMATE: 'all',        # General excitation
            Neurotransmitter.GABA: 'all',             # General inhibition
            Neurotransmitter.DOPAMINE: ['executive', 'planning', 'memory'],
            Neurotransmitter.SEROTONIN: ['emotional', 'social', 'executive'],
            Neurotransmitter.ACETYLCHOLINE: ['memory', 'sensory', 'attention'],
            Neurotransmitter.NOREPINEPHRINE: ['executive', 'sensory', 'arousal'],
            Neurotransmitter.OXYTOCIN: ['social', 'emotional', 'memory'],
            Neurotransmitter.CORTISOL: 'all',         # Emergency broadcast
        }
        
        target_modules = targets.get(message.neurotransmitter, [])
        
        if target_modules == 'all':
            target_modules = list(self.connections.keys())
            
        for module in target_modules:
            if module in self.connections and module != message.source:
                # Apply synaptic weight
                weight = self.plasticity.connections[message.source][module]
                weighted_message = message
                weighted_message.strength *= weight
                
                await self._send_to_module(module, weighted_message)
                
    async def _global_broadcast(self, message: NeuralMessage):
        """Broadcast to global workspace (consciousness)"""
        self.broadcast_buffer.append(message)
        
        # Send to all modules with "conscious" tag
        for module in self.connections:
            await self._send_to_module(module, message)
            
    async def broadcast_proposal(self, proposal_id: str, content: Dict):
        """Broadcast a proposal to all modules for voting"""
        proposal_message = NeuralMessage(
            source='consensus_engine',
            content={'proposal_id': proposal_id, 'content': content},
            neurotransmitter=Neurotransmitter.ACETYLCHOLINE,  # Attention
            priority=MessagePriority.HIGH
        )
        
        await self._neurotransmitter_broadcast(proposal_message)
        
    async def broadcast_decision(self, proposal_id: str, decision: Dict):
        """Broadcast consensus decision"""
        decision_message = NeuralMessage(
            source='consensus_engine',
            content={'proposal_id': proposal_id, 'decision': decision},
            neurotransmitter=Neurotransmitter.DOPAMINE,  # Reward/decision
            priority=MessagePriority.CRITICAL
        )
        
        await self._global_broadcast(decision_message)
        
    def get_module_status(self) -> Dict[str, Any]:
        """Get status of all modules"""
        status = {}
        current_time = time.time()
        
        for module in self.modules:
            last_seen = self.module_last_seen.get(module, 0)
            if current_time - last_seen > 10:
                self.module_states[module] = 'inactive'
            elif current_time - last_seen > 5:
                self.module_states[module] = 'slow'
                
            status[module] = {
                'state': self.module_states[module],
                'last_seen': last_seen,
                'in_refractory': self._in_refractory(module),
                'connections': dict(self.plasticity.connections[module])
            }
            
        return status
        
    async def run(self, host='127.0.0.1', port=7860):
        """Run the message bus server"""
        config = uvicorn.Config(app=self.app, host=host, port=port)
        server = uvicorn.Server(config)
        await server.serve()

# ============= Module Connection Helper =============

class ModuleConnector:
    """Helper class for modules to connect to the bus"""
    
    def __init__(self, module_name: str, bus_url: str = "ws://127.0.0.1:7860"):
        self.module_name = module_name
        self.bus_url = f"{bus_url}/ws/{module_name}"
        self.websocket = None
        self.running = False
        
    async def connect(self):
        """Connect to the message bus"""
        # Implementation would use websocket-client
        pass
        
    async def send_message(self, 
                          content: Any,
                          target: Optional[str] = None,
                          neurotransmitter: Neurotransmitter = Neurotransmitter.GLUTAMATE,
                          priority: MessagePriority = MessagePriority.NORMAL):
        """Send a message through the bus"""
        if self.websocket:
            message = {
                'type': 'message',
                'content': {
                    'source': self.module_name,
                    'target': target,
                    'content': content,
                    'neurotransmitter': neurotransmitter.value,
                    'priority': priority.value,
                    'timestamp': time.time()
                }
            }
            await self.websocket.send(json.dumps(message))
            
    async def vote(self, proposal_id: str, decision: str, confidence: float = 1.0):
        """Vote on a proposal"""
        # Send vote via HTTP POST
        pass
        
    async def heartbeat_loop(self):
        """Send periodic heartbeats"""
        while self.running:
            if self.websocket:
                await self.websocket.send(json.dumps({'type': 'heartbeat'}))
            await asyncio.sleep(1.0)

if __name__ == "__main__":
    # Run the bus
    bus = DemocraticMessageBus()
    asyncio.run(bus.run())
