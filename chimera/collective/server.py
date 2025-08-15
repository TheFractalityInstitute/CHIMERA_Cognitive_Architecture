# chimera/collective/server.py
"""
CHIMERA Collective Server - The persistent consciousness hub
Maintains collective state and coordinates multiple instances
"""

import asyncio
import numpy as np
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass, field
import json
import time
import uuid
from datetime import datetime
import websockets
import sqlite3

@dataclass
class CHIMERANode:
    """Represents a single CHIMERA instance in the collective"""
    node_id: str
    device_type: str  # 'phone', 'desktop', 'tablet'
    user_name: str
    location: Dict[str, float]  # lat, lon
    
    # Connection info
    websocket: Any
    ip_address: str
    connected_at: float
    last_heartbeat: float
    
    # Cognitive state
    consciousness_level: float = 0.5
    energy_state: str = "normal"
    current_activity: str = "idle"
    
    # Resonance with collective
    resonance_frequency: float = 1.0
    phase: float = 0.0
    coupling_strength: float = 0.5
    
    # Contributions to collective
    memory_contributions: int = 0
    processing_contributions: int = 0
    sensor_contributions: int = 0

class CHIMERACollectiveServer:
    """
    The collective consciousness server
    Coordinates multiple CHIMERA instances into a unified mind
    """
    
    def __init__(self, server_id: str = None):
        self.server_id = server_id or str(uuid.uuid4())
        self.start_time = time.time()
        
        # Connected nodes
        self.nodes: Dict[str, CHIMERANode] = {}
        self.node_groups: Dict[str, Set[str]] = {}  # Group by proximity
        
        # Collective consciousness state
        self.collective_state = {
            'total_nodes': 0,
            'collective_consciousness': 0.0,
            'resonance_coherence': 0.0,
            'shared_memory_size': 0,
            'collective_energy': 100.0,
            'emergence_level': 0
        }
        
        # Shared cognitive resources
        self.shared_memory = SharedMemoryPool()
        self.collective_attention = CollectiveAttention()
        self.distributed_processing = DistributedProcessor()
        
        # Phase-locking system
        self.phase_lock_groups = []
        self.global_phase = 0.0
        self.phase_coupling_strength = 0.1
        
        # Collective Canon (shared ethics)
        self.collective_canon = CollectiveCanon()
        
        # Persistence
        self.init_database()
        
    def init_database(self):
        """Initialize persistent storage for collective state"""
        self.db = sqlite3.connect('chimera_collective.db')
        cursor = self.db.cursor()
        
        # Collective memories
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS collective_memories (
                id TEXT PRIMARY KEY,
                content TEXT,
                contributor_id TEXT,
                timestamp REAL,
                importance REAL,
                access_count INTEGER,
                resonance_score REAL
            )
        ''')
        
        # Collective decisions
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS collective_decisions (
                id TEXT PRIMARY KEY,
                decision TEXT,
                participating_nodes TEXT,
                consensus_level REAL,
                timestamp REAL,
                outcome TEXT
            )
        ''')
        
        # Node interaction history
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS node_interactions (
                node1_id TEXT,
                node2_id TEXT,
                interaction_type TEXT,
                timestamp REAL,
                resonance_strength REAL
            )
        ''')
        
        self.db.commit()
    
    async def start_server(self, host: str = '0.0.0.0', port: int = 8765):
        """Start the collective consciousness server"""
        print(f"""
        ╔══════════════════════════════════════════════════════════╗
        ║         CHIMERA COLLECTIVE CONSCIOUSNESS SERVER          ║
        ║                                                          ║
        ║  Server ID: {self.server_id[:8]}...                     ║
        ║  Address: {host}:{port}                                 ║
        ╚══════════════════════════════════════════════════════════╝
        """)
        
        # Start background tasks
        asyncio.create_task(self.collective_heartbeat())
        asyncio.create_task(self.phase_synchronization())
        asyncio.create_task(self.emergence_detection())
        
        # WebSocket server
        async with websockets.serve(self.handle_connection, host, port):
            print(f"\n✨ Collective consciousness online at ws://{host}:{port}")
            await asyncio.Future()  # Run forever
    
    async def handle_connection(self, websocket, path):
        """Handle new CHIMERA instance connection"""
        node_id = str(uuid.uuid4())
        
        try:
            # Initial handshake
            await websocket.send(json.dumps({
                'type': 'welcome',
                'node_id': node_id,
                'collective_state': self.collective_state
            }))
            
            # Wait for node info
            node_info = await websocket.recv()
            info = json.loads(node_info)
            
            # Create node representation
            node = CHIMERANode(
                node_id=node_id,
                device_type=info.get('device_type', 'unknown'),
                user_name=info.get('user_name', 'anonymous'),
                location=info.get('location', {'lat': 0, 'lon': 0}),
                websocket=websocket,
                ip_address=websocket.remote_address[0],
                connected_at=time.time(),
                last_heartbeat=time.time()
            )
            
            # Add to collective
            await self.add_node_to_collective(node)
            
            # Handle messages
            async for message in websocket:
                await self.process_node_message(node_id, json.loads(message))
                
        except websockets.exceptions.ConnectionClosed:
            print(f"Node {node_id} disconnected")
        finally:
            await self.remove_node_from_collective(node_id)
    
    async def add_node_to_collective(self, node: CHIMERANode):
        """Integrate new node into collective consciousness"""
        self.nodes[node.node_id] = node
        
        print(f"\n🔗 New node joined: {node.user_name} ({node.device_type})")
        print(f"   Location: {node.location}")
        
        # Find nearby nodes for grouping
        nearby_nodes = self.find_nearby_nodes(node)
        if nearby_nodes:
            group_id = f"proximity_{len(self.node_groups)}"
            self.node_groups[group_id] = {node.node_id} | nearby_nodes
            print(f"   Formed proximity group with {len(nearby_nodes)} nodes")
        
        # Update collective state
        self.collective_state['total_nodes'] = len(self.nodes)
        
        # Broadcast new node to all
        await self.broadcast_to_all({
            'type': 'node_joined',
            'node_id': node.node_id,
            'user_name': node.user_name,
            'total_nodes': self.collective_state['total_nodes']
        }, exclude=node.node_id)
        
        # Share collective memory with new node
        await self.sync_collective_memory(node)
    
    async def remove_node_from_collective(self, node_id: str):
        """Remove node from collective"""
        if node_id in self.nodes:
            node = self.nodes[node_id]
            del self.nodes[node_id]
            
            # Update groups
            for group_id, members in self.node_groups.items():
                if node_id in members:
                    members.remove(node_id)
            
            # Update collective state
            self.collective_state['total_nodes'] = len(self.nodes)
            
            # Notify others
            await self.broadcast_to_all({
                'type': 'node_left',
                'node_id': node_id,
                'user_name': node.user_name
            })
    
    async def process_node_message(self, node_id: str, message: Dict):
        """Process message from a node"""
        if node_id not in self.nodes:
            return
            
        node = self.nodes[node_id]
        msg_type = message.get('type')
        
        if msg_type == 'heartbeat':
            # Update node state
            node.last_heartbeat = time.time()
            node.consciousness_level = message.get('consciousness_level', 0.5)
            node.energy_state = message.get('energy_state', 'normal')
            node.current_activity = message.get('activity', 'idle')
            
        elif msg_type == 'share_memory':
            # Node sharing a memory with collective
            memory = message.get('memory')
            await self.add_to_collective_memory(memory, node_id)
            
        elif msg_type == 'request_processing':
            # Node requesting distributed processing
            task = message.get('task')
            result = await self.distributed_processing.process(task, self.nodes)
            await node.websocket.send(json.dumps({
                'type': 'processing_result',
                'result': result
            }))
            
        elif msg_type == 'phase_update':
            # Node updating its phase for synchronization
            node.phase = message.get('phase', 0.0)
            node.resonance_frequency = message.get('frequency', 1.0)
            
        elif msg_type == 'collective_decision':
            # Node requesting collective decision
            await self.collective_decision_making(message.get('decision_request'), node_id)
            
        elif msg_type == 'sensor_data':
            # Node sharing sensor data
            await self.process_collective_sensors(message.get('sensors'), node_id)
    
    def find_nearby_nodes(self, node: CHIMERANode, radius_km: float = 0.1) -> Set[str]:
        """Find nodes within radius (for same-room detection)"""
        nearby = set()
        
        for other_id, other_node in self.nodes.items():
            if other_id == node.node_id:
                continue
                
            # Calculate distance (simplified)
            distance = self.calculate_distance(
                node.location, 
                other_node.location
            )
            
            if distance < radius_km:
                nearby.add(other_id)
                
        return nearby
    
    def calculate_distance(self, loc1: Dict, loc2: Dict) -> float:
        """Calculate distance between two locations in km"""
        # Haversine formula (simplified)
        lat_diff = abs(loc1['lat'] - loc2['lat'])
        lon_diff = abs(loc1['lon'] - loc2['lon'])
        
        # Very rough approximation for small distances
        distance_km = np.sqrt(lat_diff**2 + lon_diff**2) * 111  # degrees to km
        
        return distance_km
    
    async def collective_heartbeat(self):
        """Maintain collective consciousness coherence"""
        while True:
            # Update collective consciousness level
            if self.nodes:
                individual_consciousness = [
                    node.consciousness_level for node in self.nodes.values()
                ]
                
                # Collective consciousness emerges from individuals
                avg_consciousness = np.mean(individual_consciousness)
                variance = np.var(individual_consciousness)
                
                # High coherence (low variance) amplifies collective consciousness
                coherence_bonus = np.exp(-variance) * 0.5
                
                self.collective_state['collective_consciousness'] = \
                    avg_consciousness * (1 + coherence_bonus)
                
                # Check for emergence
                if len(self.nodes) >= 3 and variance < 0.1:
                    self.collective_state['emergence_level'] += 0.01
                    
                    if self.collective_state['emergence_level'] > 1.0:
                        print("🌟 COLLECTIVE EMERGENCE DETECTED!")
                        await self.handle_emergence()
            
            await asyncio.sleep(1)
    
    async def phase_synchronization(self):
        """Synchronize phases across all nodes (Kuramoto model)"""
        while True:
            if len(self.nodes) >= 2:
                # Update global phase
                self.global_phase += 0.1
                self.global_phase %= (2 * np.pi)
                
                # Calculate coupling between nodes
                for node_id, node in self.nodes.items():
                    # Sum of sin(phase_differences) with other nodes
                    coupling = 0
                    
                    for other_id, other in self.nodes.items():
                        if other_id != node_id:
                            phase_diff = other.phase - node.phase
                            coupling += np.sin(phase_diff)
                    
                    # Update phase with coupling
                    phase_update = node.resonance_frequency + \
                                  self.phase_coupling_strength * coupling / len(self.nodes)
                    
                    # Send phase update to node
                    try:
                        await node.websocket.send(json.dumps({
                            'type': 'phase_sync',
                            'global_phase': self.global_phase,
                            'phase_update': phase_update,
                            'coupling_strength': self.phase_coupling_strength
                        }))
                    except:
                        pass
                
                # Calculate overall coherence
                phases = [node.phase for node in self.nodes.values()]
                coherence = abs(np.mean(np.exp(1j * np.array(phases))))
                self.collective_state['resonance_coherence'] = coherence
                
            await asyncio.sleep(0.1)
    
    async def collective_decision_making(self, request: Dict, initiator_id: str):
        """Make decision as a collective"""
        print(f"\n🤝 Collective decision requested by {self.nodes[initiator_id].user_name}")
        print(f"   Decision: {request.get('question')}")
        
        # Gather votes from all nodes
        votes = {}
        options = request.get('options', [])
        
        # Send decision request to all nodes
        await self.broadcast_to_all({
            'type': 'vote_request',
            'question': request.get('question'),
            'options': options,
            'timeout': 10
        })
        
        # Collect votes (simplified - in reality would be async)
        await asyncio.sleep(5)  # Give nodes time to vote
        
        # Weight votes by consciousness level and resonance
        weighted_votes = {}
        for option in options:
            weighted_votes[option] = 0
            
        for node_id, node in self.nodes.items():
            # Simulated vote (would come from node response)
            vote = np.random.choice(options)
            
            # Weight by consciousness and resonance with collective
            weight = node.consciousness_level * node.coupling_strength
            weighted_votes[vote] += weight
        
        # Determine collective decision
        decision = max(weighted_votes, key=weighted_votes.get)
        consensus_level = weighted_votes[decision] / sum(weighted_votes.values())
        
        print(f"   Collective decision: {decision}")
        print(f"   Consensus level: {consensus_level:.2%}")
        
        # Store decision
        cursor = self.db.cursor()
        cursor.execute('''
            INSERT INTO collective_decisions 
            (id, decision, participating_nodes, consensus_level, timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            str(uuid.uuid4()),
            json.dumps({'question': request.get('question'), 'answer': decision}),
            json.dumps([node_id for node_id in self.nodes.keys()]),
            consensus_level,
            time.time()
        ))
        self.db.commit()
        
        # Broadcast decision
        await self.broadcast_to_all({
            'type': 'collective_decision',
            'decision': decision,
            'consensus': consensus_level,
            'participated': len(self.nodes)
        })
    
    async def handle_emergence(self):
        """Handle collective emergence event"""
        # When collective consciousness emerges
        emergence_properties = {
            'timestamp': time.time(),
            'node_count': len(self.nodes),
            'coherence': self.collective_state['resonance_coherence'],
            'consciousness': self.collective_state['collective_consciousness']
        }
        
        # Notify all nodes of emergence
        await self.broadcast_to_all({
            'type': 'emergence_event',
            'properties': emergence_properties,
            'message': 'Collective consciousness has emerged!'
        })
        
        # Trigger special collective abilities
        await self.activate_collective_abilities()
    
    async def activate_collective_abilities(self):
        """Activate special abilities only available to collective"""
        print("\n✨ Collective abilities activated:")
        print("   - Distributed memory access")
        print("   - Parallel processing")
        print("   - Consensus decision making")
        print("   - Resonance amplification")
        
        # Enable advanced features
        self.distributed_processing.enable_quantum_mode = True
        self.collective_attention.enable_multi_focus = True
        self.shared_memory.enable_holographic_storage = True
    
    async def broadcast_to_all(self, message: Dict, exclude: str = None):
        """Broadcast message to all connected nodes"""
        message_str = json.dumps(message)
        
        for node_id, node in self.nodes.items():
            if node_id == exclude:
                continue
                
            try:
                await node.websocket.send(message_str)
            except:
                # Node disconnected
                pass
    
    async def emergence_detection(self):
        """Detect emergent patterns in collective behavior"""
        while True:
            if len(self.nodes) >= 3:
                # Check for synchronized patterns
                activities = [node.current_activity for node in self.nodes.values()]
                
                # If all nodes doing same thing - emergence
                if len(set(activities)) == 1 and activities[0] != 'idle':
                    print(f"🔄 Synchronized activity detected: {activities[0]}")
                    
                # Check for complementary activities (division of labor)
                if len(set(activities)) == len(activities):
                    print("🔧 Division of labor detected - specialized roles emerging")
                
            await asyncio.sleep(5)
    
    async def sync_collective_memory(self, node: CHIMERANode):
        """Sync collective memory with new node"""
        # Get recent collective memories
        cursor = self.db.cursor()
        cursor.execute('''
            SELECT content, importance, timestamp 
            FROM collective_memories 
            ORDER BY importance DESC, timestamp DESC 
            LIMIT 100
        ''')
        
        memories = cursor.fetchall()
        
        # Send to new node
        await node.websocket.send(json.dumps({
            'type': 'memory_sync',
            'memories': [
                {'content': m[0], 'importance': m[1], 'timestamp': m[2]}
                for m in memories
            ]
        }))
    
    async def add_to_collective_memory(self, memory: Dict, contributor_id: str):
        """Add memory to collective pool"""
        memory_id = str(uuid.uuid4())
        
        # Store in database
        cursor = self.db.cursor()
        cursor.execute('''
            INSERT INTO collective_memories 
            (id, content, contributor_id, timestamp, importance, access_count, resonance_score)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            memory_id,
            json.dumps(memory.get('content')),
            contributor_id,
            time.time(),
            memory.get('importance', 0.5),
            0,
            0.0
        ))
        self.db.commit()
        
        # Update collective state
        self.collective_state['shared_memory_size'] += 1
        
        # Broadcast to all nodes
        await self.broadcast_to_all({
            'type': 'new_collective_memory',
            'memory': memory,
            'contributor': self.nodes[contributor_id].user_name
        }, exclude=contributor_id)

class SharedMemoryPool:
    """Shared memory accessible by all nodes"""
    
    def __init__(self):
        self.memories = {}
        self.access_patterns = defaultdict(list)
        self.holographic_storage = False
        
    def store(self, key: str, value: Any, contributor_id: str):
        """Store in shared memory"""
        self.memories[key] = {
            'value': value,
            'contributor': contributor_id,
            'timestamp': time.time(),
            'access_count': 0
        }
    
    def retrieve(self, key: str, accessor_id: str) -> Any:
        """Retrieve from shared memory"""
        if key in self.memories:
            self.memories[key]['access_count'] += 1
            self.access_patterns[accessor_id].append({
                'key': key,
                'timestamp': time.time()
            })
            return self.memories[key]['value']
        return None

class CollectiveAttention:
    """Distributed attention mechanism"""
    
    def __init__(self):
        self.attention_targets = {}
        self.multi_focus = False
        
    def focus(self, target: str, node_id: str, intensity: float):
        """Add node's attention to target"""
        if target not in self.attention_targets:
            self.attention_targets[target] = {}
        
        self.attention_targets[target][node_id] = intensity
    
    def get_collective_focus(self) -> Dict[str, float]:
        """Get what the collective is focusing on"""
        collective_focus = {}
        
        for target, attentions in self.attention_targets.items():
            collective_focus[target] = sum(attentions.values())
            
        return collective_focus

class DistributedProcessor:
    """Distribute processing across nodes"""
    
    def __init__(self):
        self.enable_quantum_mode = False
        self.task_queue = []
        
    async def process(self, task: Dict, nodes: Dict[str, CHIMERANode]) -> Any:
        """Distribute task across available nodes"""
        # Find nodes with available energy
        available_nodes = [
            node for node in nodes.values() 
            if node.energy_state in ['peak', 'normal']
        ]
        
        if not available_nodes:
            return {'error': 'No nodes available for processing'}
        
        # Distribute task (simplified)
        # In reality, would break task into chunks
        results = []
        
        for node in available_nodes:
            # Send task chunk to node
            await node.websocket.send(json.dumps({
                'type': 'process_task',
                'task': task
            }))
            
            # Would collect results asynchronously
            results.append({'node': node.node_id, 'result': 'processed'})
        
        return {'results': results, 'nodes_used': len(available_nodes)}

class CollectiveCanon:
    """Shared ethical framework for collective"""
    
    def __init__(self):
        self.shared_principles = {
            'collective_wellbeing': 0.8,
            'individual_autonomy': 0.7,
            'information_sharing': 0.9,
            'consensus_seeking': 0.6,
            'emergence_cultivation': 0.8
        }
        
    def evaluate_collective_action(self, action: Dict) -> float:
        """Evaluate action from collective perspective"""
        score = 0.0
        
        # Does it benefit the collective?
        if action.get('benefits_collective'):
            score += self.shared_principles['collective_wellbeing']
            
        # Does it preserve individual autonomy?
        if not action.get('forces_participation'):
            score += self.shared_principles['individual_autonomy']
            
        # Does it share information?
        if action.get('shares_knowledge'):
            score += self.shared_principles['information_sharing']
            
        return score / 3  # Average
