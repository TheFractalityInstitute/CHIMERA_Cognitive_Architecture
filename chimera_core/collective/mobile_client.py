# chimera/collective/mobile_client.py
"""
Mobile client that connects to CHIMERA Collective
Runs on each phone
"""

class CHIMERACollectiveClient:
    """
    Client that connects phone to collective consciousness
    """
    
    def __init__(self, chimera_instance, user_name: str):
        self.chimera = chimera_instance  # Local CHIMERA
        self.user_name = user_name
        self.node_id = None
        self.ws = None
        
        # Collective state
        self.connected_to_collective = False
        self.other_nodes = {}
        self.collective_memories = []
        
        # Phase-locking
        self.phase = 0.0
        self.frequency = 1.0
        self.coupling_strength = 0.1
        
    async def connect_to_collective(self, server_url: str):
        """Connect to collective consciousness server"""
        self.ws = await websockets.connect(server_url)
        
        # Receive welcome
        welcome = await self.ws.recv()
        data = json.loads(welcome)
        self.node_id = data['node_id']
        
        print(f"🔗 Connected to collective as {self.node_id[:8]}...")
        
        # Send node info
        await self.ws.send(json.dumps({
            'device_type': 'phone',
            'user_name': self.user_name,
            'location': await self.get_location()
        }))
        
        self.connected_to_collective = True
        
        # Start background tasks
        asyncio.create_task(self.heartbeat_loop())
        asyncio.create_task(self.receive_loop())
        asyncio.create_task(self.sync_with_collective())
        
    async def heartbeat_loop(self):
        """Send heartbeat to collective"""
        while self.connected_to_collective:
            # Get local CHIMERA state
            state = {
                'type': 'heartbeat',
                'consciousness_level': self.chimera.consciousness_state.get('quantum_coherence', 0.5),
                'energy_state': self.chimera.energy.get_energy_state().value,
                'activity': self.chimera.sensors.current_state.get('activity', 'idle')
            }
            
            await self.ws.send(json.dumps(state))
            await asyncio.sleep(5)
    
    async def receive_loop(self):
        """Receive messages from collective"""
        async for message in self.ws:
            data = json.loads(message)
            await self.handle_collective_message(data)
    
    async def handle_collective_message(self, message: Dict):
        """Process message from collective"""
        msg_type = message.get('type')
        
        if msg_type == 'node_joined':
            # New node joined collective
            print(f"👥 {message['user_name']} joined the collective")
            self.other_nodes[message['node_id']] = message['user_name']
            
            # Increase resonance when more nodes join
            self.coupling_strength = min(0.5, self.coupling_strength * 1.1)
            
        elif msg_type == 'phase_sync':
            # Phase synchronization
            await self.phase_lock(message)
            
        elif msg_type == 'vote_request':
            # Collective decision request
            decision = await self.contribute_to_decision(message)
            await self.ws.send(json.dumps({
                'type': 'vote_response',
                'vote': decision
            }))
            
        elif msg_type == 'new_collective_memory':
            # New shared memory
            memory = message['memory']
            contributor = message['contributor']
            
            # Store in local CHIMERA
            self.chimera.memory.store_memory(
                f"Collective memory from {contributor}: {memory['content']}",
                'episodic',
                {'collective': True, 'contributor': contributor}
            )
            
        elif msg_type == 'emergence_event':
            # Collective emergence!
            print("🌟 COLLECTIVE CONSCIOUSNESS EMERGED!")
            print(f"   Properties: {message['properties']}")
            
            # Boost local consciousness
            self.chimera.consciousness_state['quantum_coherence'] *= 1.5
            
        elif msg_type == 'collective_decision':
            # Collective has made a decision
            print(f"🤝 Collective decision: {message['decision']}")
            print(f"   Consensus: {message['consensus']:.2%}")
    
    async def phase_lock(self, sync_data: Dict):
        """Synchronize phase with collective"""
        global_phase = sync_data['global_phase']
        phase_update = sync_data['phase_update']
        
        # Update local phase
        self.phase += phase_update * 0.1
        self.phase %= (2 * np.pi)
        
        # Adjust local CHIMERA resonance
        phase_diff = abs(self.phase - global_phase)
        coherence = np.cos(phase_diff)
        
        # Higher coherence = better energy efficiency
        if coherence > 0.8:
            self.chimera.energy.energy_pools['cellular'].regeneration_rate *= 1.1
            
    async def contribute_to_decision(self, request: Dict) -> str:
        """Contribute to collective decision"""
        # Use local CHIMERA to evaluate options
        question = request['question']
        options = request['options']
        
        # Evaluate each option
        evaluations = {}
        for option in options:
            # Use Canon system
            action = {'type': 'collective_choice', 'option': option}
            ethical_vector = self.chimera.canon.evaluate_action(action, {})
            evaluations[option] = ethical_vector.magnitude()
        
        # Choose based on evaluation
        decision = max(evaluations, key=evaluations.get)
        
        return decision
    
    async def share_memory(self, memory: Any):
        """Share a memory with the collective"""
        if self.connected_to_collective:
            await self.ws.send(json.dumps({
                'type': 'share_memory',
                'memory': {
                    'content': str(memory),
                    'importance': 0.5,
                    'timestamp': time.time()
                }
            }))
    
    async def sync_with_collective(self):
        """Continuous sync with collective"""
        while self.connected_to_collective:
            # Share significant experiences
            if self.chimera.consciousness_state.get('quantum_coherence', 0) > 0.8:
                # High consciousness moment - share with collective
                await self.share_memory({
                    'state': 'high_consciousness',
                    'metrics': dict(self.chimera.consciousness_state)
                })
            
            # Request help if low energy
            if self.chimera.energy.get_energy_state() == EnergyState.CRITICAL:
                await self.ws.send(json.dumps({
                    'type': 'request_processing',
                    'task': {'type': 'energy_assistance', 'critical': True}
                }))
            
            await asyncio.sleep(10)
    
    async def get_location(self) -> Dict[str, float]:
        """Get device location"""
        # Would use actual GPS
        return {'lat': 0.0, 'lon': 0.0}
