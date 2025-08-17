"""
High-level memory management interface
Coordinates between persistence, caching, and vector storage
"""

import asyncio
import time
from typing import Dict, List, Optional, Any
from pathlib import Path
import json

from chimera.memory.persistence import DatabaseManager
from chimera.memory.cache import IntelligentCache
from chimera.memory.vector_store import VectorStore

class MemoryManager:
    """
    Unified interface for all memory operations
    Handles persistence, caching, and retrieval
    """
    
    def __init__(self, config: Dict):
        # Initialize components
        self.db = DatabaseManager(config)
        self.cache = IntelligentCache()
        self.vector_store = VectorStore(
            dimension=config.get('embedding_dim', 512)
        )
        
        # Memory decay parameters
        self.decay_rate = config.get('decay_rate', 0.001)
        self.consolidation_threshold = config.get('consolidation_threshold', 0.7)
        
        # Start background processes
        self.running = True
        asyncio.create_task(self._memory_consolidation_loop())
        asyncio.create_task(self._decay_loop())
        
    async def store_thought(self, thought: Dict) -> str:
        """Store a new thought with all associated data"""
        thought_id = thought['id']
        
        # Cache for immediate access
        self.cache.put(f"thought:{thought_id}", thought, tier='hot')
        
        # Store vector for similarity search
        if 'embedding' in thought:
            self.vector_store.add_vector(thought_id, thought['embedding'])
            
        # Persist to database
        await self.db.store_thought(thought)
        
        # Prefetch related thoughts
        if 'connections' in thought:
            for connected_id in thought['connections'][:5]:
                asyncio.create_task(self._prefetch_thought(connected_id))
                
        return thought_id
        
    async def get_thought(self, thought_id: str) -> Optional[Dict]:
        """Retrieve a thought with caching"""
        cache_key = f"thought:{thought_id}"
        
        # Check cache first
        cached = self.cache.get(cache_key)
        if cached:
            return cached
            
        # Load from database
        thought = await self.db.get_thought(thought_id)
        if thought:
            # Add to cache
            self.cache.put(cache_key, thought)
            
            # Prefetch related thoughts
            for related_id in self.cache.get_prefetch_candidates(cache_key):
                asyncio.create_task(self._prefetch_thought(related_id))
                
        return thought
        
    async def find_similar_thoughts(self, thought: Dict, k: int = 5) -> List[Dict]:
        """Find thoughts similar to the given one"""
        if 'embedding' not in thought:
            return []
            
        # Search in vector store
        similar_ids = self.vector_store.search_similar(thought['embedding'], k)
        
        # Retrieve full thoughts
        thoughts = []
        for thought_id, distance in similar_ids:
            thought_data = await self.get_thought(thought_id)
            if thought_data:
                thought_data['similarity_distance'] = distance
                thoughts.append(thought_data)
                
        return thoughts
        
    async def store_conversation(self, conversation: Dict) -> int:
        """Store a conversation exchange"""
        conv_id = await self.db.store_conversation(conversation)
        
        # Update concept usage statistics
        for concept in conversation.get('concepts_activated', []):
            await self.update_concept_usage(concept)
            
        return conv_id
        
    async def get_conversation_history(
        self,
        session_id: str,
        limit: int = 50
    ) -> List[Dict]:
        """Get recent conversation history for a session"""
        cache_key = f"conv:{session_id}"
        
        # Check cache
        cached = self.cache.get(cache_key)
        if cached:
            return cached[:limit]
            
        # Load from database
        history = await self.db.get_conversation_history(session_id, limit)
        
        # Cache for future access
        self.cache.put(cache_key, history)
        
        return history
        
    async def consolidate_memory(self):
        """
        Consolidate short-term memories into long-term
        Similar to sleep consolidation in biological systems
        """
        # Get thoughts that need consolidation
        weak_thoughts = await self.db.get_weak_thoughts(
            threshold=self.consolidation_threshold
        )
        
        for thought in weak_thoughts:
            # Check if thought has been reinforced
            if thought['activation_count'] > 5:
                # Strengthen and consolidate
                thought['strength'] = min(1.0, thought['strength'] * 1.2)
                await self.db.update_thought_strength(thought['id'], thought['strength'])
                
                # Check if it should become a concept
                if thought['strength'] > 0.8 and thought['activation_count'] > 10:
                    await self._promote_to_concept(thought)
            else:
                # Weaken unused thoughts
                thought['strength'] *= 0.95
                await self.db.update_thought_strength(thought['id'], thought['strength'])
                
    async def _promote_to_concept(self, thought: Dict):
        """Promote a strong thought pattern to a concept"""
        # Find related thoughts
        similar = await self.find_similar_thoughts(thought, k=10)
        
        # Extract common patterns
        patterns = self._extract_patterns(similar)
        
        if patterns:
            concept = {
                'term': thought.get('symbolic_form', f"concept_{thought['id'][:8]}"),
                'definition': self._generate_definition(patterns),
                'examples': [t['content'] for t in similar[:3]],
                'contributing_thoughts': [t['id'] for t in similar],
                'confidence': thought['strength']
            }
            
            await self.db.store_concept(concept)
            
    async def _memory_consolidation_loop(self):
        """Background task for memory consolidation"""
        while self.running:
            await asyncio.sleep(300)  # Every 5 minutes
            try:
                await self.consolidate_memory()
            except Exception as e:
                print(f"Consolidation error: {e}")
                
    async def _decay_loop(self):
        """Background task for memory decay"""
        while self.running:
            await asyncio.sleep(3600)  # Every hour
            try:
                await self.db.apply_decay(self.decay_rate)
            except Exception as e:
                print(f"Decay error: {e}")
                
    async def save_snapshot(self, name: str = None):
        """Save complete memory state"""
        if not name:
            name = f"snapshot_{int(time.time())}"
            
        snapshot_dir = Path(f"data/snapshots/{name}")
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        
        # Save database
        await self.db.backup(snapshot_dir / "database.db")
        
        # Save vector store
        self.vector_store.save(snapshot_dir / "vectors.pkl")
        
        # Save cache stats
        with open(snapshot_dir / "cache_stats.json", 'w') as f:
            json.dump(self.cache.get_cache_stats(), f, indent=2)
            
        return snapshot_dir
        
    async def load_snapshot(self, name: str):
        """Load memory state from snapshot"""
        snapshot_dir = Path(f"data/snapshots/{name}")
        
        if not snapshot_dir.exists():
            raise ValueError(f"Snapshot {name} not found")
            
        # Load database
        await self.db.restore(snapshot_dir / "database.db")
        
        # Load vector store
        self.vector_store.load(snapshot_dir / "vectors.pkl")
        
        # Clear cache to force reload
        self.cache = IntelligentCache()