"""
Intelligent caching system for frequently accessed data
Implements LRU with predictive prefetching
"""

import time
import heapq
from collections import OrderedDict, defaultdict
from typing import Any, Optional, Dict, List, Tuple
import threading

class IntelligentCache:
    """
    Multi-tier caching system with access pattern learning
    """
    
    def __init__(self, hot_size: int = 100, warm_size: int = 1000):
        # Hot cache - most frequently accessed
        self.hot_cache = OrderedDict()
        self.hot_size = hot_size
        
        # Warm cache - recently accessed
        self.warm_cache = OrderedDict()
        self.warm_size = warm_size
        
        # Access statistics for predictive caching
        self.access_counts = defaultdict(int)
        self.access_times = defaultdict(list)
        self.co_access = defaultdict(set)  # Track what's accessed together
        
        # Thread safety
        self.lock = threading.RLock()
        
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache with automatic promotion"""
        with self.lock:
            # Check hot cache first
            if key in self.hot_cache:
                # Move to end (most recently used)
                self.hot_cache.move_to_end(key)
                self._record_access(key)
                return self.hot_cache[key]
                
            # Check warm cache
            if key in self.warm_cache:
                value = self.warm_cache[key]
                
                # Promote to hot cache if accessed frequently
                self.access_counts[key] += 1
                if self.access_counts[key] > 3:  # Threshold for promotion
                    self._promote_to_hot(key, value)
                    
                self.warm_cache.move_to_end(key)
                self._record_access(key)
                return value
                
            return None
            
    def put(self, key: str, value: Any, tier: str = 'warm'):
        """Add item to cache"""
        with self.lock:
            if tier == 'hot':
                self._add_to_hot(key, value)
            else:
                self._add_to_warm(key, value)
                
    def _add_to_hot(self, key: str, value: Any):
        """Add to hot cache with LRU eviction"""
        if key in self.hot_cache:
            self.hot_cache.move_to_end(key)
        else:
            self.hot_cache[key] = value
            if len(self.hot_cache) > self.hot_size:
                # Evict least recently used to warm cache
                evicted_key, evicted_value = self.hot_cache.popitem(last=False)
                self._add_to_warm(evicted_key, evicted_value)
                
    def _add_to_warm(self, key: str, value: Any):
        """Add to warm cache with LRU eviction"""
        if key in self.warm_cache:
            self.warm_cache.move_to_end(key)
        else:
            self.warm_cache[key] = value
            if len(self.warm_cache) > self.warm_size:
                # Evict least recently used
                self.warm_cache.popitem(last=False)
                
    def _promote_to_hot(self, key: str, value: Any):
        """Promote item from warm to hot cache"""
        del self.warm_cache[key]
        self._add_to_hot(key, value)
        
    def _record_access(self, key: str):
        """Record access patterns for predictive caching"""
        current_time = time.time()
        self.access_times[key].append(current_time)
        
        # Track co-access patterns (what's accessed together)
        # This helps prefetch related data
        for other_key in list(self.hot_cache.keys())[-5:]:  # Last 5 accessed
            if other_key != key:
                self.co_access[key].add(other_key)
                self.co_access[other_key].add(key)
                
    def get_prefetch_candidates(self, key: str, limit: int = 5) -> List[str]:
        """Get items likely to be accessed together with key"""
        candidates = list(self.co_access.get(key, set()))
        return candidates[:limit]
        
    def get_cache_stats(self) -> Dict:
        """Get cache performance statistics"""
        with self.lock:
            total_accesses = sum(self.access_counts.values())
            hot_hits = sum(1 for k in self.access_counts if k in self.hot_cache)
            warm_hits = sum(1 for k in self.access_counts if k in self.warm_cache)
            
            return {
                'hot_cache_size': len(self.hot_cache),
                'warm_cache_size': len(self.warm_cache),
                'total_accesses': total_accesses,
                'hot_hit_rate': hot_hits / max(1, total_accesses),
                'warm_hit_rate': warm_hits / max(1, total_accesses),
                'most_accessed': sorted(
                    self.access_counts.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
            }