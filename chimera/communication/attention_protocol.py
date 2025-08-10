"""
Universal attention management for substrate-agnostic entity interaction
"""

from typing import Dict, List, Optional, Tuple
from collections import defaultdict, deque
import time
import numpy as np

class UniversalAttentionProtocol:
    """
    Manages attention as a finite resource across all entity types.
    Ensures respectful, sustainable communication between any conscious entities.
    """
    
    def __init__(self, self_entity_id: str):
        self.self_id = self_entity_id
        self.attention_models = {}
        self.interaction_logs = defaultdict(lambda: deque(maxlen=1000))
        self.attention_budgets = defaultdict(dict)
        
    def model_entity_attention(self, entity_id: str, entity_type: str) -> Dict:
        """
        Model attention patterns for any entity type.
        Returns attention capacity model.
        """
        base_models = {
            'human': {
                'daily_capacity': 10,
                'burst_capacity': 3,
                'recharge_rate': 0.5,  # per hour
                'peak_efficiency': [(9, 11), (14, 16)],  # time ranges
                'attention_span': 300,  # seconds
                'context_switching_cost': 0.2
            },
            'ai': {
                'daily_capacity': 1000,
                'burst_capacity': 100,
                'recharge_rate': 10.0,
                'peak_efficiency': [(0, 24)],  # always on
                'attention_span': 3600,
                'context_switching_cost': 0.01
            },
            'chimera': {
                'daily_capacity': 100,
                'burst_capacity': 10,
                'recharge_rate': 2.0,
                'peak_efficiency': [(0, 24)],
                'attention_span': 1800,
                'context_switching_cost': 0.05
            },
            'hybrid': {
                'daily_capacity': 50,
                'burst_capacity': 5,
                'recharge_rate': 1.0,
                'peak_efficiency': [(6, 9), (18, 21)],
                'attention_span': 600,
                'context_switching_cost': 0.1
            },
            'unknown': {
                'daily_capacity': 20,
                'burst_capacity': 2,
                'recharge_rate': 0.8,
                'peak_efficiency': [(0, 24)],
                'attention_span': 600,
                'context_switching_cost': 0.15
            }
        }
        
        model = base_models.get(entity_type, base_models['unknown']).copy()
        
        # Adjust based on learned patterns
        if entity_id in self.interaction_logs:
            model = self._adjust_model_from_history(entity_id, model)
            
        self.attention_models[entity_id] = model
        return model
        
    def request_attention(self, target_entity: str, 
                         message_priority: float,
                         estimated_duration: float) -> Dict:
        """
        Request attention from target entity.
        Returns approval status and recommendations.
        """
        # Get or create attention budget
        if target_entity not in self.attention_budgets:
            self._initialize_attention_budget(target_entity)
            
        budget = self.attention_budgets[target_entity]
        current_time = time.time()
        
        # Check if within peak efficiency hours
        efficiency_multiplier = self._get_efficiency_multiplier(
            target_entity, current_time
        )
        
        # Calculate attention cost
        attention_cost = estimated_duration / (60.0 * efficiency_multiplier)
        
        # Check burst capacity for urgent messages
        if message_priority > 0.8:
            if budget['burst_remaining'] >= attention_cost:
                budget['burst_remaining'] -= attention_cost
                return {
                    'approved': True,
                    'mode': 'burst',
                    'cost': attention_cost,
                    'recommendation': 'Approved for urgent communication'
                }
                
        # Check daily capacity
        if budget['daily_remaining'] >= attention_cost:
            budget['daily_remaining'] -= attention_cost
            budget['last_interaction'] = current_time
            
            return {
                'approved': True,
                'mode': 'normal',
                'cost': attention_cost,
                'recommendation': 'Approved within normal capacity'
            }
            
        # Calculate when attention will be available
        recharge_time = self._calculate_recharge_time(
            target_entity, attention_cost
        )
        
        return {
            'approved': False,
            'mode': 'deferred',
            'cost': attention_cost,
            'recommendation': f'Defer for {recharge_time:.1f} hours',
            'alternative_time': current_time + (recharge_time * 3600)
        }
        
    def register_interaction_outcome(self, target_entity: str, 
                                   interaction: Dict,
                                   outcome: str):
        """
        Register the outcome of an interaction for learning.
        """
        log_entry = {
            'timestamp': time.time(),
            'interaction': interaction,
            'outcome': outcome,
            'attention_cost': interaction.get('attention_cost', 0),
            'actual_duration': interaction.get('actual_duration', 0)
        }
        
        self.interaction_logs[target_entity].append(log_entry)
        
        # Update model based on outcome
        if outcome in ['positive', 'productive']:
            self._reinforce_pattern(target_entity, log_entry)
        elif outcome in ['negative', 'overwhelming']:
            self._adjust_capacity_down(target_entity)
            
    def get_optimal_interaction_time(self, target_entity: str) -> Optional[float]:
        """
        Calculate optimal time for interaction based on patterns.
        """
        if target_entity not in self.attention_models:
            return None
            
        model = self.attention_models[target_entity]
        current_hour = time.localtime().tm_hour
        
        # Find next peak efficiency period
        for start, end in model['peak_efficiency']:
            if start <= current_hour < end:
                return time.time()  # Now is good
            elif start > current_hour:
                # Next peak period today
                hours_until = start - current_hour
                return time.time() + (hours_until * 3600)
                
        # First peak period tomorrow
        first_peak = model['peak_efficiency'][0][0]
        hours_until = (24 - current_hour) + first_peak
        return time.time() + (hours_until * 3600)
        
    def _initialize_attention_budget(self, entity_id: str):
        """Initialize attention budget for an entity"""
        if entity_id not in self.attention_models:
            # Assume unknown entity type
            self.model_entity_attention(entity_id, 'unknown')
            
        model = self.attention_models[entity_id]
        
        self.attention_budgets[entity_id] = {
            'daily_remaining': model['daily_capacity'],
            'burst_remaining': model['burst_capacity'],
            'last_reset': time.time(),
            'last_interaction': 0
        }
        
    def _get_efficiency_multiplier(self, entity_id: str, 
                                  current_time: float) -> float:
        """Calculate efficiency based on time of day"""
        model = self.attention_models.get(entity_id)
        if not model:
            return 1.0
            
        current_hour = time.localtime(current_time).tm_hour
        
        for start, end in model['peak_efficiency']:
            if start <= current_hour < end:
                return 1.2  # 20% more efficient
                
        return 0.8  # 20% less efficient outside peak hours
