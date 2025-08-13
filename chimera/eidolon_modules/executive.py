"""
CHIMERA Executive Eidolon Module v2.0
The prefrontal cortex analog - coordinates without dictating
Implements:
- Working memory management
- Goal hierarchies and planning
- Inhibitory control
- Conflict resolution
- Resource allocation
- Emergency override capabilities
"""

import asyncio
import time
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import json
import uuid

# Import from other CHIMERA modules
from chimera.core.message_bus import (
    NeuralMessage, 
    Neurotransmitter,
    MessagePriority,
    ModuleConnector
)

# ============= Executive Functions =============

class ExecutiveFunction(Enum):
    """Core executive functions based on neuroscience"""
    WORKING_MEMORY = "working_memory"          # Maintain active information
    INHIBITORY_CONTROL = "inhibitory_control"  # Suppress inappropriate responses
    COGNITIVE_FLEXIBILITY = "flexibility"      # Switch between tasks/strategies
    PLANNING = "planning"                      # Goal-directed behavior
    ATTENTION_CONTROL = "attention"           # Focus resources
    CONFLICT_MONITORING = "conflict"          # Detect and resolve conflicts
    ERROR_DETECTION = "error"                 # Monitor for mistakes
    DECISION_MAKING = "decision"              # Choose between options

@dataclass
class Goal:
    """Hierarchical goal representation"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    priority: float = 1.0
    parent_id: Optional[str] = None
    subgoals: List[str] = field(default_factory=list)
    status: str = "pending"  # pending, active, completed, failed
    created_at: float = field(default_factory=time.time)
    deadline: Optional[float] = None
    resources_required: Dict[str, float] = field(default_factory=dict)
    success_criteria: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        if self.deadline:
            return time.time() > self.deadline
        return False

@dataclass
class WorkingMemoryItem:
    """Item in working memory with decay"""
    content: Any
    timestamp: float = field(default_factory=time.time)
    relevance: float = 1.0
    decay_rate: float = 0.1
    source: str = ""
    
    def get_activation(self) -> float:
        """Calculate current activation level"""
        age = time.time() - self.timestamp
        return self.relevance * np.exp(-self.decay_rate * age)

# ============= Prefrontal Cortex Regions =============

class DorsolateralPFC:
    """
    Dorsolateral Prefrontal Cortex
    Handles: Working memory, cognitive flexibility, abstract reasoning
    """
    
    def __init__(self, capacity: int = 7):
        self.working_memory = deque(maxlen=capacity)
        self.memory_index = {}  # Fast lookup
        self.rehearsal_rate = 0.5  # How often to refresh memory
        self.last_rehearsal = time.time()
        
    def update(self, item: WorkingMemoryItem):
        """Add or update item in working memory"""
        # Remove items with low activation
        self._decay_memory()
        
        # Check if item already exists
        item_hash = hash(str(item.content))
        if item_hash in self.memory_index:
            # Refresh existing item
            old_item = self.memory_index[item_hash]
            old_item.timestamp = time.time()
            old_item.relevance = max(old_item.relevance, item.relevance)
        else:
            # Add new item
            self.working_memory.append(item)
            self.memory_index[item_hash] = item
            
    def _decay_memory(self):
        """Remove items with low activation"""
        threshold = 0.1
        to_remove = []
        
        for item in self.working_memory:
            if item.get_activation() < threshold:
                to_remove.append(item)
                
        for item in to_remove:
            self.working_memory.remove(item)
            item_hash = hash(str(item.content))
            if item_hash in self.memory_index:
                del self.memory_index[item_hash]
                
    def rehearse(self):
        """Rehearsal to maintain important items"""
        current_time = time.time()
        if current_time - self.last_rehearsal > self.rehearsal_rate:
            # Refresh high-relevance items
            for item in self.working_memory:
                if item.relevance > 0.7:
                    item.timestamp = current_time
            self.last_rehearsal = current_time
            
    def query(self, query: str) -> List[WorkingMemoryItem]:
        """Query working memory"""
        results = []
        query_lower = query.lower()
        
        for item in self.working_memory:
            if isinstance(item.content, dict):
                # Check if query matches any values
                for key, value in item.content.items():
                    if query_lower in str(value).lower():
                        results.append(item)
                        break
            elif query_lower in str(item.content).lower():
                results.append(item)
                
        # Sort by activation
        results.sort(key=lambda x: x.get_activation(), reverse=True)
        return results

class VentromedialPFC:
    """
    Ventromedial Prefrontal Cortex
    Handles: Value assessment, emotional regulation, social cognition
    """
    
    def __init__(self):
        self.value_estimates = {}  # Action -> expected value
        self.somatic_markers = {}  # Gut feelings about options
        self.social_context = {}
        self.moral_weights = {
            'harm_prevention': 0.9,
            'fairness': 0.7,
            'loyalty': 0.5,
            'authority': 0.3,
            'sanctity': 0.4
        }
        
    def evaluate_option(self, 
                        option: str, 
                        features: Dict[str, Any]) -> float:
        """
        Evaluate an option using somatic marker hypothesis
        Combines logical analysis with "gut feelings"
        """
        
        # Logical value (from past experience)
        logical_value = self.value_estimates.get(option, 0.5)
        
        # Emotional value (somatic markers)
        emotional_value = self._compute_somatic_marker(option, features)
        
        # Social value
        social_value = self._compute_social_value(option, features)
        
        # Moral value
        moral_value = self._compute_moral_value(option, features)
        
        # Weighted combination (vmPFC integrates these)
        total_value = (
            0.3 * logical_value +
            0.3 * emotional_value +
            0.2 * social_value +
            0.2 * moral_value
        )
        
        return total_value
        
    def _compute_somatic_marker(self, option: str, features: Dict) -> float:
        """Compute gut feeling about option"""
        # Check for danger signals
        if features.get('risk', 0) > 0.7:
            return 0.2  # Bad feeling
            
        # Check for reward signals
        if features.get('reward', 0) > 0.7:
            return 0.8  # Good feeling
            
        # Default neutral
        return 0.5
        
    def _compute_social_value(self, option: str, features: Dict) -> float:
        """Evaluate social implications"""
        if features.get('affects_others', False):
            if features.get('helps_others', False):
                return 0.9
            elif features.get('harms_others', False):
                return 0.1
        return 0.5
        
    def _compute_moral_value(self, option: str, features: Dict) -> float:
        """Evaluate moral implications"""
        score = 0.5
        
        for principle, weight in self.moral_weights.items():
            if principle in features:
                score += weight * (features[principle] - 0.5)
                
        return np.clip(score, 0, 1)

class AnteriorCingulate:
    """
    Anterior Cingulate Cortex
    Handles: Conflict monitoring, error detection, attention
    """
    
    def __init__(self):
        self.conflict_history = deque(maxlen=100)
        self.error_history = deque(maxlen=100)
        self.attention_focus = None
        self.conflict_threshold = 0.5
        
    def detect_conflict(self, 
                        options: List[Tuple[str, float]]) -> float:
        """
        Detect conflict between competing options
        High conflict when multiple options have similar high values
        """
        if len(options) < 2:
            return 0.0
            
        # Sort by value
        sorted_options = sorted(options, key=lambda x: x[1], reverse=True)
        
        # Calculate conflict based on similarity of top options
        if sorted_options[0][1] > 0:
            conflict = sorted_options[1][1] / sorted_options[0][1]
        else:
            conflict = 0
            
        # Record conflict
        self.conflict_history.append({
            'timestamp': time.time(),
            'conflict_level': conflict,
            'options': options
        })
        
        return conflict
        
    def detect_error(self, 
                     predicted: Any, 
                     actual: Any) -> float:
        """Detect prediction error"""
        if predicted is None or actual is None:
            return 0
            
        # Calculate error magnitude
        if isinstance(predicted, (int, float)) and isinstance(actual, (int, float)):
            error = abs(predicted - actual) / (abs(predicted) + 1e-6)
        else:
            error = 0 if predicted == actual else 1
            
        # Record error
        self.error_history.append({
            'timestamp': time.time(),
            'error': error,
            'predicted': predicted,
            'actual': actual
        })
        
        return error

# ============= The Executive Eidolon =============

class ExecutiveEidolon:
    """
    The Executive Module - Prefrontal Cortex of CHIMERA
    Coordinates without dictating, synthesizes inputs, makes decisions
    """
    
    def __init__(self, name: str = "Executive"):
        self.name = name
        self.role = "coordination_and_decision_making"
        
        # Prefrontal regions
        self.dlpfc = DorsolateralPFC()
        self.vmpfc = VentromedialPFC()
        self.acc = AnteriorCingulate()
        
        # Goal management
        self.goal_hierarchy = {}  # goal_id -> Goal
        self.active_goals = set()
        self.goal_stack = []  # For nested goal pursuit
        
        # Resource management
        self.resource_allocation = {
            'attention': 1.0,
            'memory': 1.0,
            'processing': 1.0,
            'energy': 1.0
        }
        
        # Module coordination
        self.module_expertise = {
            'sensory': ['perception', 'environment', 'detection'],
            'memory': ['recall', 'storage', 'episodic', 'semantic'],
            'language': ['communication', 'understanding', 'expression'],
            'logical': ['reasoning', 'calculation', 'analysis'],
            'interoceptive': ['internal', 'energy', 'health'],
            'social': ['interaction', 'emotion', 'empathy']
        }
        
        # Decision tracking
        self.decision_history = deque(maxlen=100)
        self.pending_decisions = {}
        
        # Inhibitory control
        self.suppressed_actions = set()
        self.inhibition_strength = defaultdict(float)
        
        # Emergency overrides
        self.emergency_mode = False
        self.emergency_protocols = {
            'danger': self._danger_protocol,
            'system_failure': self._system_failure_protocol,
            'ethical_violation': self._ethical_violation_protocol
        }
        
        # Bus connection
        self.bus = None
        self.connector = None
        
        # State
        self.current_situation = {}
        self.confidence_threshold = 0.6
        
    async def initialize(self, bus_url: str = "ws://127.0.0.1:7860"):
        """Initialize connection to message bus"""
        self.connector = ModuleConnector(self.name, bus_url)
        await self.connector.connect()
        
    # ============= Core Executive Functions =============
    
    async def deliberate(self, topic: str) -> Dict[str, Any]:
        """
        Form executive opinion on a topic
        Synthesizes information from working memory and other sources
        """
        
        # Update working memory with topic
        self.dlpfc.update(WorkingMemoryItem(
            content={'topic': topic, 'type': 'deliberation'},
            relevance=1.0,
            source='user_query'
        ))
        
        # Query working memory for relevant information
        relevant_items = self.dlpfc.query(topic)
        
        # Identify which modules should be consulted
        relevant_modules = self._identify_relevant_modules(topic)
        
        # Check for conflicts
        if self.acc.conflict_history:
            recent_conflict = self.acc.conflict_history[-1]
            if recent_conflict['conflict_level'] > self.acc.conflict_threshold:
                return self._handle_conflict_deliberation(topic, recent_conflict)
                
        # Generate executive synthesis
        opinion = self._synthesize_opinion(topic, relevant_items, relevant_modules)
        
        return {
            'module': self.name,
            'opinion': opinion,
            'confidence': self._calculate_confidence(relevant_items),
            'reasoning': self._explain_executive_reasoning(topic, relevant_items),
            'consulted_modules': relevant_modules,
            'working_memory_items': len(relevant_items)
        }
        
    def _identify_relevant_modules(self, topic: str) -> List[str]:
        """Identify which modules are relevant to a topic"""
        relevant = []
        topic_lower = topic.lower()
        
        for module, keywords in self.module_expertise.items():
            for keyword in keywords:
                if keyword in topic_lower:
                    relevant.append(module)
                    break
                    
        return relevant
        
    def _synthesize_opinion(self, 
                           topic: str,
                           memory_items: List[WorkingMemoryItem],
                           modules: List[str]) -> str:
        """Synthesize executive opinion"""
        
        opinion_parts = [f"Executive synthesis on '{topic}':"]
        
        # Check if emergency protocols apply
        if self._check_emergency_conditions(topic):
            opinion_parts.append("EMERGENCY PROTOCOL ACTIVATED.")
            opinion_parts.append(self._get_emergency_response(topic))
            return " ".join(opinion_parts)
            
        # Normal synthesis
        if memory_items:
            opinion_parts.append(f"Drawing from {len(memory_items)} relevant memories.")
            
        if modules:
            opinion_parts.append(f"Recommend consulting: {', '.join(modules)}.")
            
        # Add current goal context
        if self.active_goals:
            top_goal = self._get_top_priority_goal()
            if top_goal:
                opinion_parts.append(f"Aligns with goal: {top_goal.description}.")
                
        # Add value assessment
        value = self.vmpfc.evaluate_option(topic, {'topic': topic})
        if value > 0.7:
            opinion_parts.append("High value proposition detected.")
        elif value < 0.3:
            opinion_parts.append("Low value - recommend alternative approach.")
            
        return " ".join(opinion_parts)
        
    def _calculate_confidence(self, memory_items: List[WorkingMemoryItem]) -> float:
        """Calculate confidence in executive decision"""
        
        if not memory_items:
            return 0.3  # Low confidence without memory support
            
        # Base confidence on memory activation
        activations = [item.get_activation() for item in memory_items]
        memory_confidence = np.mean(activations)
        
        # Adjust for recent errors
        if self.acc.error_history:
            recent_errors = [e['error'] for e in list(self.acc.error_history)[-5:]]
            error_penalty = np.mean(recent_errors)
            memory_confidence *= (1 - error_penalty * 0.5)
            
        # Adjust for resource availability
        resource_factor = np.mean(list(self.resource_allocation.values()))
        
        return memory_confidence * resource_factor
        
    def _explain_executive_reasoning(self, 
                                    topic: str,
                                    memory_items: List[WorkingMemoryItem]) -> str:
        """Explain executive reasoning process"""
        
        reasoning_parts = []
        
        # Working memory contribution
        if memory_items:
            strongest = max(memory_items, key=lambda x: x.get_activation())
            reasoning_parts.append(
                f"Primary consideration from {strongest.source}"
            )
            
        # Goal alignment
        if self.active_goals:
            reasoning_parts.append(
                f"Pursuing {len(self.active_goals)} active goals"
            )
            
        # Resource constraints
        constrained = [r for r, v in self.resource_allocation.items() if v < 0.3]
        if constrained:
            reasoning_parts.append(
                f"Limited by: {', '.join(constrained)}"
            )
            
        # Inhibitory control
        if self.suppressed_actions:
            reasoning_parts.append(
                f"Suppressing {len(self.suppressed_actions)} inappropriate responses"
            )
            
        return "; ".join(reasoning_parts) if reasoning_parts else "Intuitive assessment"
        
    # ============= Goal Management =============
    
    def create_goal(self, 
                   description: str,
                   priority: float = 1.0,
                   deadline: Optional[float] = None,
                   parent_id: Optional[str] = None) -> Goal:
        """Create a new goal"""
        
        goal = Goal(
            description=description,
            priority=priority,
            deadline=deadline,
            parent_id=parent_id
        )
        
        self.goal_hierarchy[goal.id] = goal
        
        # Add to parent's subgoals if applicable
        if parent_id and parent_id in self.goal_hierarchy:
            self.goal_hierarchy[parent_id].subgoals.append(goal.id)
            
        return goal
        
    def activate_goal(self, goal_id: str):
        """Activate a goal for pursuit"""
        if goal_id in self.goal_hierarchy:
            self.active_goals.add(goal_id)
            self.goal_stack.append(goal_id)
            
            # Update working memory
            goal = self.goal_hierarchy[goal_id]
            self.dlpfc.update(WorkingMemoryItem(
                content={'goal': goal.description, 'id': goal_id},
                relevance=goal.priority,
                source='goal_system'
            ))
            
    def complete_goal(self, goal_id: str, success: bool = True):
        """Mark goal as completed"""
        if goal_id in self.goal_hierarchy:
            goal = self.goal_hierarchy[goal_id]
            goal.status = 'completed' if success else 'failed'
            
            if goal_id in self.active_goals:
                self.active_goals.remove(goal_id)
                
            if goal_id in self.goal_stack:
                self.goal_stack.remove(goal_id)
                
    def _get_top_priority_goal(self) -> Optional[Goal]:
        """Get highest priority active goal"""
        if not self.active_goals:
            return None
            
        active = [self.goal_hierarchy[gid] for gid in self.active_goals]
        return max(active, key=lambda g: g.priority)
        
    # ============= Resource Allocation =============
    
    def allocate_resources(self, 
                          demands: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """
        Allocate limited resources among competing demands
        demands = {module: {resource: amount_needed}}
        """
        
        allocations = defaultdict(lambda: defaultdict(float))
        
        for resource, available in self.resource_allocation.items():
            # Calculate total demand
            total_demand = sum(
                demands.get(module, {}).get(resource, 0)
                for module in demands
            )
            
            if total_demand <= available:
                # Enough for everyone
                for module in demands:
                    if resource in demands[module]:
                        allocations[module][resource] = demands[module][resource]
            else:
                # Need to prioritize
                # Use module importance and goal relevance
                priorities = self._calculate_module_priorities(demands.keys())
                
                # Proportional allocation based on priority
                for module in demands:
                    if resource in demands[module]:
                        proportion = priorities.get(module, 0.1)
                        allocations[module][resource] = available * proportion
                        
        return dict(allocations)
        
    def _calculate_module_priorities(self, modules: List[str]) -> Dict[str, float]:
        """Calculate priority for each module"""
        priorities = {}
        
        # Base priorities
        base_priority = {
            'sensory': 0.2,      # Always need perception
            'executive': 0.3,    # Self-priority (metacognitive)
            'memory': 0.15,      # Important for continuity
            'logical': 0.15,     # Reasoning
            'language': 0.1,     # Communication
            'interoceptive': 0.1 # Self-monitoring
        }
        
        for module in modules:
            priorities[module] = base_priority.get(module, 0.1)
            
        # Adjust for current goals
        if self.active_goals:
            top_goal = self._get_top_priority_goal()
            if top_goal:
                relevant_modules = self._identify_relevant_modules(top_goal.description)
                for module in relevant_modules:
                    if module in priorities:
                        priorities[module] *= 1.5
                        
        # Normalize
        total = sum(priorities.values())
        if total > 0:
            priorities = {m: p/total for m, p in priorities.items()}
            
        return priorities
        
    # ============= Inhibitory Control =============
    
    def inhibit_action(self, action: str, strength: float = 1.0):
        """Suppress an inappropriate action"""
        self.suppressed_actions.add(action)
        self.inhibition_strength[action] = strength
        
        # Send GABA-ergic inhibition signal
        if self.connector:
            asyncio.create_task(self.connector.send_message(
                content={'inhibit': action, 'strength': strength},
                neurotransmitter=Neurotransmitter.GABA,
                priority=MessagePriority.HIGH
            ))
            
    def release_inhibition(self, action: str):
        """Release inhibition on an action"""
        if action in self.suppressed_actions:
            self.suppressed_actions.remove(action)
            del self.inhibition_strength[action]
            
    def should_inhibit(self, action: str) -> bool:
        """Check if action should be inhibited"""
        if action in self.suppressed_actions:
            # Check if inhibition has decayed
            strength = self.inhibition_strength[action]
            if strength > 0.1:
                return True
            else:
                self.release_inhibition(action)
        return False
        
    # ============= Conflict Resolution =============
    
    def _handle_conflict_deliberation(self, 
                                     topic: str,
                                     conflict: Dict) -> Dict[str, Any]:
        """Handle deliberation when conflict is detected"""
        
        opinion = (
            f"Conflict detected regarding '{topic}'. "
            f"Multiple valid perspectives with conflict level: "
            f"{conflict['conflict_level']:.2f}. "
        )
        
        # Attempt resolution strategies
        if conflict['conflict_level'] > 0.8:
            opinion += "Recommend gathering more information before deciding."
        elif conflict['conflict_level'] > 0.6:
            opinion += "Suggest sequential testing of top options."
        else:
            opinion += "Proceeding with highest-value option despite minor conflict."
            
        return {
            'module': self.name,
            'opinion': opinion,
            'confidence': 1.0 - conflict['conflict_level'] * 0.5,
            'reasoning': 'Conflict resolution protocol engaged',
            'conflict_details': conflict
        }
        
    async def resolve_conflict(self, 
                              options: List[Tuple[str, float]],
                              context: Dict[str, Any]) -> str:
        """
        Resolve conflict between competing options
        Uses multiple strategies based on context
        """
        
        conflict_level = self.acc.detect_conflict(options)
        
        if conflict_level < 0.3:
            # Low conflict - go with best option
            return max(options, key=lambda x: x[1])[0]
            
        # High conflict - need sophisticated resolution
        resolution_strategy = self._select_resolution_strategy(conflict_level, context)
        
        if resolution_strategy == 'deliberation':
            # Request more input from modules
            return await self._request_module_elaboration(options)
        elif resolution_strategy == 'sequential':
            # Try options in sequence
            return self._plan_sequential_testing(options)
        elif resolution_strategy == 'compromise':
            # Find middle ground
            return self._find_compromise(options)
        else:
            # Default to value-based
            return max(options, key=lambda x: x[1])[0]
            
    def _select_resolution_strategy(self, 
                                   conflict_level: float,
                                   context: Dict) -> str:
        """Select conflict resolution strategy"""
        
        if context.get('time_pressure', False):
            return 'value'  # Quick decision needed
        elif context.get('reversible', True):
            return 'sequential'  # Can try multiple options
        elif conflict_level > 0.8:
            return 'deliberation'  # Need more information
        else:
            return 'compromise'  # Find middle ground
            
    # ============= Emergency Protocols =============
    
    def _check_emergency_conditions(self, topic: str) -> bool:
        """Check if emergency conditions apply"""
        
        emergency_keywords = [
            'danger', 'emergency', 'critical', 'failure',
            'violation', 'threat', 'urgent', 'crisis'
        ]
        
        topic_lower = topic.lower()
        return any(keyword in topic_lower for keyword in emergency_keywords)
        
    def _get_emergency_response(self, topic: str) -> str:
        """Generate emergency response"""
        
        if 'danger' in topic.lower() or 'threat' in topic.lower():
            return self._danger_protocol()
        elif 'failure' in topic.lower():
            return self._system_failure_protocol()
        elif 'violation' in topic.lower():
            return self._ethical_violation_protocol()
        else:
            return "Emergency protocol activated. Prioritizing safety and stability."
            
    def _danger_protocol(self) -> str:
        """Response to danger"""
        self.emergency_mode = True
        
        # Inhibit non-essential functions
        self.inhibit_action('exploration', strength=1.0)
        self.inhibit_action('social_interaction', strength=0.8)
        
        # Boost sensory attention
        self.resource_allocation['attention'] = 1.5  # Hypervigilance
        
        return (
            "DANGER PROTOCOL: All non-essential functions suspended. "
            "Sensory systems heightened. Evaluating escape routes."
        )
        
    def _system_failure_protocol(self) -> str:
        """Response to system failure"""
        return (
            "SYSTEM FAILURE PROTOCOL: Initiating graceful degradation. "
            "Preserving core functions. Attempting self-repair."
        )
        
    def _ethical_violation_protocol(self) -> str:
        """Response to ethical violation"""
        self.inhibit_action('requested_action', strength=1.0)
        
        return (
            "ETHICAL VIOLATION PROTOCOL: Requested action violates ethical constraints. "
            "Action blocked. Suggesting ethical alternatives."
        )
        
    # ============= Main Processing Loop =============
    
    async def process_input(self, message: NeuralMessage):
        """Process incoming message"""
        
        # Update working memory
        self.dlpfc.update(WorkingMemoryItem(
            content=message.content,
            relevance=message.get_influence(),
            source=message.source
        ))
        
        # Rehearse working memory periodically
        self.dlpfc.rehearse()
        
        # Check for errors against predictions
        if 'actual' in message.content and 'predicted' in self.current_situation:
            error = self.acc.detect_error(
                self.current_situation['predicted'],
                message.content['actual']
            )
            
            if error > 0.5:
                # Significant prediction error - update models
                await self._handle_prediction_error(error, message)
                
        # Update situation model
        self.current_situation.update(message.content)
        
        # Check if response needed
        if message.neurotransmitter == Neurotransmitter.CORTISOL:
            # Emergency signal
            response = self._get_emergency_response(str(message.content))
            await self._broadcast_executive_decision(response, MessagePriority.EMERGENCY)
            
        elif message.neurotransmitter == Neurotransmitter.ACETYLCHOLINE:
            # Attention request
            if 'proposal_id' in message.content:
                await self._evaluate_proposal(message.content)
                
    async def _handle_prediction_error(self, error: float, message: NeuralMessage):
        """Handle significant prediction error"""
        
        # Update value estimates
        if message.source in self.vmpfc.value_estimates:
            # Reduce confidence in this source
            self.vmpfc.value_estimates[message.source] *= (1 - error * 0.1)
            
        # Send error signal for learning
        if self.connector:
            await self.connector.send_message(
                content={'error': error, 'source': message.source},
                neurotransmitter=Neurotransmitter.DOPAMINE,  # Prediction error = dopamine
                priority=MessagePriority.HIGH
            )
            
    async def _evaluate_proposal(self, proposal: Dict):
        """Evaluate a proposal for voting"""
        
        proposal_id = proposal.get('proposal_id')
        content = proposal.get('content', {})
        
        # Evaluate using vmPFC
        value = self.vmpfc.evaluate_option(
            str(content),
            content
        )
        
        # Check working memory for relevant context
        relevant = self.dlpfc.query(str(content))
        
        # Adjust confidence based on available information
        confidence = self._calculate_confidence(relevant)
        
        # Determine vote
        if value > 0.6:
            vote = 'approve'
        elif value < 0.4:
            vote = 'reject'
        else:
            vote = 'abstain'
            
        # Cast vote
        if self.connector:
            await self.connector.vote(proposal_id, vote, confidence)
            
    async def _broadcast_executive_decision(self, 
                                          decision: str,
                                          priority: MessagePriority = MessagePriority.HIGH):
        """Broadcast executive decision to all modules"""
        
        if self.connector:
            await self.connector.send_message(
                content={'executive_decision': decision, 'timestamp': time.time()},
                neurotransmitter=Neurotransmitter.NOREPINEPHRINE,  # Executive broadcast
                priority=priority
            )
            
        # Record decision
        self.decision_history.append({
            'decision': decision,
            'timestamp': time.time(),
            'confidence': self._calculate_confidence(list(self.dlpfc.working_memory))
        })
        
    # ============= Module Interface =============
    
    async def run(self):
        """Main executive loop"""
        
        while True:
            # Process any pending decisions
            if self.pending_decisions:
                for decision_id, decision_data in list(self.pending_decisions.items()):
                    if time.time() - decision_data['created_at'] > 5.0:
                        # Timeout - make decision
                        result = await self.make_decision(decision_data)
                        del self.pending_decisions[decision_id]
                        
            # Update resource allocation
            if hasattr(self, 'module_demands'):
                allocations = self.allocate_resources(self.module_demands)
                # Broadcast allocations
                
            # Check goal deadlines
            for goal_id in list(self.active_goals):
                goal = self.goal_hierarchy[goal_id]
                if goal.is_expired():
                    self.complete_goal(goal_id, success=False)
                    
            # Decay inhibitions
            for action in list(self.inhibition_strength.keys()):
                self.inhibition_strength[action] *= 0.95
                if self.inhibition_strength[action] < 0.1:
                    self.release_inhibition(action)
                    
            await asyncio.sleep(0.1)  # 100ms executive cycle
            
    async def make_decision(self, decision_data: Dict) -> Dict[str, Any]:
        """Make an executive decision"""
        
        options = decision_data.get('options', [])
        context = decision_data.get('context', {})
        
        # Evaluate each option
        evaluated = []
        for option in options:
            value = self.vmpfc.evaluate_option(option, context)
            evaluated.append((option, value))
            
        # Check for conflict
        conflict = self.acc.detect_conflict(evaluated)
        
        if conflict > self.acc.conflict_threshold:
            # Resolve conflict
            decision = await self.resolve_conflict(evaluated, context)
        else:
            # Choose best option
            decision = max(evaluated, key=lambda x: x[1])[0]
            
        # Check inhibition
        if self.should_inhibit(decision):
            # Find alternative
            for option, value in evaluated:
                if not self.should_inhibit(option):
                    decision = option
                    break
                    
        return {
            'decision': decision,
            'confidence': self._calculate_confidence(list(self.dlpfc.working_memory)),
            'conflict_level': conflict,
            'reasoning': self._explain_executive_reasoning(str(decision), [])
        }

# ============= Integration Example =============

async def example_executive_integration():
    """Example of executive module in action"""
    
    # Create executive
    executive = ExecutiveEidolon()
    await executive.initialize()
    
    # Create a goal hierarchy
    main_goal = executive.create_goal(
        "Navigate safely to destination",
        priority=1.0,
        deadline=time.time() + 300  # 5 minutes
    )
    
    sub_goal1 = executive.create_goal(
        "Avoid obstacles",
        priority=0.9,
        parent_id=main_goal.id
    )
    
    sub_goal2 = executive.create_goal(
        "Minimize energy expenditure",
        priority=0.6,
        parent_id=main_goal.id
    )
    
    # Activate main goal
    executive.activate_goal(main_goal.id)
    
    # Simulate sensory input suggesting obstacle
    sensory_message = NeuralMessage(
        source="sensory",
        content={
            'obstacle_detected': True,
            'distance': 2.0,
            'confidence': 0.85
        },
        neurotransmitter=Neurotransmitter.GLUTAMATE,
        priority=MessagePriority.HIGH
    )
    
    await executive.process_input(sensory_message)
    
    # Get executive opinion
    opinion = await executive.deliberate("Should we continue forward?")
    print(f"Executive says: {opinion['opinion']}")
    print(f"Confidence: {opinion['confidence']:.2f}")
    print(f"Reasoning: {opinion['reasoning']}")
    
    # Simulate resource allocation request
    demands = {
        'sensory': {'attention': 0.5, 'processing': 0.3},
        'memory': {'memory': 0.4, 'processing': 0.2},
        'logical': {'processing': 0.5}
    }
    
    allocations = executive.allocate_resources(demands)
    print(f"\nResource allocations: {allocations}")
    
    # Run executive loop
    await executive.run()

if __name__ == "__main__":
    asyncio.run(example_executive_integration())
