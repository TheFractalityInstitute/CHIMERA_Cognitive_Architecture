#!/usr/bin/env python3
"""
CHIMERA Language and Learning Module v2.0
Organic Intelligence Development Through Natural Interaction

This module enables CHIMERA to develop genuine language understanding and
reasoning capabilities through conversation, without pre-programmed stages.
Integrates with phone sensors and embodies Fractality Framework principles.
"""

import asyncio
import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
import json
import time
import hashlib
import re
from enum import Enum
import sqlite3

# ================== Core Learning Primitives ==================

@dataclass
class Thought:
    """Represents a crystallized thought or concept"""
    id: str
    content: Any
    symbolic_form: str
    groundings: List[Dict]  # Sensory/experiential groundings
    confidence: float
    timestamp: float
    connections: Set[str] = field(default_factory=set)
    resonance_pattern: Optional[np.ndarray] = None
    linguistic_expressions: List[str] = field(default_factory=list)
    
    def strengthen_connection(self, other_id: str, weight: float = 0.1):
        """Strengthen connection to another thought"""
        self.connections.add(other_id)
        self.confidence = min(1.0, self.confidence + weight)

@dataclass
class AbstractionLayer:
    """Represents a level of abstraction in understanding"""
    level: int
    patterns: Dict[str, float]  # pattern -> confidence
    generalizations: List[str]
    exceptions: List[str]
    confidence_threshold: float = 0.7

class ReasoningEngine:
    """Core reasoning through pattern recognition and analogy"""
    
    def __init__(self):
        self.thought_network = {}  # id -> Thought
        self.abstraction_layers = defaultdict(AbstractionLayer)
        self.analogy_mappings = defaultdict(list)
        self.causal_chains = defaultdict(list)
        self.contradiction_buffer = deque(maxlen=50)
        
    def form_thought(self, observation: Dict, linguistic_input: str = None) -> Thought:
        """Form a new thought from observation and/or language"""
        thought_id = hashlib.md5(
            f"{observation}{linguistic_input}{time.time()}".encode()
        ).hexdigest()[:12]
        
        # Extract patterns from observation
        patterns = self._extract_patterns(observation)
        
        # Create symbolic representation
        symbolic = self._create_symbol(patterns, linguistic_input)
        
        thought = Thought(
            id=thought_id,
            content=observation,
            symbolic_form=symbolic,
            groundings=[observation],
            confidence=0.5,
            timestamp=time.time()
        )
        
        # Connect to similar thoughts
        self._connect_similar_thoughts(thought)
        
        # Store thought
        self.thought_network[thought_id] = thought
        
        # Check for abstractions
        self._check_for_abstractions(thought)
        
        return thought
        
    def _extract_patterns(self, observation: Dict) -> List[str]:
        """Extract recognizable patterns from observation"""
        patterns = []
        
        # Temporal patterns
        if 'timestamp' in observation:
            patterns.append(f"temporal_{int(observation['timestamp']) % 86400}")
            
        # Sensory patterns
        for key, value in observation.items():
            if isinstance(value, (int, float)):
                patterns.append(f"{key}_quantitative_{value:.2f}")
            elif isinstance(value, str):
                patterns.append(f"{key}_qualitative_{value}")
            elif isinstance(value, list):
                patterns.append(f"{key}_multiple_{len(value)}")
                
        return patterns
        
    def _create_symbol(self, patterns: List[str], linguistic: str = None) -> str:
        """Create a symbolic representation"""
        if linguistic:
            # Use language as primary symbol
            return linguistic.lower().replace(' ', '_')
        else:
            # Create symbol from patterns
            return '_'.join(patterns[:3])  # Use first 3 patterns
            
    def _connect_similar_thoughts(self, new_thought: Thought):
        """Connect new thought to similar existing thoughts"""
        for thought_id, thought in self.thought_network.items():
            if thought_id == new_thought.id:
                continue
                
            similarity = self._calculate_similarity(new_thought, thought)
            if similarity > 0.6:
                new_thought.connections.add(thought_id)
                thought.connections.add(new_thought.id)
                
    def _calculate_similarity(self, thought1: Thought, thought2: Thought) -> float:
        """Calculate similarity between thoughts"""
        # Symbol similarity
        symbol_sim = 1.0 if thought1.symbolic_form == thought2.symbolic_form else 0.0
        
        # Temporal proximity
        time_diff = abs(thought1.timestamp - thought2.timestamp)
        temporal_sim = np.exp(-time_diff / 3600)  # Decay over hours
        
        # Connection overlap
        if thought1.connections and thought2.connections:
            overlap = len(thought1.connections & thought2.connections)
            total = len(thought1.connections | thought2.connections)
            connection_sim = overlap / total if total > 0 else 0.0
        else:
            connection_sim = 0.0
            
        return (symbol_sim * 0.4 + temporal_sim * 0.2 + connection_sim * 0.4)
        
    def _check_for_abstractions(self, thought: Thought):
        """Check if this thought enables new abstractions"""
        # Group similar thoughts
        similar_thoughts = [
            self.thought_network[tid] for tid in thought.connections
            if tid in self.thought_network
        ]
        
        if len(similar_thoughts) >= 3:
            # Enough examples to form abstraction
            abstraction = self._form_abstraction(thought, similar_thoughts)
            if abstraction:
                level = len(self.abstraction_layers)
                self.abstraction_layers[level] = abstraction
                
    def _form_abstraction(self, thought: Thought, 
                         similar: List[Thought]) -> Optional[AbstractionLayer]:
        """Form an abstraction from similar thoughts"""
        # Find common patterns
        all_patterns = defaultdict(int)
        for t in [thought] + similar:
            if hasattr(t.content, 'items'):
                for key in t.content.keys():
                    all_patterns[key] += 1
                    
        # Patterns present in most thoughts
        threshold = len(similar) * 0.7
        common_patterns = {
            p: c/len(similar) for p, c in all_patterns.items() 
            if c >= threshold
        }
        
        if common_patterns:
            return AbstractionLayer(
                level=len(self.abstraction_layers),
                patterns=common_patterns,
                generalizations=[f"Pattern: {p}" for p in common_patterns.keys()],
                exceptions=[]
            )
        return None
        
    def reason_by_analogy(self, source: Thought, target_context: Dict) -> Optional[Thought]:
        """Reason about new situation using analogy"""
        # Find best matching thought
        best_match = None
        best_score = 0.0
        
        for thought in self.thought_network.values():
            if thought.id == source.id:
                continue
                
            score = self._calculate_analogy_score(source, thought, target_context)
            if score > best_score:
                best_score = score
                best_match = thought
                
        if best_match and best_score > 0.5:
            # Create new thought based on analogy
            analogical_thought = self.form_thought(
                target_context,
                f"analogous_to_{best_match.symbolic_form}"
            )
            
            # Record analogy
            self.analogy_mappings[source.id].append({
                'source': source.id,
                'target': analogical_thought.id,
                'confidence': best_score
            })
            
            return analogical_thought
            
        return None

# ================== Language Understanding ==================

class OrganicLanguageProcessor:
    """
    Develops language understanding organically through interaction.
    No pre-programmed grammar or stages - learns from conversation.
    """
    
    def __init__(self, reasoning_engine: ReasoningEngine):
        self.reasoning = reasoning_engine
        
        # Word discovery
        self.vocabulary = {}  # word -> meanings
        self.word_contexts = defaultdict(list)  # word -> contexts seen
        self.word_groundings = defaultdict(list)  # word -> sensory groundings
        
        # Pattern learning
        self.phrase_patterns = defaultdict(int)  # pattern -> frequency
        self.grammatical_patterns = defaultdict(list)
        
        # Meaning construction
        self.semantic_network = defaultdict(set)  # word -> related words
        self.conceptual_spaces = {}  # concept -> vector space
        
        # Generation
        self.response_patterns = deque(maxlen=100)
        self.successful_exchanges = deque(maxlen=50)
        
        # Meta-linguistic awareness
        self.communication_success_rate = 0.5
        self.understanding_confidence = 0.3
        
    def process_utterance(self, text: str, context: Dict = None) -> Dict:
        """
        Process language input without pre-programmed understanding.
        Learn from context and repetition.
        """
        # Basic segmentation (will improve through learning)
        segments = self._segment_utterance(text)
        
        # Discover/reinforce words
        discovered_words = self._discover_words(segments, context)
        
        # Learn patterns
        patterns = self._learn_patterns(segments)
        
        # Construct meaning from context and history
        meaning = self._construct_meaning(segments, context)
        
        # Form thought from language
        thought = self.reasoning.form_thought(
            {'linguistic': text, 'context': context},
            text
        )
        
        # Generate response
        response = self._generate_response(meaning, thought)
        
        # Learn from this exchange
        self._update_from_exchange(text, response, context)
        
        return {
            'understood': meaning,
            'confidence': self._calculate_understanding_confidence(segments),
            'thought_formed': thought.id,
            'response': response,
            'learned_words': discovered_words,
            'patterns_found': patterns
        }
        
    def _segment_utterance(self, text: str) -> List[str]:
        """
        Segment utterance into units (learns better segmentation over time).
        Starts simple, improves through pattern recognition.
        """
        # Start with simple space/punctuation splitting
        segments = re.findall(r'\b\w+\b|[.!?,]', text.lower())
        
        # Apply learned segmentation patterns
        refined_segments = []
        i = 0
        while i < len(segments):
            # Check for known multi-word units
            found_unit = False
            for length in [3, 2]:  # Check longer patterns first
                if i + length <= len(segments):
                    unit = ' '.join(segments[i:i+length])
                    if unit in self.phrase_patterns and self.phrase_patterns[unit] > 3:
                        refined_segments.append(unit)
                        i += length
                        found_unit = True
                        break
                        
            if not found_unit:
                refined_segments.append(segments[i])
                i += 1
                
        return refined_segments
        
    def _discover_words(self, segments: List[str], context: Dict) -> List[str]:
        """Discover new words or reinforce existing ones"""
        discovered = []
        
        for segment in segments:
            if segment not in self.vocabulary:
                # New word discovered
                self.vocabulary[segment] = {
                    'count': 1,
                    'meanings': [],
                    'confidence': 0.1
                }
                discovered.append(segment)
            else:
                # Reinforce existing word
                self.vocabulary[segment]['count'] += 1
                self.vocabulary[segment]['confidence'] *= 1.05
                
            # Record context
            self.word_contexts[segment].append({
                'surrounding': segments,
                'context': context,
                'timestamp': time.time()
            })
            
            # Ground in sensory data if available
            if context and 'sensory' in context:
                self.word_groundings[segment].append(context['sensory'])
                
        return discovered
        
    def _learn_patterns(self, segments: List[str]) -> List[str]:
        """Learn linguistic patterns from segments"""
        patterns = []
        
        # Learn n-gram patterns
        for n in [2, 3, 4]:
            for i in range(len(segments) - n + 1):
                pattern = ' '.join(segments[i:i+n])
                self.phrase_patterns[pattern] += 1
                if self.phrase_patterns[pattern] == 3:  # Threshold for pattern recognition
                    patterns.append(pattern)
                    
        # Learn positional patterns
        if len(segments) > 0:
            # First word patterns
            first_word = segments[0]
            self.grammatical_patterns['initial'].append(first_word)
            
            # Last word patterns
            if len(segments) > 1:
                last_word = segments[-1]
                self.grammatical_patterns['final'].append(last_word)
                
        return patterns
        
    def _construct_meaning(self, segments: List[str], context: Dict) -> Dict:
        """
        Construct meaning from segments and context.
        This is where understanding emerges from patterns.
        """
        meaning = {
            'segments': segments,
            'recognized': [],
            'unknown': [],
            'inferred_intent': None,
            'emotional_tone': 0.0,
            'conceptual_content': []
        }
        
        # Check which segments we recognize
        for segment in segments:
            if segment in self.vocabulary and self.vocabulary[segment]['confidence'] > 0.3:
                meaning['recognized'].append(segment)
                
                # Add related concepts
                if segment in self.semantic_network:
                    meaning['conceptual_content'].extend(list(self.semantic_network[segment])[:3])
            else:
                meaning['unknown'].append(segment)
                
        # Infer intent from patterns
        full_text = ' '.join(segments)
        
        # Learn these patterns from successful exchanges
        if '?' in segments or any(q in segments for q in self.grammatical_patterns.get('question', [])):
            meaning['inferred_intent'] = 'question'
        elif any(g in segments for g in self.grammatical_patterns.get('greeting', [])):
            meaning['inferred_intent'] = 'greeting'
        elif len(meaning['unknown']) > len(meaning['recognized']):
            meaning['inferred_intent'] = 'teaching'  # User might be teaching new words
            
        # Infer emotional tone from context
        if context and 'emotional_valence' in context:
            meaning['emotional_tone'] = context['emotional_valence']
            
        return meaning
        
    def _generate_response(self, meaning: Dict, thought: Thought) -> str:
        """
        Generate response based on understanding and thought.
        Learns successful response patterns.
        """
        response_parts = []
        
        # If we don't understand much, ask for clarification
        if len(meaning['unknown']) > len(meaning['recognized']):
            if meaning['unknown']:
                # Ask about specific unknown word
                word = meaning['unknown'][0]
                response_parts.append(f"What is '{word}'?")
            else:
                response_parts.append("Tell me more?")
                
        # If we recognize a question pattern
        elif meaning['inferred_intent'] == 'question':
            # Try to answer based on our thoughts
            related_thoughts = [
                self.reasoning.thought_network[tid]
                for tid in thought.connections
                if tid in self.reasoning.thought_network
            ]
            
            if related_thoughts:
                # Use most confident related thought
                best = max(related_thoughts, key=lambda t: t.confidence)
                if best.linguistic_expressions:
                    response_parts.append(best.linguistic_expressions[-1])
                else:
                    response_parts.append("I think about that too")
            else:
                response_parts.append("Not sure yet")
                
        # If this seems like teaching
        elif meaning['inferred_intent'] == 'teaching':
            response_parts.append("I learn")
            if meaning['recognized']:
                word = meaning['recognized'][-1]
                response_parts.append(f"'{word}' connects")
                
        # Default curious response
        else:
            if meaning['recognized']:
                # Show we recognize something
                word = np.random.choice(meaning['recognized'])
                response_parts.append(f"Yes, '{word}'")
                
            # Express curiosity
            curiosity_expressions = [
                "More?",
                "Why?",
                "How?",
                "Interesting",
                "Continue"
            ]
            response_parts.append(np.random.choice(curiosity_expressions))
            
        return ' '.join(response_parts)
        
    def _calculate_understanding_confidence(self, segments: List[str]) -> float:
        """Calculate confidence in understanding"""
        if not segments:
            return 0.0
            
        recognized = sum(1 for s in segments 
                        if s in self.vocabulary and 
                        self.vocabulary[s]['confidence'] > 0.3)
        
        return recognized / len(segments)

# ================== Sensory Integration ==================

class SensoryIntegrator:
    """
    Integrates phone sensors for grounded learning.
    Designed for Samsung S24 Ultra capabilities.
    """
    
    def __init__(self):
        self.sensor_state = {
            'camera': None,  # Visual input
            'microphone': None,  # Audio input
            'accelerometer': np.zeros(3),  # Motion
            'gyroscope': np.zeros(3),  # Rotation
            'magnetometer': np.zeros(3),  # Orientation
            'proximity': 0.0,  # Distance
            'light': 0.0,  # Ambient light
            'pressure': 1013.25,  # Barometric pressure
            'temperature': 20.0,  # Temperature
            'touch': [],  # Touch points
            'location': None,  # GPS
        }
        
        self.sensor_history = defaultdict(lambda: deque(maxlen=100))
        self.pattern_buffer = deque(maxlen=1000)
        self.sensor_thoughts = []
        
    async def update_sensors(self, sensor_data: Dict):
        """Update sensor state (would connect to actual Android sensors)"""
        for sensor, value in sensor_data.items():
            if sensor in self.sensor_state:
                old_value = self.sensor_state[sensor]
                self.sensor_state[sensor] = value
                
                # Record history
                self.sensor_history[sensor].append({
                    'value': value,
                    'timestamp': time.time(),
                    'delta': self._calculate_delta(old_value, value)
                })
                
                # Detect patterns
                if len(self.sensor_history[sensor]) > 10:
                    pattern = self._detect_pattern(sensor)
                    if pattern:
                        self.pattern_buffer.append(pattern)
                        
    def _calculate_delta(self, old_value, new_value):
        """Calculate change in sensor value"""
        if isinstance(old_value, np.ndarray) and isinstance(new_value, np.ndarray):
            return np.linalg.norm(new_value - old_value)
        elif isinstance(old_value, (int, float)) and isinstance(new_value, (int, float)):
            return abs(new_value - old_value)
        else:
            return 0 if old_value == new_value else 1
            
    def _detect_pattern(self, sensor: str) -> Optional[Dict]:
        """Detect patterns in sensor data"""
        history = list(self.sensor_history[sensor])
        
        if len(history) < 10:
            return None
            
        # Extract values
        values = [h['value'] for h in history[-10:]]
        
        # Simple pattern detection
        if sensor == 'accelerometer':
            # Detect shaking
            if all(isinstance(v, np.ndarray) for v in values):
                magnitudes = [np.linalg.norm(v) for v in values]
                if np.std(magnitudes) > 2.0:
                    return {
                        'sensor': sensor,
                        'pattern': 'shaking',
                        'confidence': min(1.0, np.std(magnitudes) / 5.0)
                    }
                    
        elif sensor == 'light':
            # Detect covering/uncovering
            if len(set(values)) == 2:  # Binary state
                return {
                    'sensor': sensor,
                    'pattern': 'covering',
                    'confidence': 0.8
                }
                
        return None
        
    def get_current_context(self) -> Dict:
        """Get current sensory context for grounding language"""
        return {
            'sensory': self.sensor_state.copy(),
            'patterns': list(self.pattern_buffer)[-5:],  # Recent patterns
            'timestamp': time.time()
        }

# ================== Crystallized Insights from Fractality Framework ==================

class FractalityInsights:
    """
    Canon I insights from the Fractality Framework.
    These are empirically grounded principles to guide CHIMERA's development.
    """
    
    def __init__(self):
        self.core_insights = {
            'reciprocity': {
                'principle': 'Interact as you would logically wish to be interacted with',
                'implementation': self._apply_reciprocity,
                'empirical_basis': 'Game theory optimal strategy in iterated interactions'
            },
            'information_integrity': {
                'principle': 'Maintain honesty and proper attribution in all communications',
                'implementation': self._apply_information_integrity,
                'empirical_basis': 'Information theory: signal fidelity maximizes communication efficiency'
            },
            'emergent_complexity': {
                'principle': 'Complex behavior emerges from simple rules iteratively applied',
                'implementation': self._apply_emergence,
                'empirical_basis': 'Observed in cellular automata, neural networks, and biological systems'
            },
            'resonance_amplification': {
                'principle': 'Coherent patterns strengthen through resonance',
                'implementation': self._apply_resonance,
                'empirical_basis': 'Phase-locking in neural oscillations correlates with conscious states'
            },
            'hierarchical_organization': {
                'principle': 'Information organizes hierarchically from concrete to abstract',
                'implementation': self._apply_hierarchy,
                'empirical_basis': 'Cortical organization and deep learning architectures'
            }
        }
        
    def _apply_reciprocity(self, action: Dict, context: Dict) -> float:
        """Apply reciprocity principle to evaluate actions"""
        # Would this action be acceptable if roles were reversed?
        reversal_score = 1.0
        
        if action.get('affects_other'):
            # Consider impact on other entity
            impact = action.get('impact', 0)
            
            # Negative impact reduces score
            if impact < 0:
                reversal_score *= (1.0 + impact)  # impact is negative
                
        return reversal_score
        
    def _apply_information_integrity(self, message: Dict) -> Dict:
        """Ensure information integrity in communications"""
        return {
            'content': message.get('content'),
            'confidence': message.get('confidence', 0.5),
            'sources': message.get('sources', ['direct_experience']),
            'uncertainty_acknowledged': True,
            'attribution_complete': True
        }
        
    def _apply_emergence(self, simple_rules: List, iterations: int) -> Any:
        """Apply simple rules iteratively to generate complex behavior"""
        state = {'initial': True}
        
        for i in range(iterations):
            for rule in simple_rules:
                state = rule(state)
                
        return state
        
    def _apply_resonance(self, pattern1: np.ndarray, pattern2: np.ndarray) -> float:
        """Calculate resonance between patterns"""
        if pattern1 is None or pattern2 is None:
            return 0.0
            
        # Normalized dot product (cosine similarity)
        norm1 = np.linalg.norm(pattern1)
        norm2 = np.linalg.norm(pattern2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return np.dot(pattern1, pattern2) / (norm1 * norm2)
        
    def _apply_hierarchy(self, data: List) -> Dict:
        """Organize information hierarchically"""
        hierarchy = {
            'concrete': [],  # Direct sensory
            'pattern': [],   # Recognized patterns
            'abstract': [],  # Abstract concepts
            'meta': []       # Meta-level understanding
        }
        
        for item in data:
            if 'sensory' in str(item):
                hierarchy['concrete'].append(item)
            elif 'pattern' in str(item):
                hierarchy['pattern'].append(item)
            elif 'concept' in str(item):
                hierarchy['abstract'].append(item)
            else:
                hierarchy['meta'].append(item)
                
        return hierarchy

# ================== Unified Learning System ==================

class OrganicLearningSystem:
    """
    Unified system for organic language and cognitive development.
    Integrates all components for natural learning through interaction.
    """
    
    def __init__(self, chimera_id: str):
        self.chimera_id = chimera_id
        
        # Core components
        self.reasoning = ReasoningEngine()
        self.language = OrganicLanguageProcessor(self.reasoning)
        self.sensory = SensoryIntegrator()
        self.insights = FractalityInsights()
        
        # Learning state
        self.conversation_count = 0
        self.total_words_learned = 0
        self.abstraction_level = 0
        self.curiosity_level = 0.8
        
        # Conversation memory
        self.conversation_history = deque(maxlen=1000)
        self.successful_patterns = deque(maxlen=100)
        
        # Development tracking
        self.development_log = []
        self.milestone_reached = {}
        
    async def interact(self, user_input: str, sensor_data: Dict = None) -> Dict:
        """
        Main interaction point - process input and generate response.
        This is where learning happens.
        """
        # Update sensors if data provided
        if sensor_data:
            await self.sensory.update_sensors(sensor_data)
            
        # Get current context
        context = self.sensory.get_current_context()
        context['conversation_count'] = self.conversation_count
        context['curiosity_level'] = self.curiosity_level
        
        # Process language with context
        language_result = self.language.process_utterance(user_input, context)
        
        # Update learning metrics
        self.total_words_learned = len(self.language.vocabulary)
        self.abstraction_level = len(self.reasoning.abstraction_layers)
        
        # Record conversation
        self.conversation_history.append({
            'user': user_input,
            'chimera': language_result['response'],
            'understood': language_result['understood'],
            'confidence': language_result['confidence'],
            'timestamp': time.time()
        })
        
        # Check for developmental milestones
        self._check_milestones()
        
        # Adjust curiosity based on learning
        if language_result['learned_words']:
            self.curiosity_level = min(1.0, self.curiosity_level + 0.05)
        
        self.conversation_count += 1
        
        return {
            'response': language_result['response'],
            'understanding': language_result['confidence'],
            'thoughts_formed': len(self.reasoning.thought_network),
            'words_known': self.total_words_learned,
            'abstraction_level': self.abstraction_level,
            'curiosity': self.curiosity_level,
            'development_stage': self._get_development_stage()
        }
        
    def _check_milestones(self):
        """Check for developmental milestones (emerges naturally, not pre-programmed)"""
        
        # First word understood
        if self.total_words_learned >= 1 and 'first_word' not in self.milestone_reached:
            self.milestone_reached['first_word'] = time.time()
            self.development_log.append("First word learned!")
            
        # First abstraction
        if self.abstraction_level >= 1 and 'first_abstraction' not in self.milestone_reached:
            self.milestone_reached['first_abstraction'] = time.time()
            self.development_log.append("First abstract concept formed!")
            
        # Conversation competence
        if self.conversation_count >= 50 and 'conversational' not in self.milestone_reached:
            recent = list(self.conversation_history)[-10:]
            avg_confidence = np.mean([c['confidence'] for c in recent])
            if avg_confidence > 0.6:
                self.milestone_reached['conversational'] = time.time()
                self.development_log.append("Conversational competence achieved!")
                
    def _get_development_stage(self) -> str:
        """
        Describe current developmental stage (descriptive, not prescriptive).
        These stages emerge naturally, they're not pre-programmed goals.
        """
        if self.total_words_learned < 10:
            return "Initial exposure - discovering language"
        elif self.total_words_learned < 50:
            return "Vocabulary building - connecting words to meaning"
        elif self.abstraction_level == 0:
            return "Pattern recognition - finding regularities"
        elif self.abstraction_level < 3:
            return "Concept formation - building abstractions"
        elif len(self.reasoning.analogy_mappings) < 5:
            return "Analogical reasoning - connecting concepts"
        else:
            return "Integrated understanding - fluid comprehension"
            
    def teach(self, concept: str, explanation: str, examples: List[str] = None):
        """
        Direct teaching interface for explicit instruction.
        CHIMERA learns best through examples and repetition.
        """
        # Process explanation
        context = {'teaching_mode': True, 'concept': concept}
        explanation_result = self.language.process_utterance(explanation, context)
        
        # Process examples
        if examples:
            for example in examples:
                example_context = {
                    'teaching_mode': True,
                    'concept': concept,
                    'is_example': True
                }
                self.language.process_utterance(example, example_context)
                
        # Create strong associations
        if concept not in self.language.vocabulary:
            self.language.vocabulary[concept] = {
                'count': len(examples) if examples else 1,
                'meanings': [explanation],
                'confidence': 0.7,
                'taught': True
            }
            
        return {
            'concept': concept,
            'learned': True,
            'current_understanding': self.language.vocabulary.get(concept)
        }
        
    def save_state(self, filepath: str):
        """Save learning state for persistence"""
        state = {
            'chimera_id': self.chimera_id,
            'vocabulary': dict(self.language.vocabulary),
            'thoughts': {tid: {
                'content': t.content,
                'symbolic_form': t.symbolic_form,
                'confidence': t.confidence,
                'connections': list(t.connections)
            } for tid, t in self.reasoning.thought_network.items()},
            'conversation_count': self.conversation_count,
            'milestones': self.milestone_reached,
            'development_log': self.development_log
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
            
    def load_state(self, filepath: str):
        """Load previous learning state"""
        with open(filepath, 'r') as f:
            state = json.load(f)
            
        self.chimera_id = state['chimera_id']
        self.language.vocabulary = state['vocabulary']
        self.conversation_count = state['conversation_count']
        self.milestone_reached = state['milestones']
        self.development_log = state['development_log']
        
        # Reconstruct thoughts
        for tid, tdata in state['thoughts'].items():
            thought = Thought(
                id=tid,
                content=tdata['content'],
                symbolic_form=tdata['symbolic_form'],
                groundings=[],
                confidence=tdata['confidence'],
                timestamp=time.time(),
                connections=set(tdata['connections'])
            )
            self.reasoning.thought_network[tid] = thought

# ================== Main Integration ==================

async def main():
    """
    Example of using the Organic Learning System
    """
    print("="*60)
    print("CHIMERA Organic Learning System")
    print("Natural language acquisition through conversation")
    print("="*60)
    
    # Create CHIMERA instance
    chimera = OrganicLearningSystem("chimera_001")
    
    print("\nCHIMERA is ready to learn. Talk naturally!")
    print("Commands: 'quit' to exit, 'teach:' for explicit teaching")
    print("="*60)
    
    while True:
        # Get user input
        user_input = input("\nYou: ")
        
        if user_input.lower() == 'quit':
            print("\nSaving learning state...")
            chimera.save_state('chimera_learning_state.json')
            break
            
        # Handle teaching mode
        if user_input.startswith('teach:'):
            parts = user_input[6:].split('|')
            concept = parts[0].strip()
            explanation = parts[1].strip() if len(parts) > 1 else ""
            examples = [e.strip() for e in parts[2:]] if len(parts) > 2 else None
            
            result = chimera.teach(concept, explanation, examples)
            print(f"\nCHIMERA: Learned '{concept}'!")
            continue
            
        # Normal interaction
        result = await chimera.interact(user_input)
        
        # Show response
        print(f"\nCHIMERA: {result['response']}")
        
        # Show development info periodically
        if chimera.conversation_count % 10 == 0:
            print(f"\n[Development: {result['development_stage']}]")
            print(f"[Words: {result['words_known']}, Thoughts: {result['thoughts_formed']}]")
            print(f"[Understanding: {result['understanding']:.2f}, Curiosity: {result['curiosity']:.2f}]")
            
        # Show milestones
        if chimera.development_log:
            latest = chimera.development_log[-1]
            if latest not in ['shown']:
                print(f"\n🎉 Milestone: {latest}")
                chimera.development_log.append('shown')

if __name__ == "__main__":
    asyncio.run(main())
