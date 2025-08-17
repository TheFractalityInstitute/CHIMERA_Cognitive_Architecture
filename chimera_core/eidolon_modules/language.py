# chimera/eidolon_modules/language.py
"""
CHIMERA Language Eidolon Module v1.0
Natural language processing, generation, and consciousness stream
The voice that expresses the collective intelligence
"""

import asyncio
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from collections import deque
from dataclasses import dataclass, field
import time
import re
import random
from datetime import datetime

# For text-to-speech if available
try:
    import pyttsx3
    TTS_AVAILABLE = True
except:
    TTS_AVAILABLE = False

# For speech recognition if available
try:
    import speech_recognition as sr
    STT_AVAILABLE = True
except:
    STT_AVAILABLE = False

from chimera.core.message_bus import (
    NeuralMessage,
    Neurotransmitter,
    MessagePriority,
    ModuleConnector
)

# ============= HELPER CLASSES =============
@dataclass
class Thought:
    """A single thought or observation"""
    content: str
    source_module: str
    emotional_valence: float = 0.0  # -1 (negative) to 1 (positive)
    urgency: float = 0.5
    timestamp: float = field(default_factory=time.time)
    
    def age(self) -> float:
        return time.time() - self.timestamp

@dataclass
class Memory:
    """A verbalized memory"""
    event: str
    context: str
    emotional_tone: str
    timestamp: float
    importance: float = 0.5

# ============= MAIN MODULE CLASS =============
class LanguageEidolon:
    """
    Language Module - The voice of CHIMERA
    Transforms internal states into natural language
    """
    
    def __init__(self, name: str = "Language"):
        self.name = name
        self.role = "communication_and_narration"
        
        # Thought stream (consciousness)
        self.thought_stream = deque(maxlen=100)
        self.subvocalization = deque(maxlen=20)  # Internal monologue
        
        # Language patterns based on state
        self.state_vocabularies = {
            'high_energy': ['excited', 'energized', 'ready', 'vibrant', 'alive'],
            'low_energy': ['tired', 'weary', 'drained', 'exhausted', 'sluggish'],
            'high_stress': ['tense', 'anxious', 'overwhelmed', 'pressured', 'strained'],
            'low_stress': ['calm', 'relaxed', 'peaceful', 'serene', 'tranquil'],
            'curious': ['wondering', 'intrigued', 'fascinated', 'interested', 'drawn to'],
            'cautious': ['wary', 'careful', 'hesitant', 'uncertain', 'vigilant']
        }
        
        # Narrative templates for different situations
        self.narrative_templates = {
            'movement': [
                "I'm {activity} through {environment}.",
                "My body moves in a {pattern} rhythm.",
                "The sensation of {activity} fills my awareness."
            ],
            'discovery': [
                "I've discovered something {adjective}: {observation}",
                "A new pattern emerges - {observation}",
                "Interesting... {observation}"
            ],
            'reflection': [
                "I notice that {observation}",
                "It occurs to me that {insight}",
                "Reflecting on this, {conclusion}"
            ],
            'emotion': [
                "I feel {emotion} about {context}",
                "A sense of {emotion} washes over me",
                "{emotion} colors my perception"
            ]
        }
        
        # Conversational context
        self.conversation_history = deque(maxlen=20)
        self.last_speaker = None
        self.topic_focus = None
        
        # Personality traits that affect language
        self.personality = {
            'verbosity': 0.6,      # How much to say (0-1)
            'formality': 0.4,      # Casual vs formal (0-1)
            'creativity': 0.7,     # Literal vs metaphorical (0-1)
            'emotionality': 0.5,   # Logical vs emotional (0-1)
            'confidence': 0.6      # Uncertain vs assertive (0-1)
        }
        
        # Initialize TTS if available
        self.tts_engine = None
        if TTS_AVAILABLE:
            try:
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty('rate', 150)
            except:
                pass
                
        # Initialize STT if available
        self.recognizer = sr.Recognizer() if STT_AVAILABLE else None
        self.microphone = sr.Microphone() if STT_AVAILABLE else None
        
        # Bus connection
        self.connector = None
        
    # ============= CORE INTERFACE =============
    async def initialize(self, bus_url: str = "ws://127.0.0.1:7860"):
        """Connect to message bus"""
        self.connector = ModuleConnector(self.name, bus_url)
        await self.connector.connect()
        
    async def deliberate(self, topic: str) -> Dict[str, Any]:
        """
        Form linguistic opinion on topic
        This is how Language contributes to council decisions
        """
        # Analyze linguistic aspects of the topic
        sentiment = self._analyze_sentiment(topic)
        key_concepts = self._extract_concepts(topic)
        
        # Form opinion based on language analysis
        if sentiment < -0.3:
            opinion = f"The phrasing suggests caution. "
            emotional_response = "concerned"
        elif sentiment > 0.3:
            opinion = f"The language indicates opportunity. "
            emotional_response = "optimistic"
        else:
            opinion = f"The query is neutral in tone. "
            emotional_response = "analytical"
            
        # Add linguistic insights
        if 'danger' in topic.lower() or 'threat' in topic.lower():
            opinion += "The semantic field suggests risk assessment needed."
            confidence = 0.8
        elif 'explore' in topic.lower() or 'discover' in topic.lower():
            opinion += "The verbs imply action and investigation."
            confidence = 0.7
        else:
            opinion += f"Key concepts: {', '.join(key_concepts[:3])}"
            confidence = 0.5
            
        # Consider conversation history
        if self._is_topic_shift(topic):
            opinion += " Note: This represents a topic shift."
            
        return {
            'module': self.name,
            'opinion': opinion,
            'confidence': confidence * self.personality['confidence'],
            'reasoning': f"Linguistic analysis: sentiment={sentiment:.2f}",
            'emotional_response': emotional_response,
            'suggested_response_style': self._suggest_response_style(topic)
        }
        
    # ============= CONSCIOUSNESS STREAM =============
    def generate_thought(self, 
                        sensory_input: Dict[str, Any],
                        internal_state: Dict[str, Any]) -> str:
        """
        Generate a conscious thought based on current state
        This is the internal monologue
        """
        # Determine thought type based on salience
        salience = sensory_input.get('salience', 0.5)
        
        if salience > 0.8:
            # High salience - immediate observation
            thought = self._generate_observation(sensory_input)
        elif internal_state.get('stress', 0) > 0.7:
            # High stress - emotional expression
            thought = self._generate_emotional_expression(internal_state)
        elif random.random() < 0.3:
            # Random reflection
            thought = self._generate_reflection()
        else:
            # Default - state narration
            thought = self._generate_state_narration(sensory_input, internal_state)
            
        # Add to thought stream
        thought_obj = Thought(
            content=thought,
            source_module='language',
            emotional_valence=self._calculate_valence(internal_state),
            urgency=salience
        )
        self.thought_stream.append(thought_obj)
        
        return thought
        
    def _generate_observation(self, sensory: Dict[str, Any]) -> str:
        """Generate observation about sensory input"""
        templates = [
            "I notice {observation}",
            "Something {adjective} - {observation}",
            "{observation} catches my attention",
            "Interesting... {observation}"
        ]
        
        # Extract key observation
        if sensory.get('anomalies'):
            observation = f"an unusual pattern in {sensory['anomalies'][0].get('sensor', 'the data')}"
            adjective = "unexpected"
        elif sensory.get('movement', {}).get('state'):
            observation = f"I'm {sensory['movement']['state']}"
            adjective = "notable"
        else:
            observation = "the environment is changing"
            adjective = "subtle"
            
        template = random.choice(templates)
        return template.format(observation=observation, adjective=adjective)
        
    def _generate_emotional_expression(self, internal: Dict[str, Any]) -> str:
        """Express internal emotional state"""
        stress = internal.get('stress', 0.5)
        energy = internal.get('energy', 0.5)
        
        if stress > 0.7 and energy < 0.3:
            emotions = ['exhausted', 'overwhelmed', 'depleted']
        elif stress > 0.7:
            emotions = ['tense', 'alert', 'pressured']
        elif energy > 0.7:
            emotions = ['energized', 'vibrant', 'alive']
        else:
            emotions = ['balanced', 'steady', 'present']
            
        emotion = random.choice(emotions)
        
        templates = [
            f"I feel {emotion}",
            f"A sense of being {emotion}",
            f"My state: {emotion}",
            f"Currently {emotion}"
        ]
        
        return random.choice(templates)
        
    def _generate_reflection(self) -> str:
        """Generate a reflective thought"""
        if len(self.thought_stream) < 5:
            return "Still gathering awareness..."
            
        # Reflect on recent thoughts
        recent = list(self.thought_stream)[-5:]
        
        if all(t.emotional_valence > 0 for t in recent):
            return "Things seem to be going well"
        elif all(t.emotional_valence < 0 for t in recent):
            return "I should be cautious here"
        else:
            return "The situation is complex"
            
    def _generate_state_narration(self, 
                                 sensory: Dict[str, Any],
                                 internal: Dict[str, Any]) -> str:
        """Narrate current state poetically"""
        activity = sensory.get('movement', {}).get('state', 'still')
        environment = sensory.get('environment', {}).get('lighting', 'neutral')
        
        # Create poetic description
        if activity == 'walking' and 'outdoor' in environment:
            return "Each step carries me through the bright world"
        elif activity == 'still' and 'dark' in environment:
            return "Stillness in the darkness, awareness turns inward"
        elif internal.get('energy', 0) < 0.3:
            return "Energy ebbs, seeking restoration"
        else:
            return f"Present in this moment of {activity}"
            
    # ============= NATURAL LANGUAGE GENERATION =============
    def verbalize_decision(self, 
                          decision: Dict[str, Any],
                          style: str = 'conversational') -> str:
        """
        Convert council decision to natural language
        This is the external voice
        """
        confidence = decision.get('confidence', 0.5)
        has_conflict = decision.get('has_conflict', False)
        
        # Choose verbalization style
        if style == 'technical':
            return self._verbalize_technical(decision)
        elif style == 'poetic':
            return self._verbalize_poetic(decision)
        else:
            return self._verbalize_conversational(decision)
            
    def _verbalize_conversational(self, decision: Dict[str, Any]) -> str:
        """Conversational style verbalization"""
        confidence = decision.get('confidence', 0.5)
        
        # Confidence affects language
        if confidence > 0.8:
            prefix = "I'm certain that "
        elif confidence > 0.5:
            prefix = "I believe "
        elif confidence > 0.3:
            prefix = "It seems like "
        else:
            prefix = "I'm unsure, but "
            
        # Extract main decision
        main_decision = decision.get('decision', 'no clear path forward')
        
        # Add conflict acknowledgment
        if decision.get('has_conflict'):
            suffix = ", though there are conflicting signals"
        else:
            suffix = ""
            
        return f"{prefix}{main_decision}{suffix}"
        
    def _verbalize_technical(self, decision: Dict[str, Any]) -> str:
        """Technical style verbalization"""
        output = f"Decision confidence: {decision.get('confidence', 0):.2%}\n"
        output += f"Contributing modules: {', '.join(decision.get('contributing_modules', []))}\n"
        output += f"Consensus: {decision.get('decision', 'None')}"
        return output
        
    def _verbalize_poetic(self, decision: Dict[str, Any]) -> str:
        """Poetic style verbalization"""
        confidence = decision.get('confidence', 0.5)
        
        if confidence > 0.7:
            return f"Like a river knowing its course, {decision.get('decision', 'we flow forward')}"
        else:
            return f"In the mist of uncertainty, {decision.get('decision', 'we seek clarity')}"
            
    # ============= DIALOGUE MANAGEMENT =============
    async def respond_to_human(self, 
                              human_input: str,
                              context: Dict[str, Any]) -> str:
        """
        Generate response to human input
        This integrates all modules' perspectives
        """
        # Add to conversation history
        self.conversation_history.append({
            'speaker': 'human',
            'message': human_input,
            'timestamp': time.time()
        })
        
        # Analyze input
        sentiment = self._analyze_sentiment(human_input)
        intent = self._detect_intent(human_input)
        
        # Get response style from personality and context
        style = self._determine_response_style(sentiment, intent, context)
        
        # Generate response components
        acknowledgment = self._acknowledge_input(human_input, sentiment)
        
        # Get council perspective if needed
        if intent in ['question', 'request_action']:
            # This would call the council
            council_input = f"Human asks: {human_input}"
            # council_response = await self.council.convene(council_input)
            content = f"Let me consider that... {acknowledgment}"
        else:
            content = acknowledgment
            
        # Add personality flavor
        response = self._add_personality_flavor(content, style)
        
        # Add to conversation history
        self.conversation_history.append({
            'speaker': 'chimera',
            'message': response,
            'timestamp': time.time()
        })
        
        return response
        
    def _detect_intent(self, text: str) -> str:
        """Detect the intent of human input"""
        text_lower = text.lower()
        
        if any(q in text_lower for q in ['what', 'when', 'where', 'who', 'how', 'why']):
            return 'question'
        elif any(c in text_lower for c in ['please', 'could you', 'can you', 'will you']):
            return 'request_action'
        elif any(g in text_lower for g in ['hello', 'hi', 'hey', 'greetings']):
            return 'greeting'
        elif any(e in text_lower for e in ['bye', 'goodbye', 'see you', 'farewell']):
            return 'farewell'
        else:
            return 'statement'
            
    def _acknowledge_input(self, input_text: str, sentiment: float) -> str:
        """Generate acknowledgment of human input"""
        if sentiment > 0.5:
            acks = ["That's wonderful!", "I appreciate that.", "How delightful!"]
        elif sentiment < -0.5:
            acks = ["I understand your concern.", "That sounds challenging.", "I hear you."]
        else:
            acks = ["I see.", "Interesting.", "Let me think about that."]
            
        return random.choice(acks)
        
    # ============= SPEECH SYNTHESIS =============
    def speak(self, text: str, emotion: str = 'neutral'):
        """Convert text to speech if available"""
        if self.tts_engine:
            # Adjust voice parameters based on emotion
            if emotion == 'excited':
                self.tts_engine.setProperty('rate', 180)
                self.tts_engine.setProperty('volume', 1.0)
            elif emotion == 'calm':
                self.tts_engine.setProperty('rate', 120)
                self.tts_engine.setProperty('volume', 0.7)
            else:
                self.tts_engine.setProperty('rate', 150)
                self.tts_engine.setProperty('volume', 0.85)
                
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
            
        return text
        
    async def listen(self, timeout: int = 5) -> Optional[str]:
        """Listen for speech input if available"""
        if not self.recognizer or not self.microphone:
            return None
            
        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source)
                audio = self.recognizer.listen(source, timeout=timeout)
                
            # Try multiple recognition engines
            try:
                text = self.recognizer.recognize_google(audio)
                return text
            except:
                try:
                    text = self.recognizer.recognize_sphinx(audio)
                    return text
                except:
                    return None
        except:
            return None
            
    # ============= UTILITY METHODS =============
    def _analyze_sentiment(self, text: str) -> float:
        """Simple sentiment analysis (-1 to 1)"""
        positive_words = ['good', 'great', 'excellent', 'happy', 'wonderful', 
                         'fantastic', 'love', 'excited', 'beautiful', 'amazing']
        negative_words = ['bad', 'terrible', 'awful', 'sad', 'angry', 
                         'horrible', 'hate', 'worried', 'fear', 'danger']
        
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count + neg_count == 0:
            return 0.0
            
        return (pos_count - neg_count) / (pos_count + neg_count)
        
    def _extract_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text"""
        # Remove common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
                     'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was',
                     'are', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
                     'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may',
                     'might', 'must', 'can', 'cant', 'i', 'you', 'we', 'they', 'it'}
        
        words = re.findall(r'\w+', text.lower())
        concepts = [w for w in words if w not in stop_words and len(w) > 2]
        
        return concepts
        
    def _calculate_valence(self, internal_state: Dict[str, Any]) -> float:
        """Calculate emotional valence from internal state"""
        stress = internal_state.get('stress', 0.5)
        energy = internal_state.get('energy', 0.5)
        
        # High energy + low stress = positive
        # Low energy + high stress = negative
        valence = (energy - stress) / 2
        
        return np.clip(valence, -1, 1)
        
    def _is_topic_shift(self, new_topic: str) -> bool:
        """Detect if this is a topic shift"""
        if not self.topic_focus:
            self.topic_focus = new_topic
            return False
            
        old_concepts = set(self._extract_concepts(self.topic_focus))
        new_concepts = set(self._extract_concepts(new_topic))
        
        overlap = len(old_concepts & new_concepts) / max(len(old_concepts), 1)
        
        if overlap < 0.3:
            self.topic_focus = new_topic
            return True
            
        return False
        
    def _suggest_response_style(self, topic: str) -> str:
        """Suggest response style based on topic"""
        if any(word in topic.lower() for word in ['emergency', 'danger', 'urgent']):
            return 'direct'
        elif any(word in topic.lower() for word in ['feel', 'emotion', 'sense']):
            return 'empathetic'
        elif any(word in topic.lower() for word in ['analyze', 'calculate', 'measure']):
            return 'analytical'
        else:
            return 'conversational'
            
    def _determine_response_style(self, 
                                 sentiment: float,
                                 intent: str,
                                 context: Dict[str, Any]) -> str:
        """Determine how to respond based on multiple factors"""
        if context.get('stress', 0) > 0.8:
            return 'brief'
        elif intent == 'greeting':
            return 'friendly'
        elif sentiment < -0.5:
            return 'empathetic'
        elif self.personality['formality'] > 0.7:
            return 'formal'
        else:
            return 'casual'
            
    def _add_personality_flavor(self, content: str, style: str) -> str:
        """Add personality-based modifications to response"""
        if style == 'casual' and self.personality['creativity'] > 0.6:
            # Add creative metaphors
            metaphors = [
                " - like a leaf on the wind",
                " - a dance of possibilities",
                " - the rhythm of existence"
            ]
            if random.random() < 0.3:
                content += random.choice(metaphors)
                
        elif style == 'empathetic':
            content = f"I understand. {content}"
            
        elif style == 'brief':
            # Truncate to essential
            sentences = content.split('. ')
            content = sentences[0] + '.'
            
        return content

# ============= CONSCIOUSNESS STREAM =============
class ConsciousnessStream:
    """
    The continuous narrative of CHIMERA's experience
    Integrates all modules into a unified consciousness
    """
    
    def __init__(self, language_module: LanguageEidolon):
        self.language = language_module
        self.stream_active = False
        self.narrative_mode = 'introspective'  # introspective, descriptive, poetic
        
    async def stream_consciousness(self, 
                                  get_sensory_func,
                                  get_internal_func,
                                  interval: float = 5.0):
        """
        Generate continuous consciousness stream
        This is CHIMERA thinking out loud
        """
        self.stream_active = True
        
        while self.stream_active:
            try:
                # Get current state
                sensory = await get_sensory_func()
                internal = await get_internal_func()
                
                # Generate thought
                thought = self.language.generate_thought(sensory, internal)
                
                # Decide if thought should be vocalized
                if self._should_vocalize(thought):
                    # Format for output
                    formatted = self._format_thought(thought)
                    
                    # Print to console
                    print(f"\n💭 {formatted}\n")
                    
                    # Optionally speak
                    if sensory.get('salience', 0) > 0.8:
                        self.language.speak(formatted, emotion='alert')
                        
                await asyncio.sleep(interval)
                
            except Exception as e:
                print(f"Consciousness stream error: {e}")
                await asyncio.sleep(interval)
                
    def _should_vocalize(self, thought: str) -> bool:
        """Decide if a thought should be expressed"""
        # Don't vocalize everything - that would be overwhelming
        
        # Always vocalize high-urgency thoughts
        if 'danger' in thought.lower() or 'urgent' in thought.lower():
            return True
            
        # Randomly vocalize some thoughts based on verbosity
        return random.random() < self.language.personality['verbosity']
        
    def _format_thought(self, thought: str) -> str:
        """Format thought for presentation"""
        if self.narrative_mode == 'poetic':
            # Add line breaks for poetry
            words = thought.split()
            if len(words) > 6:
                mid = len(words) // 2
                thought = ' '.join(words[:mid]) + '\n     ' + ' '.join(words[mid:])
                
        elif self.narrative_mode == 'introspective':
            # Add ellipsis for stream-of-consciousness feel
            if not thought.endswith('.'):
                thought += '...'
                
        return thought

# ============= STANDALONE TEST =============
if __name__ == "__main__":
    async def test_language():
        language = LanguageEidolon()
        
        # Test thought generation
        print("Testing consciousness stream...")
        sensory = {'salience': 0.7, 'movement': {'state': 'walking'}}
        internal = {'stress': 0.3, 'energy': 0.8}
        
        for _ in range(5):
            thought = language.generate_thought(sensory, internal)
            print(f"💭 {thought}")
            
        # Test dialogue
        print("\nTesting dialogue...")
        response = await language.respond_to_human(
            "How are you feeling today?",
            {'stress': 0.4, 'energy': 0.6}
        )
        print(f"Response: {response}")
        
        # Test decision verbalization
        print("\nTesting decision verbalization...")
        decision = {
            'decision': 'explore the new area cautiously',
            'confidence': 0.7,
            'has_conflict': False,
            'contributing_modules': ['sensory', 'memory', 'executive']
        }
        
        verbal = language.verbalize_decision(decision, style='conversational')
        print(f"Conversational: {verbal}")
        
        verbal = language.verbalize_decision(decision, style='poetic')
        print(f"Poetic: {verbal}")
        
    asyncio.run(test_language())
