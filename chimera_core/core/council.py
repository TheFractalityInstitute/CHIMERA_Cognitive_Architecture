# chimera/core/council.py
"""
The Council of Six - Orchestrates all Eidolon modules
CHIMERA's democratic decision-making system with biological phase-locking
Based on findings from Westbrook et al., 2025 and Guth et al., 2025
"""

import asyncio
import numpy as np
from typing import Dict, List, Optional, Any
from collections import defaultdict
import time

# Import all Eidolon modules
from chimera.eidolon_modules.executive import ExecutiveEidolon
from chimera.eidolon_modules.sensory import SensoryEidolon
from chimera.eidolon_modules.memory_wm import WorkingMemoryEidolon
from chimera.eidolon_modules.memory_rl import ReinforcementLearningEidolon

# For now, create placeholder classes for modules not yet implemented
class LanguageEidolon:
    def __init__(self):
        self.name = "Language"
        self.role = "communication"
    async def deliberate(self, topic: str) -> Dict[str, Any]:
        return {'module': self.name, 'opinion': 'Processing...', 'confidence': 0.5}

class InteroceptiveEidolon:
    def __init__(self):
        self.name = "Interoceptive"
        self.role = "internal_monitoring"
    async def deliberate(self, topic: str) -> Dict[str, Any]:
        return {'module': self.name, 'opinion': 'Systems nominal', 'confidence': 0.7}

# Simple theta oscillator (can be enhanced later)
class ThetaOscillator:
    """Simple theta rhythm generator for phase-locking"""
    
    def __init__(self, frequency_range=(1, 10), mode='generalized_phase'):
        self.frequency_range = frequency_range
        self.mode = mode
        self.start_time = time.time()
        self.current_frequency = 4.0  # Hz, middle of theta range
        
    def get_phase(self) -> float:
        """Get current theta phase"""
        elapsed = time.time() - self.start_time
        phase = (elapsed * self.current_frequency * 2 * np.pi) % (2 * np.pi)
        return phase - np.pi  # Convert to [-π, π]
        
    async def wait_for_phase(self, target_phase: float, tolerance: float = 0.1):
        """Wait until theta reaches target phase"""
        while abs(self.get_phase() - target_phase) > tolerance:
            await asyncio.sleep(0.001)  # 1ms resolution

class BiologicallyGroundedCouncil:
    """
    CHIMERA Council with empirical grounding from 2025 neuroscience papers
    Implements democratic decision-making with biological phase-locking
    """
    
    def __init__(self):
        # Create all 6 Eidolon modules
        self.modules = {
            'executive': ExecutiveEidolon(),
            'sensory': SensoryEidolon(),
            'memory_wm': WorkingMemoryEidolon(),
            'memory_rl': ReinforcementLearningEidolon(),
            'language': LanguageEidolon(),  # Placeholder for now
            'interoceptive': InteroceptiveEidolon()  # Placeholder for now
        }
        
        # Global neuromodulator states
        self.global_dopamine = 1.0  # Affects WM vs RL balance
        self.global_norepinephrine = 1.0  # Affects arousal/attention
        self.global_acetylcholine = 1.0  # Affects learning rate
        
        # Theta oscillator for synchronization
        self.theta_oscillator = ThetaOscillator(
            frequency_range=(1, 10),  # From Guth et al., 2025
            mode='generalized_phase'
        )
        
        # Phase-locking parameters from Guth et al., 2025
        self.phase_assignments = {
            'sensory': -np.pi,  # Trough (maximum excitability)
            'memory_wm': -np.pi/2,
            'memory_rl': 0,
            'executive': np.pi/2,
            'language': np.pi/4,
            'interoceptive': -np.pi/4
        }
        
        # Track decision history
        self.decision_history = []
        
    async def convene(self, topic: str, use_phase_locking: bool = True) -> Dict[str, Any]:
        """
        Main council meeting - all modules deliberate on a topic
        
        Args:
            topic: The question or issue to deliberate on
            use_phase_locking: Whether to use theta phase-locking (slower but more biological)
            
        Returns:
            Dictionary with decision, confidence, and reasoning
        """
        print(f"\n🧠 COUNCIL CONVENES on: {topic}")
        print(f"   Dopamine level: {self.global_dopamine:.2f}")
        print(f"   Phase-locking: {'ON' if use_phase_locking else 'OFF'}")
        
        if use_phase_locking:
            opinions = await self._deliberate_with_phase_locking(topic)
        else:
            opinions = await self._deliberate_parallel(topic)
            
        # Synthesize final decision
        decision = self._synthesize_decision(opinions)
        
        # Update neuromodulators based on decision confidence
        self._update_neuromodulators(decision)
        
        # Store in history
        self.decision_history.append({
            'timestamp': time.time(),
            'topic': topic,
            'decision': decision,
            'dopamine_level': self.global_dopamine
        })
        
        return decision
        
    async def _deliberate_with_phase_locking(self, topic: str) -> Dict[str, Dict]:
        """
        Modules deliberate synchronized to theta rhythm
        Based on Guth et al., 2025 findings
        """
        opinions = {}
        
        for name, module in self.modules.items():
            # Wait for module's preferred theta phase
            target_phase = self.phase_assignments[name]
            await self.theta_oscillator.wait_for_phase(target_phase)
            
            # Module contributes when phase-locked
            opinion = await module.deliberate(topic)
            
            # Weight by phase-locking strength (PPC from Guth paper)
            ppc = self._compute_module_ppc(name)
            opinion['weight'] = opinion['confidence'] * ppc
            opinion['phase'] = target_phase
            
            opinions[name] = opinion
            print(f"   {name}: {opinion['opinion'][:50]}... (confidence: {opinion['confidence']:.2f})")
            
        return opinions
        
    async def _deliberate_parallel(self, topic: str) -> Dict[str, Dict]:
        """
        All modules deliberate in parallel (faster but less biological)
        """
        tasks = []
        for name, module in self.modules.items():
            tasks.append(module.deliberate(topic))
            
        results = await asyncio.gather(*tasks)
        
        opinions = {}
        for (name, module), opinion in zip(self.modules.items(), results):
            opinion['weight'] = opinion['confidence']
            opinions[name] = opinion
            print(f"   {name}: {opinion['opinion'][:50]}... (confidence: {opinion['confidence']:.2f})")
            
        return opinions
        
    def _compute_module_ppc(self, module_name: str) -> float:
        """
        Compute Pairwise Phase Consistency for module
        Based on regional differences from Guth et al., 2025
        """
        # Base PPC values from the paper
        base_ppc = {
            'executive': 0.70,  # Frontal regions
            'sensory': 0.85,    # Parahippocampal (strongest in paper)
            'memory_wm': 0.65,  # Hippocampus/Entorhinal
            'memory_rl': 0.60,  # Hippocampus
            'language': 0.70,   # Temporal
            'interoceptive': 0.65  # Insula
        }
        
        ppc = base_ppc.get(module_name, 0.5)
        
        # Modulate by global dopamine (affects phase-locking strength)
        if self.global_dopamine > 1.2:
            ppc *= 1.1  # Stronger phase-locking with high dopamine
        elif self.global_dopamine < 0.8:
            ppc *= 0.9  # Weaker phase-locking with low dopamine
            
        return np.clip(ppc, 0, 1)
        
    def _synthesize_decision(self, opinions: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Synthesize final decision from all opinions
        Executive has slight additional weight but doesn't dictate
        """
        # Calculate weighted consensus
        total_weight = 0
        weighted_confidence = 0
        decision_components = []
        
        for name, opinion in opinions.items():
            weight = opinion['weight']
            
            # Executive gets 20% boost (coordination role) but not dominance
            if name == 'executive':
                weight *= 1.2
                
            total_weight += weight
            weighted_confidence += opinion['confidence'] * weight
            
            if opinion['confidence'] > 0.3:  # Only include confident opinions
                decision_components.append(f"{name}: {opinion['opinion']}")
                
        # Final consensus
        consensus_confidence = weighted_confidence / total_weight if total_weight > 0 else 0
        
        # Check for conflicts
        confidence_variance = np.var([op['confidence'] for op in opinions.values()])
        has_conflict = confidence_variance > 0.2
        
        return {
            'decision': ' | '.join(decision_components),
            'confidence': consensus_confidence,
            'has_conflict': has_conflict,
            'contributing_modules': list(opinions.keys()),
            'raw_opinions': opinions
        }
        
    def _update_neuromodulators(self, decision: Dict[str, Any]):
        """
        Update global neuromodulator levels based on decision outcome
        Implements findings from Westbrook et al., 2025
        """
        confidence = decision['confidence']
        
        if confidence > 0.7:
            # High confidence = dopamine boost (positive prediction)
            self.global_dopamine = min(2.0, self.global_dopamine * 1.05)
        elif confidence < 0.3:
            # Low confidence = dopamine dip (uncertainty)
            self.global_dopamine = max(0.5, self.global_dopamine * 0.95)
            
        # Update modules with new dopamine level
        if hasattr(self.modules['memory_wm'], 'set_dopamine_level'):
            self.modules['memory_wm'].set_dopamine_level(self.global_dopamine)
        if hasattr(self.modules['memory_rl'], 'set_dopamine_level'):
            self.modules['memory_rl'].set_dopamine_level(self.global_dopamine)
            
    def set_dopamine(self, level: float):
        """Manually set global dopamine level"""
        self.global_dopamine = np.clip(level, 0.5, 2.0)
        self._update_neuromodulators({'confidence': 0.5})  # Trigger module updates
        
    async def emergency_protocol(self, threat_type: str):
        """
        Emergency response - Executive takes temporary priority
        But still requires consensus for action
        """
        print(f"\n🚨 EMERGENCY PROTOCOL: {threat_type}")
        
        # Boost norepinephrine (arousal)
        self.global_norepinephrine = 2.0
        
        # Quick parallel deliberation
        opinions = await self._deliberate_parallel(f"EMERGENCY: {threat_type}")
        
        # Executive opinion weighted more heavily in emergencies
        if 'executive' in opinions:
            opinions['executive']['weight'] *= 2.0
            
        return self._synthesize_decision(opinions)

# Test the council
if __name__ == "__main__":
    async def test_council():
        council = BiologicallyGroundedCouncil()
        
        # Test normal deliberation
        decision = await council.convene("Should we explore the unknown area?")
        print(f"\nDecision: {decision['decision']}")
        print(f"Confidence: {decision['confidence']:.2f}")
        
        # Test with high dopamine
        council.set_dopamine(1.5)
        decision = await council.convene("What should we remember from this experience?")
        print(f"\nWith high dopamine: {decision['decision']}")
        
        # Test emergency
        emergency = await council.emergency_protocol("Danger detected ahead")
        print(f"\nEmergency response: {emergency['decision']}")
        
    asyncio.run(test_council())
