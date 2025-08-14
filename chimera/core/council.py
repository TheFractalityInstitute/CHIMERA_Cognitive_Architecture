# chimera/core/council.py
"""
The Council of Six - Orchestrates all Eidolon modules
This is where they come together for democratic decision-making
"""

from chimera.eidolon_modules.executive import ExecutiveEidolon
from chimera.eidolon_modules.sensory import SensoryEidolon
from chimera.eidolon_modules.memory_wm import WorkingMemoryEidolon
from chimera.eidolon_modules.memory_rl import ReinforcementLearningEidolon
# ... import other modules

class CHIMERACouncil:
    """Brings all 6 modules together"""
    
    def __init__(self):
        # Create all 6 modules
        self.modules = {
            'executive': ExecutiveEidolon(),
            'sensory': SensoryEidolon(),
            'memory_wm': WorkingMemoryEidolon(),
            'memory_rl': ReinforcementLearningEidolon(),
            # ... etc
        }
