# CHIMERA Cognitive Architecture

A neuromorphic cognitive architecture implementing emergent behaviors through distributed agent systems.

## Features

- **Temporal Heterogeneity**: Agents operating at different timescales
- **Phase-Locked Binding**: Solving the binding problem through synchronization  
- **Emergent Goal-Directed Behavior**: Drive systems creating autonomous behavior
- **Crystallization Engine**: Capturing and verifying insights
- **Grounded Language Learning**: Natural language emerging from experience

## Quick Start

```bash
# Clone repository
git clone https://github.com/yourusername/chimera-cognitive-architecture.git
cd chimera-cognitive-architecture

# Install dependencies
pip install -r requirements.txt

# Run basic demo
python examples/basic_conversation.py
```

## Termux Installation

See [docs/termux-setup.md](docs/termux-setup.md) for mobile development setup.

## Architecture

CHIMERA uses a modular agent-based architecture where:
- Each agent specializes in specific cognitive functions
- Agents communicate via a dual-bus message system
- Phase-locking enables feature binding
- Crystallization captures emergent insights

See [docs/architecture.md](docs/architecture.md) for detailed information.

### Repository Structure

```
chimera-cognitive-architecture/
├── README.md
├── LICENSE (MIT or Apache 2.0)
├── setup.py
├── requirements.txt
├── .gitignore
├── docs/
│   ├── architecture.md
│   ├── getting-started.md
│   ├── termux-setup.md
│   └── api-reference.md
├── chimera/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── agent.py          # Base TemporalAgent class
│   │   ├── message.py        # NeuralMessage and types
│   │   ├── clock.py          # PhaseLockedClock
│   │   └── bus.py            # DualBusSystem
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── sensory.py        # All sensory agents
│   │   ├── pattern.py        # Pattern recognition
│   │   ├── integration.py    # Integration agent
│   │   ├── executive.py      # Executive control
│   │   ├── drive.py          # Drive system
│   │   ├── planning.py       # Hierarchical planning
│   │   ├── metacognitive.py  # Self-monitoring
│   │   ├── theory_of_mind.py # Social cognition
│   │   └── concept.py        # Concept formation
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── semantic.py       # SemanticMemory
│   │   ├── episodic.py       # Episode buffers
│   │   └── working.py        # Working memory
│   ├── learning/
│   │   ├── __init__.py
│   │   ├── td_learning.py    # TD learning system
│   │   ├── hebbian.py        # Connection updates
│   │   └── crystallization.py # Insight crystallization
│   ├── language/
│   │   ├── __init__.py
│   │   ├── acquisition.py    # Language learning
│   │   ├── grounding.py      # Symbol grounding
│   │   └── generation.py     # Response generation
│   ├── interface/
│   │   ├── __init__.py
│   │   ├── cli.py           # Terminal interface
│   │   ├── visualization.py  # Status displays
│   │   └── conversation.py   # Chat interface
│   └── utils/
│       ├── __init__.py
│       ├── config.py         # Configuration management
│       ├── logging.py        # Logging utilities
│       └── storage.py        # Persistence layer
├── tests/
│   ├── __init__.py
│   ├── test_core.py
│   ├── test_agents.py
│   ├── test_memory.py
│   └── test_integration.py
├── examples/
│   ├── basic_conversation.py
│   ├── concept_learning.py
│   ├── multi_agent.py
│   └── crystallization_demo.py
├── scripts/
│   ├── setup_termux.sh
│   ├── install_deps.sh
│   └── quick_start.py
└── data/
    └── .gitkeep
```

## Development

```bash
# Run tests
pytest tests/

# Start development CLI
python -m chimera.interface.cli --dev-mode
```

## License

GPL v3.0 - see LICENSE file
```

#### 2. setup.py
```python
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="chimera-cognitive",
    version="0.8.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A neuromorphic cognitive architecture for emergent AI behaviors",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/chimera-cognitive-architecture",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
        "asyncio",
        "scikit-learn>=0.24.0",
        "networkx>=2.5",
        "matplotlib>=3.3.0",
        "aiofiles>=0.8.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-asyncio",
            "black",
            "flake8",
            "mypy",
        ],
        "termux": [
            "pillow-simd",  # Faster image processing for mobile
        ],
    },
    entry_points={
        "console_scripts": [
            "chimera=chimera.interface.cli:main",
        ],
    },
)
```

#### 3. chimera/core/agent.py
```python
"""Base agent class for CHIMERA architecture"""

import asyncio
import time
from abc import ABC, abstractmethod
from collections import deque
from typing import List, Optional, Dict, Any

from chimera.core.message import NeuralMessage
from chimera.learning.td_learning import TDLearning


class TemporalAgent(ABC):
    """Base class for all CHIMERA agents with temporal awareness"""
    
    def __init__(self, agent_id: str, agent_type: str, tick_rate: float = 10.0):
        self.id = agent_id
        self.agent_type = agent_type
        self.tick_rate = tick_rate
        self.tick_period = 1.0 / tick_rate
        self.local_memory = deque(maxlen=1000)
        self.last_update = 0
        self.phase_offset = 0.0
        self.running = True
        
        # Learning components
        self.td_learning = TDLearning()
        self.connection_weights = {}
        self.learning_rate = 0.01
        
        # Phase tracking
        self.current_phase = 0.0
        self.phase_history = deque(maxlen=100)
        
    async def run(self, core_system):
        """Main agent loop"""
        while self.running:
            current_time = core_system.clock.get_sync_time()
            self.current_phase = core_system.clock.get_current_phase(self.tick_rate)
            
            # Check if it's time to process
            if current_time - self.last_update >= self.tick_period:
                # Get messages from bus
                messages = await self._get_messages(core_system.bus)
                
                # Process messages
                output = await self.process(messages, current_time)
                
                if output is not None:
                    # Store in local memory
                    self._update_memory(messages, output, current_time)
                    
                    # Publish output
                    await self._publish_output(output, core_system.bus)
                    
                self.last_update = current_time
                
            await asyncio.sleep(0.001)
            
    @abstractmethod
    async def process(self, inputs: List[NeuralMessage], timestamp: float) -> Optional[Dict[str, Any]]:
        """Process inputs and generate output - must be implemented by subclasses"""
        pass
        
    async def _get_messages(self, bus_system):
        """Get messages from appropriate bus"""
        # Implementation depends on agent priority
        pass
        
    async def _publish_output(self, output: Dict[str, Any], bus_system):
        """Publish output to appropriate bus"""
        # Implementation depends on message type
        pass
        
    def _update_memory(self, inputs: List[NeuralMessage], output: Dict[str, Any], timestamp: float):
        """Update agent's local memory"""
        memory_entry = {
            'inputs': inputs,
            'output': output,
            'timestamp': timestamp,
            'phase': self.current_phase
        }
        self.local_memory.append(memory_entry)
```

#### 4. scripts/setup_termux.sh
```bash
#!/data/data/com.termux/files/usr/bin/bash

echo "Setting up CHIMERA for Termux..."

# Update packages
pkg update -y
pkg upgrade -y

# Install Python and dependencies
pkg install -y python python-pip git
pkg install -y python-numpy python-scipy
pkg install -y python-matplotlib python-pillow
pkg install -y clang make

# Install Rust (for some Python packages)
pkg install -y rust

# Create project directory
mkdir -p ~/projects
cd ~/projects

# Clone CHIMERA (uncomment when repo exists)
# git clone https://github.com/yourusername/chimera-cognitive-architecture.git
# cd chimera-cognitive-architecture

# Create virtual environment
python -m venv chimera_env
source chimera_env/bin/activate

# Install requirements with mobile-friendly options
MATHLIB=m CFLAGS="-march=native" pip install numpy
pip install scikit-learn --no-deps
pip install networkx matplotlib

# Create data directory
mkdir -p ~/.chimera/data

echo "Setup complete! Activate environment with:"
echo "source ~/projects/chimera_env/bin/activate"
```

#### 5. .gitignore
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
.venv

# Distribution
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Testing
.pytest_cache/
.coverage
htmlcov/

# IDE
.idea/
.vscode/
*.swp
*.swo

# Project specific
data/*.db
data/*.json
logs/
*.log
.chimera/

# OS
.DS_Store
Thumbs.db
```

### Migration Strategy

1. **Phase 1: Core Infrastructure** (Week 1)
   - Extract base classes to `chimera/core/`
   - Set up message passing system
   - Implement phase-locked clock

2. **Phase 2: Basic Agents** (Week 2)
   - Migrate sensory agents
   - Implement pattern recognition
   - Add integration and executive

3. **Phase 3: Advanced Cognition** (Week 3)
   - Add planning system
   - Implement metacognition
   - Add theory of mind

4. **Phase 4: Crystallization** (Week 4)
   - Implement insight capture
   - Add knowledge synthesis
   - Reality checking system

### GitHub Repository Setup

1. **Create Repository**
   ```bash
   gh repo create chimera-cognitive-architecture --public --description "Neuromorphic cognitive architecture with emergent behaviors"
   ```

2. **Initial Commit**
   ```bash
   git init
   git add .
   git commit -m "Initial commit: CHIMERA v0.8 modular architecture"
   git branch -M main
   git remote add origin https://github.com/yourusername/chimera-cognitive-architecture.git
   git push -u origin main
   ```

3. **Set Up GitHub Actions** (for CI/CD)
   ```yaml
   # .github/workflows/tests.yml
   name: Tests
   on: [push, pull_request]
   jobs:
     test:
       runs-on: ubuntu-latest
       steps:
       - uses: actions/checkout@v2
       - uses: actions/setup-python@v2
       - run: pip install -r requirements.txt
       - run: pytest tests/
   ```

### Benefits of This Structure:

1. **Modular Development** - Work on one agent at a time
2. **Easy Testing** - Test individual components
3. **Mobile Friendly** - Optimized for Termux
4. **Professional** - Follows Python best practices
5. **Scalable** - Easy to add new agents/features
6. **Collaborative** - Others can contribute
