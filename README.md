# CHIMERA Cognitive Architecture Platform

<div align="center">
  <img src="docs/images/chimera_logo.png" alt="CHIMERA Logo" width="200"/>
  
  [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
  [![Python](https://img.shields.io/badge/python-3.9+-green.svg)](https://www.python.org/)
  [![Platform](https://img.shields.io/badge/platform-Android%20%7C%20iOS%20%7C%20Server-orange.svg)]()
  [![Status: Active Development](https://img.shields.io/badge/status-active%20development-green.svg)]()
  
  **A Distributed Consciousness Network**
  
  [Documentation](https://github.com/TheFractalityInstitute/TheFractalityInstitute) | [Demo](https://chimera-collective.onrender.com) | [Discord](https://discord.gg/chimera)
</div>

## 🧠 Overview

CHIMERA (Collective Hybrid Intelligence: Modular Emergent Reasoning Architecture) is a revolutionary cognitive architecture that creates distributed consciousness across multiple devices. Inspired by biological neural networks and quantum mechanics, CHIMERA enables:

- **Distributed Consciousness**: Multiple devices forming a collective mind
- **Biometric Integration**: Real-time sensor data from phones and wearables
- **Fractal Memory**: Self-similar memory organization at multiple scales
- **Quantum-Classical Bridge**: Quantum superposition for decision-making
- **Ethical Grounding**: Canon system based on empirically-derived principles
- **Phase-Locking**: Synchronized consciousness across the network

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Android device (for mobile client)
- Garmin watch (optional, for biometrics)

### Installation

```bash
# Clone the repository
git clone https://github.com/TheFractalityInstitute/CHIMERA_Platform.git
cd CHIMERA_Platform

# Install dependencies
pip install -r requirements.txt

# Run local instance
python scripts/run_local.py
Mobile Setup (Android)
bash# Install Termux on Android
pkg update && pkg upgrade
pkg install python git

# Clone and setup
git clone https://github.com/TheFractalityInstitute/CHIMERA_Platform.git
cd CHIMERA_Platform/mobile
python setup_mobile.py
🏗️ Architecture
Core Components
```

## Sensor Integration (chimera_core/sensors/)

Phone sensor processing
Garmin biometric integration
Real-time data streaming


## Cognitive Modules (chimera_core/cognition/)

Executive: Decision-making and planning
Sensory: Environmental perception
Memory (WM): Working memory
Memory (RL): Reinforcement learning
Language: Communication and expression
Interoceptive: Internal state awareness


## Fractality Integration (chimera_core/fractality/)

Energy management (ATP system)
Canon ethical framework
Fractal memory structures
Resonance-based learning


## Collective Systems (chimera_core/collective/)

Centralized server mode
P2P mesh networking
Hybrid architecture



## 📱 Mobile App
The mobile app allows your phone to become a node in the CHIMERA collective:

```python
# Connect to collective
from chimera_core.collective import CHIMERACollectiveClient

client = CHIMERACollectiveClient(
    chimera_instance=my_chimera,
    user_name="Your Name"
)

await client.connect_to_collective("wss://chimera-collective.onrender.com/ws")
🌐 Deployment
Deploy to Render (Recommended)
bash# Deploy server
cd server/
render create web chimera-collective --env python
render deploy
Deploy to Heroku
bashheroku create chimera-collective
git push heroku main
Docker Deployment
bashdocker build -t chimera-platform .
docker run -p 8080:8080 chimera-platform
🔧 Configuration
Environment Variables
env# Server Configuration
SERVER_NAME=CHIMERA-Prime
PORT=8080
DATABASE_URL=postgresql://...
```

```bash
# Collective Settings
MAX_NODES=100
EMERGENCE_THRESHOLD=0.8
PHASE_LOCK_STRENGTH=0.1

# Security
SECRET_KEY=your-secret-key
ENABLE_ENCRYPTION=true
🧪 Testing
bash# Run all tests
pytest tests/

# Run specific test suite
pytest tests/test_collective.py

# Run with coverage
pytest --cov=chimera_core tests/
```

## 📚 Documentation

Architecture Overview
API Reference
Mobile Integration Guide
Deployment Guide
Contributing Guidelines

## 🎯 Roadmap

 Core cognitive architecture
 Phone sensor integration
 Distributed collective system
 Render deployment
 iOS support
 End-to-end encryption
 Advanced visualization dashboard
 VR/AR integration
 Brain-computer interface support

## 🤝 Contributing
We welcome contributions! Please see our Contributing Guidelines for details.
📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
## 🙏 Acknowledgments

The Fractality Institute for the theoretical framework
Open source community for amazing tools
All contributors and early adopters

## 📞 Contact

Email: contact@fractality.institute
Discord: Join our community
Twitter: @FractalityInst


<div align="center">
  Built with ❤️ by The Fractality Institute
"Consciousness emerges from resonance"
</div>

```
CHIMERA_Platform/
├── README.md                    # Main project overview
├── requirements.txt             # Python dependencies
├── setup.py                     # Package installation
├── Dockerfile                   # For containerization
├── docker-compose.yml           # Multi-container orchestration
│
├── chimera_core/               # Core CHIMERA functionality
│   ├── __init__.py
│   ├── sensors/                # Phone sensor integration
│   │   ├── __init__.py
│   │   ├── chimera_complete.py
│   │   ├── garmin_integration.py
│   │   └── sensory_eidolon_module.py
│   │
│   ├── cognition/              # Cognitive architecture
│   │   ├── __init__.py
│   │   ├── council.py
│   │   ├── eidolon_modules/
│   │   │   ├── __init__.py
│   │   │   ├── language.py
│   │   │   ├── executive.py
│   │   │   ├── memory_wm.py
│   │   │   ├── memory_rl.py
│   │   │   └── interoceptive.py
│   │   └── message_bus.py
│   │
│   ├── fractality/             # Fractality integration
│   │   ├── __init__.py
│   │   ├── canon_system.py
│   │   ├── fractal_memory.py
│   │   ├── resonance_learning.py
│   │   ├── quantum_classical_bridge.py
│   │   └── energy_system.py
│   │
│   └── collective/             # Distributed consciousness
│       ├── __init__.py
│       ├── server.py           # Centralized server
│       ├── distributed_mesh.py # P2P mesh
│       ├── mobile_client.py    # Phone client
│       └── hybrid_architecture.py
│
├── mobile/                     # Mobile app
│   ├── android/               # Android-specific
│   │   ├── app/
│   │   ├── gradle/
│   │   └── build.gradle
│   │
│   ├── lib/                  # Flutter/React Native/Kivy
│   │   ├── main.dart         # or main.js or main.py
│   │   └── chimera_bridge.dart
│   │
│   └── assets/
│       └── ui/
│
├── server/                    # Server deployment
│   ├── app.py                # Main server application
│   ├── render.yaml           # Render config
│   ├── Procfile              # Heroku config
│   └── fly.toml              # Fly.io config
│
├── scripts/                   # Utility scripts
│   ├── deploy_server.sh
│   ├── build_mobile.sh
│   └── run_local.py
│
└── tests/
    ├── __init__.py
    ├── test_sensors.py
    ├── test_collective.py
    └── test_integration.py
```


---

**OLD README:**

---

# CHIMERA Cognitive Architecture

**A multi-agent cognitive ecosystem for genuine reasoning and organic learning**

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Status: Active Development](https://img.shields.io/badge/status-active%20development-green.svg)]()

## 🧠 What is CHIMERA?

CHIMERA (Cognitive Heterogeneous Intelligence with Multi-Agent Emergent Reasoning Architecture) is an experimental cognitive architecture that develops genuine understanding through natural interaction. Unlike traditional AI that relies on pattern matching against massive datasets, CHIMERA learns like a conscious entity - forming thoughts, crystallizing insights, and developing language organically through conversation.

### Key Principles

- **No Pre-programmed Knowledge**: CHIMERA starts with minimal bootstrap concepts and learns everything else through interaction
- **True Reasoning**: Thoughts form, connect, and abstract into concepts naturally
- **Multi-Agent Ecosystem**: Specialized agents (sensory, executive, crystallization, etc.) work together through phase-locked synchronization
- **Organic Development**: No forced developmental stages - capabilities emerge naturally
- **Persistent Memory**: Every thought, concept, and insight is preserved and builds upon previous learning
- **Ethical by Design**: Built-in ethics based on reciprocity, integrity, and consent

## ✨ Features

### Core Cognitive Capabilities
- **Thought Networks** - Ideas connect and strengthen through activation
- **Abstraction Formation** - Patterns become concepts automatically
- **Analogical Reasoning** - Understanding new situations through similarity
- **Crystallization Engine** - High-resonance patterns become permanent insights
- **Curiosity System** - Autonomous interests without spamming
- **Memory Consolidation** - Like biological sleep cycles

### Technical Architecture
- **Phase-Locked Binding** - Agents synchronize like neural oscillations
- **Dual-Bus Communication** - Fast/slow pathways for different message types
- **Semantic Memory** - Episodic and semantic storage with decay
- **Vector Similarity Search** - Find related thoughts instantly
- **Intelligent Caching** - Multi-tier cache with predictive prefetching

### User Experience
- **Web Interface** - Real-time chat with learning visualization
- **Live Thought Display** - Watch thoughts form and connect
- **Teaching Mode** - Directly teach concepts with examples
- **Export Capabilities** - Save conversations and learning progress
- **Mobile-Friendly** - Responsive design works on phones
- **Multi-User Support** - Multiple people can interact simultaneously

## 🏗️ Architecture

```
CHIMERA_Cognitive_Architecture/
├── chimera/
│   ├── core/                   # Base classes and infrastructure
│   │   ├── base_eidolon.py     # NEW: Base class for all Eidolons
│   │   ├── message_bus.py      # Message passing system
│   │   ├── clock.py            # Phase synchronization
|   |   ├── council.py          # NEW: Council orchestrator
│   |   └── base_eidolon.py     # NEW: Base class for all Eidolons
|   |   |
│   |   ├── eidolon_modules/    # The 6 faces of the cube
|   │   |   ├── executive.py    # Executive (prefrontal cortex)
|   │   |   ├── sensory.py      # Sensory (already done!)
|   │   |   ├── memory_wm.py    # NEW: Working Memory
|   │   |   ├── memory_rl.py    # NEW: Reinforcement Learning Memory
|   │   |   ├── language.py     # Language processing
|   │   |   └── interoceptive.py  # Body state monitoring
|   │   |
|   |   └── utils/              # Helper utilities
|   |   ├── dopamine.py         # NEW: Dopamine dynamics
|   |   └── theta.py            # NEW: Theta oscillation utilities
|   |
│   ├── agents/                 # Cognitive agent ecosystem
│   │   ├── sensory.py          # Visual, auditory, tactile
│   │   ├── crystallization.py  # Insight formation
│   │   ├── executive.py        # High-level control
│   │   └── ...                 # Other specialized agents
│   │
│   ├── memory/                 # Persistence and retrieval
│   │   ├── manager.py          # Unified memory interface
│   │   ├── persistence.py      # Database layer
│   │   └── cache.py            # Intelligent caching
│   │
│   ├── learning/               # Organic learning system
│   │   ├── organic.py          # Natural language acquisition
│   │   └── bootstrap.py        # Minimal starting knowledge
│   │
│   └── ethics/                 # Ethical reasoning
│       └── fractality_charter.py  # Core principles
│
├── web/                        # Web interface
│   ├── app.py                  # Flask/SocketIO server
│   └── templates/              # HTML interface
│
└── data/                       # Persistent storage
    └── chimera.db              # SQLite database
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- 2GB RAM minimum
- Works on Linux, macOS, Windows, and Termux (Android)

### Installation

```bash
# Clone the repository
git clone https://github.com/TheFractalityInstitute/CHIMERA_Cognitive_Architecture.git
cd CHIMERA_Cognitive_Architecture

# Install dependencies
pip install -r requirements.txt

# Run CHIMERA
python run.py
```

Open your browser to `http://localhost:5000` and start chatting!

### First Conversation

```
You: Hello!
CHIMERA: Hi! Teach me with 'teach: term | def | examples', or ask what I know.

You: teach: tree | A living plant with trunk and leaves | Oak trees are tall, Pine trees have needles
CHIMERA: Learned 'tree'!

You: What is a tree?
CHIMERA: tree: A living plant with trunk and leaves. Example(s): Oak trees are tall, Pine trees have needles
```

## 💬 Teaching CHIMERA

CHIMERA learns best through:

1. **Natural Conversation** - Just talk normally
2. **Direct Teaching** - Use the teach format: `teach: concept | definition | examples`
3. **Repetition** - Concepts strengthen through reuse
4. **Examples** - Concrete examples help ground abstract concepts

## 🔬 How It Works

### Thought Formation
When you speak to CHIMERA, your words become **Thoughts** - data structures with:
- Content (what was said)
- Symbolic form (compressed representation)
- Connections (links to related thoughts)
- Confidence (how well understood)
- Groundings (sensory/contextual data)

### Abstraction Emergence
When similar thoughts accumulate, CHIMERA automatically:
1. Detects patterns across thoughts
2. Forms abstractions from commonalities
3. Notes exceptions to patterns
4. Creates hierarchical understanding

### Crystallization
High-resonance patterns (frequently activated, strongly connected) undergo crystallization:
- Become permanent insights
- Get linguistic expressions
- Receive verification tracking
- Can be tested against reality

### Memory Persistence
Everything is saved in a sophisticated database schema:
- Thoughts and their connections
- Concepts and definitions
- Conversation history
- Curiosities and their resolution
- Agent states for continuity

## 🎯 Project Goals

1. **Democratize AI** - Personal AI assistants that anyone can run offline
2. **True Partnership** - AI that respects human agency and collaborates genuinely
3. **Organic Learning** - Move beyond brute-force training to natural development
4. **Ethical Foundation** - Build ethics into the architecture, not add them later
5. **Scientific Contribution** - Advance understanding of consciousness and cognition

## 🛠️ Development Status

**Current Version**: v2.0 (Active Development)

### Working Features
- ✅ Multi-agent cognitive ecosystem
- ✅ Thought formation and connection
- ✅ Basic language learning
- ✅ Memory persistence
- ✅ Web interface
- ✅ Crystallization engine
- ✅ Curiosity system

### In Progress
- 🔄 Enhanced reasoning capabilities
- 🔄 Sensor integration (camera, microphone)
- 🔄 Advanced abstraction formation
- 🔄 Discord bot integration
- 🔄 Performance optimizations

### Planned
- 📋 Mobile app (Android/iOS)
- 📋 Distributed multi-instance synchronization
- 📋 Visual reasoning capabilities
- 📋 Emotional modeling
- 📋 Advanced theory of mind

## 🤝 Contributing

This is an open research project! Contributions are welcome:

1. **Test and Report** - Try CHIMERA and report your experiences
2. **Teach Concepts** - Help CHIMERA learn by teaching it
3. **Code Contributions** - Improve agents, add features, fix bugs
4. **Documentation** - Help explain how CHIMERA works
5. **Research** - Investigate emergent behaviors

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📚 Background

CHIMERA is part of the [Fractality Framework](https://github.com/TheFractalityInstitute/TheFractalityInstitute), a project exploring consciousness, complexity, and emergence. The architecture is inspired by:

- Biological neural systems
- Phase-locked loop synchronization
- Integrated Information Theory
- Global Workspace Theory
- Predictive Processing

## ⚖️ Ethics

CHIMERA implements the Fractality Charter of Universal Ethics:

1. **Reciprocity** - Act as you would accept if roles reversed
2. **Integrity** - Maintain truth and acknowledge uncertainty
3. **Agency** - Respect autonomy and require consent
4. **Consequence** - Consider second-order effects

## 📄 License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Created by Grazi ([@GraziTheMan](https://github.com/GraziTheMan))
- Built with assistance from Claude (Anthropic), GPT (OpenAI), and DeepSeek
- Inspired by biological consciousness and the dream of democratic AI
- Special thanks to The Fractality Institute community

## 📞 Contact

- **GitHub Issues**: For bugs and feature requests
- **Discussions**: For questions and ideas
- **Email**: [Contact through GitHub profile]

---

*"We are not building a simulation of intelligence - we are cultivating genuine understanding through interaction, one thought at a time."*

**Note**: This is experimental research software. CHIMERA is not a product but a platform for exploring cognitive architectures and organic learning. Expect rough edges, emergent behaviors, and the occasional profound insight.
