#!/data/data/com.termux/files/usr/bin/bash
#
# One-shot setup for running CHIMERA on an Android phone via Termux.
# Run it from inside the cloned repo:  bash scripts/setup_termux.sh
#
set -e

echo "🧠  Setting up CHIMERA for your phone..."
echo

# 1. System packages. We install numpy via pkg (not pip) because building it
#    from source on a phone is slow and error-prone.
echo "📦  Installing packages (python, git, termux-api, numpy)..."
pkg update -y
pkg install -y python git termux-api python-numpy

# 2. Python web dependencies (these pip-install cleanly on Termux).
echo "🐍  Installing CHIMERA's web dependencies..."
pip install --upgrade pip
pip install flask flask-socketio flask-cors "python-socketio[client]" websocket-client

# 3. Let Termux ask for storage + sensor access. The FIRST time CHIMERA reads a
#    sensor, Android will pop up a permission request — tap Allow.
echo "📱  Setting up device access..."
termux-setup-storage || true

echo
echo "🔎  Sensors your phone reports:"
termux-sensor -l 2>/dev/null || echo "   (If this failed, install the 'Termux:API' APP — see docs/TERMUX_SETUP.md)"

echo
echo "✅  Done!"
echo
echo "   Start CHIMERA with:   python run.py"
echo "   Then open your phone's browser to:   http://localhost:5000"
echo
echo "   Name your CHIMERA, tap 'Give CHIMERA senses', and move your phone —"
echo "   it will feel it. 🌱"
