#!/data/data/com.termux/files/usr/bin/bash

echo "🧠 Setting up Embodied CHIMERA for Termux..."

# Core packages
pkg update -y
pkg upgrade -y
pkg install -y python git termux-api

# Python dependencies
pip install --upgrade pip
pip install rich numpy asyncio

# Sensor permissions
echo "📱 Requesting sensor permissions..."
termux-setup-storage

# Test sensor access
echo "Testing sensors..."
termux-sensor -l

# Create CHIMERA directories
mkdir -p ~/.chimera/sensor_logs
mkdir -p ~/.chimera/crystallizations
mkdir -p ~/.chimera/memories

echo "✅ Setup complete!"
echo "Run with: python examples/embodied_chimera.py"
