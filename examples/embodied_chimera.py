#!/usr/bin/env python3
"""
Launch embodied CHIMERA with full sensor integration
"""
import asyncio
import signal
import sys
from chimera.core.embodied_chimera import EmbodiedCHIMERA
from chimera.interface.embodied_cli import EmbodiedDashboard

async def main():
    print("="*60)
    print("🧠 CHIMERA Embodied Cognition System v0.9")
    print("="*60)
    print()
    print("Initializing consciousness from sensor streams...")
    print()
    
    # Create embodied CHIMERA
    chimera = EmbodiedCHIMERA()
    await chimera.initialize()
    
    # Create dashboard
    dashboard = EmbodiedDashboard(chimera)
    
    # Handle graceful shutdown
    def signal_handler(sig, frame):
        print("\n👋 CHIMERA: Gracefully shutting down...")
        chimera.running = False
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Run both CHIMERA and dashboard
    await asyncio.gather(
        chimera.run_embodied(),
        dashboard.run()
    )

if __name__ == "__main__":
    asyncio.run(main())
