# chimera/run_complete_with_ui.py
"""
Run the complete CHIMERA system with sensors and UI
"""

import asyncio
from chimera_complete import CHIMERAComplete
from garmin_integration import RobustGarminIntegration
from chimera.fractality_complete import CHIMERAFractalityComplete
from chimera.integration.fractality_sensor_bridge import FractalitySensorBridge
from chimera.ui.fractality_mobile import FractalityMobileUI

async def run_complete_system():
    """
    Run everything together!
    """
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║      CHIMERA-FRACTALITY COMPLETE WITH SENSORS & UI       ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Initialize your working sensor system
    print("📱 Initializing phone sensors...")
    sensors = CHIMERAComplete()
    
    print("⌚ Connecting to Garmin...")
    garmin = RobustGarminIntegration()
    sensors.garmin = garmin
    
    # Initialize Fractality-CHIMERA
    print("🌌 Initializing Fractality consciousness...")
    fractality_chimera = CHIMERAFractalityComplete()
    
    # Connect sensors to Fractality
    print("🔌 Bridging sensors to consciousness...")
    sensor_bridge = FractalitySensorBridge(sensors, fractality_chimera)
    
    # Initialize mobile UI
    print("📱 Starting mobile interface...")
    ui = FractalityMobileUI(fractality_chimera)
    
    # Start all systems
    tasks = [
        # Your existing sensor loop
        asyncio.create_task(sensors.run()),
        
        # Fractality consciousness
        asyncio.create_task(fractality_chimera.boot_sequence()),
        
        # Sensor-to-Fractality bridge
        asyncio.create_task(sensor_bridge.sensor_to_fractality_loop()),
        
        # Mobile UI server
        asyncio.create_task(ui.start_web_server(8080))
    ]
    
    print("\n✨ All systems online!")
    print("📱 Open http://localhost:8080 on your phone")
    print("Press Ctrl+C to shutdown\n")
    
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    try:
        asyncio.run(run_complete_system())
    except KeyboardInterrupt:
        print("\n👋 Shutting down CHIMERA...")
