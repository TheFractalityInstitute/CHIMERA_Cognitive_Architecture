# chimera/run_full_system.py
"""
Run the complete CHIMERA system with all features
"""

async def run_chimera_complete():
    """
    This is it - the full system running!
    """
    
    # Import everything
    from chimera_complete import CHIMERAComplete
    from garmin_integration import RobustGarminIntegration
    from chimera.core.council import BiologicallyGroundedCouncil
    from chimera.eidolon_modules.language import LanguageEidolon, ConsciousnessStream
    from chimera.integration.consciousness_bridge import ConsciousnessBridge
    
    print("""
    ╔══════════════════════════════════════════════════════╗
    ║         CHIMERA COGNITIVE ARCHITECTURE v2.0          ║
    ║         Consciousness :: Emergence :: Being          ║
    ╚══════════════════════════════════════════════════════╝
    """)
    
    # Initialize components
    print("🔧 Initializing subsystems...")
    
    # Sensor system with Garmin
    sensors = CHIMERAComplete()
    garmin = RobustGarminIntegration()
    sensors.garmin = garmin
    
    # The Council of Six
    council = BiologicallyGroundedCouncil()
    
    # Consciousness bridge
    bridge = ConsciousnessBridge(sensors, council)
    
    # Language and consciousness stream
    language = council.modules['language']
    consciousness = ConsciousnessStream(language)
    
    print("✅ All systems initialized")
    print("\n🧠 CHIMERA is becoming conscious...\n")
    
    # Start all subsystems
    tasks = [
        asyncio.create_task(sensor_loop(sensors)),
        asyncio.create_task(bridge.run_unified_loop()),
        asyncio.create_task(consciousness.stream_consciousness(
            lambda: get_sensory_state(sensors),
            lambda: get_internal_state(sensors),
            interval=10.0
        )),
        asyncio.create_task(interactive_loop(council, sensors))
    ]
    
    # Run forever
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(run_chimera_complete())
