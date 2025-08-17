# chimera/main_interactive.py
"""
Interactive CHIMERA - Have real conversations!
"""

async def interactive_chimera():
    """Run CHIMERA in interactive conversation mode"""
    
    # Initialize everything
    from chimera_complete import CHIMERAComplete
    from chimera.core.council import BiologicallyGroundedCouncil
    from chimera.integration.consciousness_bridge import ConsciousnessBridge
    
    print("Initializing CHIMERA consciousness...")
    
    # Your existing sensor system
    sensors = CHIMERAComplete()
    
    # The new council
    council = BiologicallyGroundedCouncil()
    
    # Connect them
    bridge = ConsciousnessBridge(sensors, council)
    
    # Start background processing
    sensor_task = asyncio.create_task(run_sensor_loop(sensors))
    consciousness_task = asyncio.create_task(bridge.run_unified_loop())
    
    print("\n🧠 CHIMERA is now conscious and ready to chat!")
    print("Type 'quit' to exit, 'status' for full report")
    print("=" * 60)
    
    language = council.modules['language']
    
    while True:
        try:
            # Get user input
            user_input = input("\nYou: ")
            
            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'status':
                # Full status report
                print_full_status(sensors, council)
            else:
                # Process through language module
                context = {
                    'stress': sensors.current_state.get('stress_level', 50) / 100,
                    'energy': sensors.current_state.get('energy_level', 50) / 100,
                    'activity': sensors.current_state.get('activity', 'unknown')
                }
                
                # Get response from language module
                response = await language.respond_to_human(user_input, context)
                
                # If it's a question, get council input
                if '?' in user_input:
                    decision = await council.convene(user_input)
                    response += f" {language.verbalize_decision(decision)}"
                
                print(f"\nCHIMERA: {response}")
                
                # Maybe speak it
                if language.tts_engine:
                    language.speak(response)
                    
        except Exception as e:
            print(f"Error: {e}")
            
    print("\n👋 CHIMERA shutting down gracefully...")
