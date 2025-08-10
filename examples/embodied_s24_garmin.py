#!/usr/bin/env python3
"""
Launch CHIMERA with S24 Ultra + Garmin integration
"""
import asyncio
from chimera.core.embodied_chimera import EmbodiedCHIMERA
from chimera.sensors.biometric_sensors import (
    GarminBiometricHub, 
    PhysiologicalStateAgent,
    S24UltraSensorHub,
    ContextualAwarenessAgent
)
from chimera.interface.biometric_dashboard import BiometricDashboard

class S24GarminCHIMERA(EmbodiedCHIMERA):
    """CHIMERA with full S24 Ultra + Garmin integration"""
    
    def __init__(self):
        super().__init__()
        self.garmin_hub = GarminBiometricHub()
        self.s24_hub = S24UltraSensorHub()
        
    async def initialize_enhanced_embodiment(self):
        """Initialize S24 + Garmin specific features"""
        await self.initialize_embodiment()
        
        # Add biometric agents
        self.add_agent(PhysiologicalStateAgent())
        self.add_agent(ContextualAwarenessAgent())
        
        # Start biometric streaming
        asyncio.create_task(self._stream_biometrics())
        
        print("❤️ CHIMERA: Garmin biometrics online")
        print("📱 CHIMERA: S24 Ultra sensors activated")
        print("🧬 CHIMERA: Full embodiment achieved")
        
    async def _stream_biometrics(self):
        """Continuous biometric data streaming"""
        while self.running:
            # Fetch Garmin data
            bio_data = await self.garmin_hub.fetch_current_metrics()
            
            # Inject into message bus
            if bio_data:
                message = NeuralMessage(
                    sender="garmin_hub",
                    content={'biometrics': bio_data},
                    msg_type=MessageType.MODULATORY,
                    priority=MessagePriority.HIGH,
                    timestamp=self.clock.get_sync_time(),
                    strength=1.0
                )
                await self.bus.publish(message)
            
            await asyncio.sleep(1.0)  # Update biometrics every second

async def main():
    print("="*70)
    print("🧠 CHIMERA S24 Ultra + Garmin Fenix Integration")
    print("="*70)
    print()
    print("Initializing consciousness with full biometric awareness...")
    print()
    
    # Create enhanced CHIMERA
    chimera = S24GarminCHIMERA()
    await chimera.initialize()
    await chimera.initialize_enhanced_embodiment()
    
    # Create enhanced dashboard
    dashboard = BiometricDashboard(chimera)
    
    # Run both systems
    await asyncio.gather(
        chimera.run_embodied(),
        dashboard.run()
    )

if __name__ == "__main__":
    asyncio.run(main())
