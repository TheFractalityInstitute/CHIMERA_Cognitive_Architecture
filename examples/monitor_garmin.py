#!/usr/bin/env python3
"""
Monitor Garmin data in real-time
"""
from garmin_integration import RobustGarminIntegration
import time
from datetime import datetime

def main():
    garmin = RobustGarminIntegration()
    
    print("💓 Garmin Real-time Monitor")
    print("Press Ctrl+C to stop")
    print("=" * 50)
    
    while True:
        try:
            metrics = garmin.get_current_metrics()
            
            # Clear line and display
            print(f"\r⌚ {datetime.now().strftime('%H:%M:%S')} | ", end="")
            print(f"❤️ {metrics.get('heart_rate', 0)} bpm | ", end="")
            print(f"😰 Stress: {metrics.get('stress', 0)}% | ", end="")
            print(f"🔋 Battery: {metrics.get('body_battery', 0)}% | ", end="")
            print(f"📈 SpO2: {metrics.get('spo2', 0)}%     ", end="")
            
            time.sleep(2)
            
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped.")
            break

if __name__ == "__main__":
    main()