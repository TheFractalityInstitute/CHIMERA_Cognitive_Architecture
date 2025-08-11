#!/usr/bin/env python3
"""
CHIMERA Lite - Optimized for Termux/Android
Uses only numpy and basic Python
"""
import asyncio
import json
import subprocess
import time
import numpy as np
from collections import deque
from datetime import datetime
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel

console = Console()

class CHIMERALite:
    def __init__(self):
        self.sensor_data = {}
        self.activity_history = deque(maxlen=100)
        self.running = True
        self.start_time = datetime.now()
        
        # Activity detection parameters
        self.activity_state = "unknown"
        self.movement_buffer = deque(maxlen=50)
        
        console.print("[bold green]🧠 CHIMERA Lite Initializing...[/bold green]")
        
    async def sensor_loop(self):
        """Main sensor reading loop"""
        while self.running:
            # Read accelerometer
            try:
                result = subprocess.run(
                    ['termux-sensor', '-s', 'accelerometer', '-n', '1'],
                    capture_output=True, text=True, timeout=0.5
                )
                data = json.loads(result.stdout)
                accel = data['accelerometer']['values']
                
                # Calculate magnitude using numpy
                accel_array = np.array(accel)
                magnitude = np.linalg.norm(accel_array)
                
                self.sensor_data['accelerometer'] = {
                    'x': accel[0],
                    'y': accel[1],
                    'z': accel[2],
                    'magnitude': magnitude
                }
                
                # Update movement buffer
                self.movement_buffer.append(magnitude)
                
                # Detect activity
                if len(self.movement_buffer) >= 30:
                    movement_array = np.array(self.movement_buffer)
                    std_dev = np.std(movement_array)
                    mean_mag = np.mean(movement_array)
                    
                    if std_dev < 0.5 and mean_mag < 10.5:
                        self.activity_state = "📱 Still"
                    elif std_dev < 2.0:
                        self.activity_state = "🚶 Walking"
                    elif std_dev < 5.0:
                        self.activity_state = "🏃 Running"
                    else:
                        self.activity_state = "🚗 Vehicle"
                        
            except Exception as e:
                console.print(f"[red]Accelerometer error: {e}[/red]")
                
            # Read light sensor
            try:
                result = subprocess.run(
                    ['termux-sensor', '-s', 'light', '-n', '1'],
                    capture_output=True, text=True, timeout=0.5
                )
                data = json.loads(result.stdout)
                lux = data['light']['values'][0]
                
                self.sensor_data['light'] = {
                    'lux': lux,
                    'environment': self._classify_light(lux)
                }
            except:
                pass
                
            await asyncio.sleep(0.1)  # 10Hz
            
    def _classify_light(self, lux):
        """Classify light environment"""
        if lux < 10:
            return "🌙 Dark"
        elif lux < 100:
            return "💡 Indoor"
        elif lux < 1000:
            return "☁️ Cloudy"
        else:
            return "☀️ Sunny"
            
    def create_dashboard(self):
        """Create rich dashboard"""
        # Main table
        table = Table(title="CHIMERA Lite Dashboard")
        table.add_column("Sensor", style="cyan", width=20)
        table.add_column("Value", style="green")
        table.add_column("Status", style="yellow")
        
        # Add accelerometer data
        if 'accelerometer' in self.sensor_data:
            accel = self.sensor_data['accelerometer']
            table.add_row(
                "Accelerometer",
                f"Mag: {accel['magnitude']:.2f}",
                self.activity_state
            )
            table.add_row(
                "Movement",
                f"X:{accel['x']:.2f} Y:{accel['y']:.2f} Z:{accel['z']:.2f}",
                ""
            )
        
        # Add light data
        if 'light' in self.sensor_data:
            light = self.sensor_data['light']
            table.add_row(
                "Light Sensor",
                f"{light['lux']:.1f} lux",
                light['environment']
            )
        
        # Uptime
        uptime = (datetime.now() - self.start_time).total_seconds()
        table.add_row(
            "Uptime",
            f"{uptime:.0f} seconds",
            "🟢 Active"
        )
        
        return Panel(table, border_style="bold green")
        
    async def display_loop(self):
        """Live display update loop"""
        with Live(self.create_dashboard(), refresh_per_second=4) as live:
            while self.running:
                live.update(self.create_dashboard())
                await asyncio.sleep(0.25)
                
    async def run(self):
        """Main run method"""
        console.print("[bold green]🚀 CHIMERA Lite Starting...[/bold green]")
        console.print("Press Ctrl+C to stop\n")
        
        try:
            await asyncio.gather(
                self.sensor_loop(),
                self.display_loop()
            )
        except KeyboardInterrupt:
            console.print("\n[bold yellow]👋 CHIMERA shutting down gracefully...[/bold yellow]")
            self.running = False

if __name__ == "__main__":
    chimera = CHIMERALite()
    asyncio.run(chimera.run())
