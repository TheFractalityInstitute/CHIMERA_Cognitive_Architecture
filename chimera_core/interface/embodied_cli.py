"""Real-time dashboard for embodied CHIMERA"""
import asyncio
import os
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

class EmbodiedDashboard:
    """Live dashboard for CHIMERA's embodied experience"""
    
    def __init__(self, chimera):
        self.chimera = chimera
        self.console = Console()
        self.layout = Layout()
        
    def make_layout(self):
        """Create dashboard layout"""
        self.layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=7)
        )
        
        self.layout["body"].split_row(
            Layout(name="sensors"),
            Layout(name="cognition"),
        )
        
    def get_header(self):
        """Generate header with system status"""
        grid = Table.grid(expand=True)
        grid.add_column(justify="left")
        grid.add_column(justify="center")
        grid.add_column(justify="right")
        
        uptime = datetime.now() - self.chimera.start_time if hasattr(self.chimera, 'start_time') else 0
        
        grid.add_row(
            f"🧠 CHIMERA v0.9",
            f"⚡ Phase: {self.chimera.clock.phase:.3f}",
            f"⏱️ Uptime: {uptime}"
        )
        
        return Panel(grid, style="bold blue")
    
    def get_sensor_panel(self):
        """Show live sensor data"""
        sensor_data = self.chimera.sensor_hub.sensor_cache
        
        table = Table(title="📡 Sensory Input", expand=True)
        table.add_column("Sensor", style="cyan")
        table.add_column("Value", style="green")
        table.add_column("Status", style="yellow")
        
        # Accelerometer
        if 'accelerometer' in sensor_data:
            accel = sensor_data['accelerometer']
            mag = (accel['x']**2 + accel['y']**2 + accel['z']**2)**0.5
            table.add_row(
                "Accelerometer",
                f"Mag: {mag:.2f}",
                self.chimera.sensor_agents['accelerometer'].activity_state
            )
        
        # Light
        if 'light' in sensor_data:
            lux = sensor_data['light']['lux']
            table.add_row(
                "Light",
                f"{lux:.1f} lux",
                "Dark" if lux < 10 else "Bright"
            )
        
        # Proprioception
        proprio = self.chimera.sensor_agents.get('proprioception')
        if proprio:
            table.add_row(
                "Body Schema",
                proprio.relationship_to_user,
                f"Conf: {proprio._calculate_confidence():.1%}"
            )
        
        return Panel(table, border_style="green")
    
    def get_cognition_panel(self):
        """Show cognitive state"""
        table = Table(title="🎯 Cognitive State", expand=True)
        table.add_column("System", style="cyan")
        table.add_column("State", style="magenta")
        
        # Active agents
        table.add_row("Active Agents", str(len(self.chimera.agents)))
        
        # Message rate
        if hasattr(self.chimera, 'message_count'):
            table.add_row("Messages/sec", str(self.chimera.message_rate))
        
        # Current phase
        circadian = self.chimera.sensor_agents.get('circadian')
        if circadian:
            table.add_row("Circadian Phase", circadian.current_phase)
        
        # Last crystallization
        if hasattr(self.chimera, 'last_insight'):
            table.add_row("Last Insight", str(self.chimera.last_insight)[:30])
        
        return Panel(table, border_style="magenta")
    
    def get_footer(self):
        """Show activity log"""
        log_text = "📝 Activity Log:\n"
        
        # Get recent activity
        if hasattr(self.chimera, 'activity_log'):
            for entry in list(self.chimera.activity_log)[-5:]:
                log_text += f"{entry}\n"
        else:
            log_text += "Waiting for activity..."
        
        return Panel(log_text, title="Recent Events", border_style="yellow")
    
    async def run(self):
        """Run live dashboard"""
        self.make_layout()
        
        with Live(self.layout, refresh_per_second=10, console=self.console) as live:
            while True:
                self.layout["header"].update(self.get_header())
                self.layout["sensors"].update(self.get_sensor_panel())
                self.layout["cognition"].update(self.get_cognition_panel())
                self.layout["footer"].update(self.get_footer())
                
                await asyncio.sleep(0.1)
