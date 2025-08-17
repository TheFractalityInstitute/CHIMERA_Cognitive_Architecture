"""Enhanced dashboard with biometric visualization"""
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn
from rich.columns import Columns
from rich.text import Text

class BiometricDashboard(EmbodiedDashboard):
    """Dashboard enhanced with biometric displays"""
    
    def get_biometric_panel(self):
        """Display Garmin biometric data"""
        bio_data = self.chimera.garmin_hub.biometric_cache
        
        table = Table(title="❤️ Biometrics (Garmin)", expand=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_column("Status", style="yellow")
        
        # Heart Rate with zones
        if 'heart_rate' in bio_data:
            hr = bio_data['heart_rate']['bpm']
            zone = self._get_hr_zone(hr)
            table.add_row(
                "Heart Rate",
                f"{hr:.0f} bpm",
                zone
            )
        
        # HRV with interpretation
        if 'hrv' in bio_data:
            hrv = bio_data['hrv']['ms']
            hrv_status = "Good" if hrv > 50 else "Low"
            table.add_row(
                "HRV",
                f"{hrv:.0f} ms",
                hrv_status
            )
        
        # Stress Level
        if 'stress' in bio_data:
            stress = bio_data['stress']['level']
            stress_bar = self._create_bar(stress, 100, "red")
            table.add_row(
                "Stress",
                stress_bar,
                f"{stress}/100"
            )
        
        # Body Battery
        if 'body_battery' in bio_data:
            battery = bio_data['body_battery']['level']
            battery_bar = self._create_bar(battery, 100, "green")
            table.add_row(
                "Body Battery",
                battery_bar,
                f"{battery}%"
            )
        
        # SpO2
        if 'spo2' in bio_data:
            spo2 = bio_data['spo2']['percent']
            spo2_status = "Normal" if spo2 > 95 else "Low"
            table.add_row(
                "Blood Oxygen",
                f"{spo2:.0f}%",
                spo2_status
            )
        
        return Panel(table, border_style="red")
    
    def get_s24_panel(self):
        """Display S24 Ultra specific sensors"""
        table = Table(title="📱 S24 Ultra Sensors", expand=True)
        table.add_column("Sensor", style="cyan")
        table.add_column("Reading", style="magenta")
        
        sensors = self.chimera.s24_hub.advanced_sensors
        
        # Barometer with weather
        if 'barometer' in sensors:
            pressure = sensors['barometer']['pressure']
            altitude = sensors['barometer']['altitude']
            table.add_row(
                "Barometer",
                f"{pressure:.1f} hPa | {altitude:.0f}m"
            )
        
        # Magnetometer for compass
        if 'magnetometer' in sensors:
            mag = sensors['magnetometer']
            heading = self._calculate_heading(mag)
            table.add_row(
                "Compass",
                f"{heading}° {self._heading_to_cardinal(heading)}"
            )
        
        # Temperature sensors
        if 'temperature' in sensors:
            temp = sensors['temperature']
            table.add_row(
                "Temperature",
                f"Ambient: {temp['ambient']}°C | Device: {temp['device']}°C"
            )
        
        return Panel(table, border_style="blue")
    
    def get_integrated_state_panel(self):
        """Show CHIMERA's integrated understanding"""
        physio = self.chimera.agents.get('physiological')
        context = self.chimera.agents.get('context')
        
        if not physio or not context:
            return Panel("Initializing integrated awareness...", border_style="yellow")
        
        # Create narrative description
        narrative = Text()
        narrative.append("CHIMERA's Understanding:\n\n", style="bold")
        
        # Physiological narrative
        if physio.current_state:
            state_color = {
                'optimal': 'green',
                'baseline': 'white',
                'stressed': 'red',
                'exhausted': 'orange',
                'fatigued': 'yellow'
            }.get(physio.current_state, 'white')
            
            narrative.append(f"You are ", style="white")
            narrative.append(f"{physio.current_state}", style=state_color)
            narrative.append(f" (Energy: {physio.energy_level:.0%}, Stress: {physio.stress_level:.0%})\n", style="white")
        
        # Context narrative
        if context.context_model:
            ctx = context.context_model
            narrative.append(f"You are {ctx['activity']} in a {ctx['environmental']} environment\n", style="white")
        
        # Predictions
        narrative.append("\nPredictions:\n", style="bold")
        predictions = self._generate_predictions(physio, context)
        for pred in predictions:
            narrative.append(f"• {pred}\n", style="italic")
        
        return Panel(narrative, title="🧠 Integrated Awareness", border_style="magenta")
    
    def _generate_predictions(self, physio, context):
        """Generate predictive insights based on state"""
        predictions = []
        
        if physio.stress_level > 0.7:
            predictions.append("You may benefit from a break soon")
        
        if physio.energy_level < 0.3:
            predictions.append("Rest recommended within next hour")
        
        if context.context_model['activity'] == 'exercising':
            predictions.append(f"Recovery time needed: {physio.stress_level * 30:.0f} minutes")
        
        hour = datetime.now().hour
        if hour >= 22 and physio.energy_level < 0.5:
            predictions.append("Optimal sleep window approaching")
        
        return predictions if predictions else ["Gathering more data for predictions..."]
    
    def _create_bar(self, value, max_value, color):
        """Create a progress bar visualization"""
        filled = int((value / max_value) * 20)
        bar = "█" * filled + "░" * (20 - filled)
        return Text(bar, style=color)
    
    def _get_hr_zone(self, hr):
        """Calculate heart rate zone"""
        if hr < 60: return "Resting"
        elif hr < 100: return "Zone 1"
        elif hr < 120: return "Zone 2"
        elif hr < 140: return "Zone 3"
        elif hr < 160: return "Zone 4"
        else: return "Zone 5"
