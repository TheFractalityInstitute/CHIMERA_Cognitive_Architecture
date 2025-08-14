"""
CHIMERA Sensory Integration Module v2.0
Neurobiologically-grounded multi-modal sensor fusion with uncertainty quantification
"""

import numpy as np
import asyncio
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import scipy.signal as signal
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler
import warnings

# ============= Sensory Modality Definitions =============

class SensoryModality(Enum):
    """Distinct sensory channels with biological analogs"""
    PROPRIOCEPTIVE = "proprioception"  # IMU, accelerometer (body position)
    EXTEROCEPTIVE = "exteroception"    # WiFi, Bluetooth (external environment)
    INTEROCEPTIVE = "interoception"    # Battery, temperature (internal state)
    PHOTORECEPTIVE = "photoreception"  # Light sensor (vision analog)
    BARORECEPTIVE = "baroreception"    # Pressure (altitude/weather)
    MAGNETORECEPTIVE = "magnetoreception"  # Compass (navigation)
    NOCICEPTIVE = "nociception"        # Error signals (pain analog)

@dataclass
class SensoryStream:
    """Individual sensor stream with uncertainty"""
    modality: SensoryModality
    raw_value: Any
    processed_value: Optional[np.ndarray] = None
    confidence: float = 1.0
    noise_level: float = 0.0
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
class CrossFrequencyCoupling:
    """Implements neurobiologically-realistic binding through CFC"""
    
    def __init__(self):
        # Frequency bands (Hz) based on neuroscience
        self.bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 100),
            'high_gamma': (100, 200)
        }
        
        # Cross-frequency coupling preferences
        self.coupling_matrix = {
            'theta-gamma': 0.8,  # Strong for memory encoding
            'alpha-beta': 0.6,   # Attention modulation
            'delta-gamma': 0.4,  # Sleep/wake transitions
            'theta-high_gamma': 0.7  # Conscious binding
        }
        
    def compute_phase_amplitude_coupling(self, 
                                        low_freq_signal: np.ndarray,
                                        high_freq_signal: np.ndarray,
                                        fs: float = 1000) -> float:
        """
        Compute Phase-Amplitude Coupling (PAC) between frequency bands
        This is how the brain actually binds features
        """
        # Extract phase of low frequency
        analytic_low = signal.hilbert(low_freq_signal)
        phase_low = np.angle(analytic_low)
        
        # Extract amplitude of high frequency  
        analytic_high = signal.hilbert(high_freq_signal)
        amplitude_high = np.abs(analytic_high)
        
        # Compute PAC using Kullback-Leibler divergence
        n_bins = 18
        phase_bins = np.linspace(-np.pi, np.pi, n_bins + 1)
        
        # Bin the amplitudes by phase
        amplitude_by_phase = []
        for i in range(n_bins):
            mask = (phase_low >= phase_bins[i]) & (phase_low < phase_bins[i + 1])
            if np.any(mask):
                amplitude_by_phase.append(np.mean(amplitude_high[mask]))
            else:
                amplitude_by_phase.append(0)
                
        # Normalize to probability distribution
        amplitude_dist = np.array(amplitude_by_phase)
        amplitude_dist = amplitude_dist / np.sum(amplitude_dist)
        
        # Uniform distribution for comparison
        uniform_dist = np.ones(n_bins) / n_bins
        
        # KL divergence as PAC measure
        pac = entropy(amplitude_dist, uniform_dist)
        
        return pac

class KalmanSensorFusion:
    """
    Optimal sensor fusion with uncertainty tracking
    Unlike simple averaging, this properly handles sensor reliability
    """
    
    def __init__(self, state_dim: int = 6):
        self.state_dim = state_dim
        
        # State: [x, y, z, vx, vy, vz] or similar
        self.x = np.zeros(state_dim)  # State estimate
        self.P = np.eye(state_dim)    # Covariance matrix
        
        # Process model (constant velocity for now)
        self.F = np.eye(state_dim)
        self.Q = np.eye(state_dim) * 0.01  # Process noise
        
        # Measurement models for different sensors
        self.sensor_models = {}
        
    def predict(self, dt: float):
        """Prediction step"""
        # Update state transition for time step
        if self.state_dim == 6:
            self.F[0:3, 3:6] = np.eye(3) * dt
            
        # Predict state
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        
    def update(self, measurement: np.ndarray, 
               H: np.ndarray, 
               R: np.ndarray):
        """Update step with measurement"""
        # Innovation
        y = measurement - H @ self.x
        S = H @ self.P @ H.T + R
        
        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update state
        self.x = self.x + K @ y
        self.P = (np.eye(self.state_dim) - K @ H) @ self.P
        
        # Return innovation for anomaly detection
        return y, S

class EnhancedSensoryIntegration(AdaptiveSensoryIntegration):
    """
    Sensory Integration with empirically-grounded phase locking
    Based on Guth et al., 2025 findings
    """
    
    def __init__(self):
        super().__init__()
        
        # Use the validated 1-10 Hz range from the paper
        self.theta_range = (1, 10)  # Hz
        
        # Regional phase-locking strengths from paper
        self.regional_ppc = {
            'parahippocampal': 0.85,  # Strongest
            'entorhinal': 0.70,
            'amygdala': 0.65,
            'temporal_pole': 0.60,
            'hippocampus': 0.55  # Weakest (but most flexible)
        }
        
        # SPEAR model implementation (Separate Phases of Encoding And Retrieval)
        self.encoding_phase_preference = -np.pi/2  # Trough
        self.retrieval_phase_preference = 0  # Peak
        
    def compute_phase_locking_strength(self, 
                                      theta_power: float,
                                      aperiodic_slope: float,
                                      has_oscillation: bool) -> float:
        """
        Compute phase-locking strength based on paper's findings
        """
        base_ppc = 0.5
        
        # Stronger during high theta power (Fig 3 in paper)
        if theta_power > np.median(self.power_history):
            base_ppc *= 1.3
            
        # Stronger during steep aperiodic slopes (Fig 6C)
        if aperiodic_slope < -1.5:  # Steeper = more negative
            base_ppc *= 1.2
            
        # Stronger during clear oscillations (Fig 6G)
        if has_oscillation:
            base_ppc *= 1.4
            
        return np.clip(base_ppc, 0, 1)
    
    def separate_encoding_retrieval_phases(self, 
                                          memory_state: str,
                                          neuron_id: str) -> float:
        """
        Implement SPEAR model - separate phases for encoding vs retrieval
        ~9% of neurons show significant phase shifts (paper finding)
        """
        if neuron_id in self.phase_shifting_neurons:  # 9% of neurons
            if memory_state == 'encoding':
                return self.encoding_phase_preference
            elif memory_state == 'retrieval':
                return self.retrieval_phase_preference
        else:
            # 91% maintain stable phase preference
            return self.default_phase_preference

class AdaptiveSensoryIntegration:
    """
    The actual Sensory Integration Module with biological realism
    """
    
    def __init__(self):
        self.modality_processors = {}
        self.fusion_engine = KalmanSensorFusion()
        self.binding_engine = CrossFrequencyCoupling()
        
        # Adaptive parameters
        self.reliability_history = defaultdict(lambda: deque(maxlen=100))
        self.modality_weights = defaultdict(lambda: 1.0)
        
        # Sensory memory buffers (like thalamic relay)
        self.sensory_buffers = defaultdict(lambda: deque(maxlen=50))
        
        # Attention mechanism
        self.attention_weights = np.ones(len(SensoryModality))
        self.salience_threshold = 0.7
        
        # Predictive coding
        self.predictions = {}
        self.prediction_errors = deque(maxlen=100)
        
    async def process_raw_sensor(self, 
                                sensor_data: Dict[str, Any],
                                timestamp: float) -> SensoryStream:
        """
        Process raw sensor data into standardized stream
        """
        # Identify modality
        modality = self._classify_sensor(sensor_data)
        
        # Extract and validate
        raw_value = sensor_data.get('value')
        if raw_value is None:
            return None
            
        # Compute confidence based on:
        # 1. Sensor noise characteristics
        # 2. Historical reliability
        # 3. Consistency with predictions
        
        noise_level = self._estimate_noise(raw_value, modality)
        reliability = np.mean(list(self.reliability_history[modality])[-10:] or [1.0])
        
        # Predictive coding: how surprising is this measurement?
        if modality in self.predictions:
            prediction_error = np.abs(raw_value - self.predictions[modality])
            surprise = 1.0 / (1.0 + np.exp(-prediction_error))
        else:
            surprise = 0.5
            
        confidence = reliability * (1.0 - noise_level) * (1.0 - surprise)
        
        # Create stream
        stream = SensoryStream(
            modality=modality,
            raw_value=raw_value,
            confidence=confidence,
            noise_level=noise_level,
            timestamp=timestamp,
            metadata=sensor_data
        )
        
        # Process based on modality
        stream.processed_value = await self._modality_specific_processing(stream)
        
        return stream
        
    async def _modality_specific_processing(self, 
                                          stream: SensoryStream) -> np.ndarray:
        """
        Modality-specific preprocessing (like V1, A1, S1 in cortex)
        """
        if stream.modality == SensoryModality.PROPRIOCEPTIVE:
            # IMU data: extract orientation, remove gravity
            if isinstance(stream.raw_value, dict):
                accel = np.array(stream.raw_value.get('accelerometer', [0,0,0]))
                gyro = np.array(stream.raw_value.get('gyroscope', [0,0,0]))
                
                # Remove gravity component (simple high-pass filter)
                gravity = np.array([0, 0, 9.81])
                linear_accel = accel - gravity
                
                return np.concatenate([linear_accel, gyro])
                
        elif stream.modality == SensoryModality.EXTEROCEPTIVE:
            # WiFi/Bluetooth: compute spatial gradient
            if isinstance(stream.raw_value, list):
                rssi_values = [ap.get('rssi', -100) for ap in stream.raw_value]
                if rssi_values:
                    # Spatial encoding based on signal strength
                    gradient = np.gradient(rssi_values) if len(rssi_values) > 1 else [0]
                    return np.array(gradient)
                    
        elif stream.modality == SensoryModality.PHOTORECEPTIVE:
            # Light: adapt to logarithmic response (like retina)
            if isinstance(stream.raw_value, (int, float)):
                # Weber-Fechner law
                return np.array([np.log1p(stream.raw_value)])
                
        # Default: return as array
        return np.array([stream.raw_value])
        
    def _classify_sensor(self, sensor_data: Dict) -> SensoryModality:
        """Classify sensor into biological modality"""
        sensor_type = sensor_data.get('type', '').lower()
        
        mapping = {
            'accelerometer': SensoryModality.PROPRIOCEPTIVE,
            'gyroscope': SensoryModality.PROPRIOCEPTIVE,
            'wifi': SensoryModality.EXTEROCEPTIVE,
            'bluetooth': SensoryModality.EXTEROCEPTIVE,
            'light': SensoryModality.PHOTORECEPTIVE,
            'pressure': SensoryModality.BARORECEPTIVE,
            'compass': SensoryModality.MAGNETORECEPTIVE,
            'battery': SensoryModality.INTEROCEPTIVE,
            'temperature': SensoryModality.INTEROCEPTIVE,
            'error': SensoryModality.NOCICEPTIVE
        }
        
        return mapping.get(sensor_type, SensoryModality.EXTEROCEPTIVE)
        
    def _estimate_noise(self, value: Any, modality: SensoryModality) -> float:
        """Estimate measurement noise level"""
        buffer = self.sensory_buffers[modality]
        
        if len(buffer) < 3:
            return 0.1  # Default low noise
            
        recent_values = list(buffer)[-10:]
        
        try:
            # Compute coefficient of variation
            if isinstance(value, (int, float)):
                values = [float(v.raw_value) for v in recent_values 
                         if isinstance(v.raw_value, (int, float))]
                if values:
                    cv = np.std(values) / (np.mean(values) + 1e-6)
                    return np.clip(cv, 0, 1)
        except:
            pass
            
        return 0.2  # Default moderate noise
        
    async def integrate_streams(self, 
                               streams: List[SensoryStream],
                               timestamp: float) -> Dict[str, Any]:
        """
        Main integration: fuse all sensory streams into unified percept
        """
        if not streams:
            return None
            
        # Group by modality
        modality_groups = defaultdict(list)
        for stream in streams:
            modality_groups[stream.modality].append(stream)
            self.sensory_buffers[stream.modality].append(stream)
            
        # Weighted fusion within modalities
        fused_modalities = {}
        for modality, group in modality_groups.items():
            if group:
                # Weight by confidence
                weights = np.array([s.confidence for s in group])
                weights = weights / np.sum(weights)
                
                # Fuse processed values
                values = [s.processed_value for s in group if s.processed_value is not None]
                if values:
                    # Weighted average
                    fused = np.average(values, weights=weights[:len(values)], axis=0)
                    fused_modalities[modality] = {
                        'value': fused,
                        'confidence': np.mean([s.confidence for s in group]),
                        'streams': len(group)
                    }
                    
        # Cross-modal binding via phase coupling
        binding_strength = await self._compute_binding(fused_modalities)
        
        # Update Kalman filter for state estimation
        state_estimate = self._update_state_estimate(fused_modalities, timestamp)
        
        # Compute salience map (what deserves attention?)
        salience = self._compute_salience(fused_modalities, state_estimate)
        
        # Update predictions for next iteration
        self._update_predictions(fused_modalities)
        
        # Generate integrated percept
        integrated = {
            'timestamp': timestamp,
            'modalities': fused_modalities,
            'state': state_estimate,
            'binding_strength': binding_strength,
            'salience': salience,
            'attention_focus': self._select_attention_focus(salience),
            'confidence': self._compute_overall_confidence(fused_modalities),
            'anomaly_score': self._detect_anomalies(fused_modalities)
        }
        
        return integrated
        
    async def _compute_binding(self, modalities: Dict) -> float:
        """
        Compute cross-modal binding strength using phase coupling
        """
        if len(modalities) < 2:
            return 0.0
            
        # Simulate neural oscillations for each modality
        oscillations = {}
        for modality, data in modalities.items():
            # Generate oscillation based on modality characteristics
            freq = self._get_modality_frequency(modality)
            t = np.linspace(0, 1, 100)
            phase = np.random.random() * 2 * np.pi
            oscillations[modality] = np.sin(2 * np.pi * freq * t + phase)
            
        # Compute pairwise phase coupling
        couplings = []
        modality_list = list(oscillations.keys())
        
        for i in range(len(modality_list)):
            for j in range(i + 1, len(modality_list)):
                pac = self.binding_engine.compute_phase_amplitude_coupling(
                    oscillations[modality_list[i]],
                    oscillations[modality_list[j]]
                )
                couplings.append(pac)
                
        return np.mean(couplings) if couplings else 0.0
        
    def _get_modality_frequency(self, modality: SensoryModality) -> float:
        """Get characteristic frequency for modality (Hz)"""
        # Based on neuroscience literature
        frequencies = {
            SensoryModality.PROPRIOCEPTIVE: 10,  # Alpha
            SensoryModality.EXTEROCEPTIVE: 40,   # Gamma  
            SensoryModality.PHOTORECEPTIVE: 60,  # High gamma
            SensoryModality.INTEROCEPTIVE: 0.1,  # Slow waves
            SensoryModality.BARORECEPTIVE: 1,    # Delta
            SensoryModality.MAGNETORECEPTIVE: 8, # Theta
            SensoryModality.NOCICEPTIVE: 100     # Fast gamma
        }
        return frequencies.get(modality, 20)
        
    def _update_state_estimate(self, 
                              modalities: Dict,
                              timestamp: float) -> np.ndarray:
        """Update Kalman filter with new measurements"""
        # Predict step
        dt = 0.01  # 10ms default
        self.fusion_engine.predict(dt)
        
        # Update with each modality
        for modality, data in modalities.items():
            if 'value' in data:
                # Create measurement model for this modality
                H = self._get_measurement_matrix(modality)
                R = np.eye(len(data['value'])) * (1.0 - data['confidence'])
                
                # Update
                self.fusion_engine.update(data['value'], H, R)
                
        return self.fusion_engine.x
        
    def _get_measurement_matrix(self, modality: SensoryModality) -> np.ndarray:
        """Get observation matrix for modality"""
        # This maps state to expected measurements
        # Simplified example - should be customized
        if modality == SensoryModality.PROPRIOCEPTIVE:
            # Measures position and velocity
            H = np.array([[1, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 1]])
        else:
            # Default: observes partial state
            H = np.zeros((1, 6))
            H[0, 0] = 1
            
        return H
        
    def _compute_salience(self, 
                         modalities: Dict,
                         state: np.ndarray) -> Dict[str, float]:
        """
        Compute salience map - what deserves attention?
        Based on prediction error and information content
        """
        salience = {}
        
        for modality, data in modalities.items():
            # Information content (entropy)
            if 'value' in data:
                info = entropy(np.abs(data['value']) + 1e-6)
            else:
                info = 0
                
            # Prediction error if available
            if modality in self.predictions:
                error = np.mean(np.abs(data['value'] - self.predictions[modality]))
            else:
                error = 0
                
            # Confidence-weighted salience
            salience[modality.value] = (info + error) * data.get('confidence', 1.0)
            
        return salience
        
    def _select_attention_focus(self, salience: Dict[str, float]) -> Optional[str]:
        """Select what to attend to based on salience"""
        if not salience:
            return None
            
        # Find maximum salience above threshold
        max_salience = max(salience.values())
        if max_salience > self.salience_threshold:
            return max(salience, key=salience.get)
            
        return None
        
    def _compute_overall_confidence(self, modalities: Dict) -> float:
        """Compute overall confidence in sensory state"""
        if not modalities:
            return 0.0
            
        confidences = [data.get('confidence', 0) for data in modalities.values()]
        
        # Weighted by number of streams
        weights = [data.get('streams', 1) for data in modalities.values()]
        
        return np.average(confidences, weights=weights)
        
    def _detect_anomalies(self, modalities: Dict) -> float:
        """Detect anomalous sensory patterns"""
        anomaly_scores = []
        
        for modality, data in modalities.items():
            buffer = list(self.sensory_buffers[modality])[-20:]
            if len(buffer) > 5:
                # Simple anomaly: deviation from recent history
                recent_values = [s.processed_value for s in buffer[:-1] 
                               if s.processed_value is not None]
                if recent_values and 'value' in data:
                    mean = np.mean(recent_values, axis=0)
                    std = np.std(recent_values, axis=0) + 1e-6
                    z_score = np.abs((data['value'] - mean) / std)
                    anomaly_scores.append(np.mean(z_score))
                    
        return np.mean(anomaly_scores) if anomaly_scores else 0.0
        
    def _update_predictions(self, modalities: Dict):
        """Update predictions for next timestep (predictive coding)"""
        for modality, data in modalities.items():
            if 'value' in data:
                # Simple exponential smoothing for now
                alpha = 0.3
                if modality in self.predictions:
                    self.predictions[modality] = (alpha * data['value'] + 
                                                 (1 - alpha) * self.predictions[modality])
                else:
                    self.predictions[modality] = data['value']

# ============= Integration with CHIMERA Eidolon Architecture =============

class SensoryEidolon:
    """
    Sensory Processing Eidolon Module for CHIMERA Cube
    One of the 6 face modules, specialized for sensory integration
    """
    
    def __init__(self, name="Sensory", bus_interface=None):
        self.name = name
        self.role = "environmental_awareness"
        self.integrator = AdaptiveSensoryIntegration()
        self.bus = bus_interface
        
        # Module personality
        self.confidence_threshold = 0.6
        self.speaking_style = "observational"
        
        # Internal state
        self.current_percept = None
        self.attention_focus = None
        
    async def deliberate(self, topic: str) -> Dict[str, Any]:
        """Form opinion on topic from sensory perspective"""
        
        # Analyze topic relevance to current sensory state
        relevance = self._assess_relevance(topic)
        
        opinion = f"From sensory perspective: "
        
        if self.current_percept:
            if self.attention_focus:
                opinion += f"Currently focused on {self.attention_focus}. "
                
            confidence = self.current_percept.get('confidence', 0)
            if confidence > self.confidence_threshold:
                opinion += f"Environmental state appears stable. "
            else:
                opinion += f"Detecting uncertainty in sensory input. "
                
            anomaly = self.current_percept.get('anomaly_score', 0)
            if anomaly > 0.5:
                opinion += "Anomalous patterns detected - recommend caution."
        else:
            opinion += "No recent sensory data available."
            
        return {
            'module': self.name,
            'opinion': opinion,
            'confidence': confidence if self.current_percept else 0.1,
            'reasoning': self._explain_reasoning(),
            'relevance': relevance
        }
        
    def _assess_relevance(self, topic: str) -> float:
        """Assess how relevant topic is to sensory domain"""
        sensory_keywords = ['environment', 'detect', 'sense', 'perceive', 
                          'observe', 'monitor', 'aware', 'attention']
        
        topic_lower = topic.lower()
        relevance = sum(1 for keyword in sensory_keywords 
                       if keyword in topic_lower) / len(sensory_keywords)
        
        return relevance
        
    def _explain_reasoning(self) -> str:
        """Explain sensory-based reasoning"""
        if not self.current_percept:
            return "No sensory data to base reasoning on"
            
        reasoning = []
        
        # Explain based on active modalities
        if 'modalities' in self.current_percept:
            active = list(self.current_percept['modalities'].keys())
            reasoning.append(f"Monitoring {len(active)} sensory channels")
            
        # Explain binding
        binding = self.current_percept.get('binding_strength', 0)
        if binding > 0.5:
            reasoning.append("Strong cross-modal coherence detected")
        elif binding < 0.2:
            reasoning.append("Weak sensory binding - possible interference")
            
        return "; ".join(reasoning)
        
    async def process_sensors(self, sensor_data: List[Dict], timestamp: float):
        """Main processing loop"""
        # Convert to streams
        streams = []
        for data in sensor_data:
            stream = await self.integrator.process_raw_sensor(data, timestamp)
            if stream:
                streams.append(stream)
                
        # Integrate
        if streams:
            self.current_percept = await self.integrator.integrate_streams(
                streams, timestamp
            )
            
            # Update attention
            if self.current_percept:
                self.attention_focus = self.current_percept.get('attention_focus')
                
            # Publish to bus if connected
            if self.bus and self.current_percept:
                await self._publish_percept()
                
    async def _publish_percept(self):
        """Publish integrated percept to message bus"""
        if self.bus:
            message = {
                'source': self.name,
                'type': 'percept',
                'timestamp': self.current_percept['timestamp'],
                'data': {
                    'confidence': self.current_percept['confidence'],
                    'attention': self.attention_focus,
                    'anomaly': self.current_percept['anomaly_score'],
                    'state': self.current_percept['state'].tolist()
                }
            }
            await self.bus.publish(message)
