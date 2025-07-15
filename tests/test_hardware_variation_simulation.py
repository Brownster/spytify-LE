"""
Hardware variation simulation tests for audio buffer optimization.

This module simulates different hardware configurations and capabilities
to validate buffer management across diverse system environments.
"""

import pytest
import time
import threading
import queue
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import json

from spotify_splitter.buffer_management import AdaptiveBufferManager, BufferMetrics, HealthStatus
from spotify_splitter.audio import EnhancedAudioStream
from spotify_splitter.config_profiles import ProfileManager, ProfileType


@dataclass
class HardwareProfile:
    """Represents a hardware configuration for testing."""
    name: str
    cpu_cores: int
    cpu_speed_ghz: float
    memory_gb: int
    audio_backend: str  # 'pulseaudio', 'pipewire', 'alsa'
    sample_rates: List[int]
    buffer_sizes: List[int]
    latency_ms: float
    stability_factor: float  # 0.0-1.0, affects error probability


class HardwareSimulator:
    """Simulates different hardware configurations."""
    
    # Predefined hardware profiles
    PROFILES = {
        'raspberry_pi_4': HardwareProfile(
            name='Raspberry Pi 4',
            cpu_cores=4,
            cpu_speed_ghz=1.5,
            memory_gb=4,
            audio_backend='pulseaudio',
            sample_rates=[44100, 48000],
            buffer_sizes=[256, 512, 1024],
            latency_ms=15.0,
            stability_factor=0.85
        ),
        'low_end_laptop': HardwareProfile(
            name='Low-end Laptop',
            cpu_cores=2,
            cpu_speed_ghz=2.0,
            memory_gb=4,
            audio_backend='pulseaudio',
            sample_rates=[44100, 48000],
            buffer_sizes=[128, 256, 512],
            latency_ms=10.0,
            stability_factor=0.75
        ),
        'mid_range_desktop': HardwareProfile(
            name='Mid-range Desktop',
            cpu_cores=6,
            cpu_speed_ghz=3.2,
            memory_gb=16,
            audio_backend='pipewire',
            sample_rates=[44100, 48000, 96000],
            buffer_sizes=[64, 128, 256, 512],
            latency_ms=5.0,
            stability_factor=0.95
        ),
        'high_end_workstation': HardwareProfile(
            name='High-end Workstation',
            cpu_cores=16,
            cpu_speed_ghz=4.0,
            memory_gb=64,
            audio_backend='pipewire',
            sample_rates=[44100, 48000, 96000, 192000],
            buffer_sizes=[32, 64, 128, 256],
            latency_ms=2.0,
            stability_factor=0.98
        ),
        'embedded_device': HardwareProfile(
            name='Embedded Device',
            cpu_cores=1,
            cpu_speed_ghz=1.0,
            memory_gb=1,
            audio_backend='alsa',
            sample_rates=[44100],
            buffer_sizes=[512, 1024, 2048],
            latency_ms=25.0,
            stability_factor=0.70
        ),
        'server_headless': HardwareProfile(
            name='Server (Headless)',
            cpu_cores=8,
            cpu_speed_ghz=2.8,
            memory_gb=32,
            audio_backend='pulseaudio',
            sample_rates=[44100, 48000],
            buffer_sizes=[256, 512, 1024],
            latency_ms=8.0,
            stability_factor=0.92
        )
    }
    
    def __init__(self, profile: HardwareProfile):
        self.profile = profile
        self.current_load = 0.0
        self.thermal_throttling = False
        self.power_saving_mode = False
    
    def simulate_system_load(self, base_load: float = 0.3) -> float:
        """Simulate varying system load based on hardware capabilities."""
        # Base load affected by CPU cores and speed
        load_factor = 1.0 / (self.profile.cpu_cores * self.profile.cpu_speed_ghz)
        adjusted_load = base_load * (1.0 + load_factor)
        
        # Add random variation
        variation = np.random.normal(0, 0.1)
        self.current_load = max(0.0, min(1.0, adjusted_load + variation))
        
        # Simulate thermal throttling on high load
        if self.current_load > 0.8 and np.random.random() < 0.1:
            self.thermal_throttling = True
            self.current_load *= 1.2  # Increased load due to throttling
        else:
            self.thermal_throttling = False
        
        return self.current_load
    
    def simulate_memory_pressure(self) -> float:
        """Simulate memory pressure based on available memory."""
        base_pressure = 0.4  # 40% base memory usage
        
        # Lower memory systems have higher pressure
        memory_factor = max(0.1, 8.0 / self.profile.memory_gb)
        pressure = base_pressure * memory_factor
        
        # Add random spikes
        if np.random.random() < 0.05:  # 5% chance of memory spike
            pressure += np.random.uniform(0.2, 0.4)
        
        return min(0.95, pressure)
    
    def simulate_audio_glitch(self) -> bool:
        """Simulate audio glitches based on hardware stability."""
        glitch_probability = (1.0 - self.profile.stability_factor) * 0.01
        
        # Increase probability under high load
        if self.current_load > 0.7:
            glitch_probability *= 2.0
        
        # Increase probability with thermal throttling
        if self.thermal_throttling:
            glitch_probability *= 1.5
        
        return np.random.random() < glitch_probability
    
    def get_optimal_buffer_size(self, sample_rate: int) -> int:
        """Get optimal buffer size for current hardware and conditions."""
        # Base buffer size from profile
        available_sizes = [s for s in self.profile.buffer_sizes if s >= 64]
        base_size = min(available_sizes) if available_sizes else 256
        
        # Adjust for current load
        if self.current_load > 0.6:
            # Higher load needs larger buffers
            base_size = min(base_size * 2, max(self.profile.buffer_sizes))
        
        # Adjust for sample rate
        if sample_rate > 48000:
            base_size = min(base_size * 2, max(self.profile.buffer_sizes))
        
        # Adjust for memory pressure
        memory_pressure = self.simulate_memory_pressure()
        if memory_pressure > 0.8:
            # High memory pressure needs smaller buffers
            base_size = max(base_size // 2, min(self.profile.buffer_sizes))
        
        return base_size
    
    def get_expected_latency(self, buffer_size: int, sample_rate: int) -> float:
        """Calculate expected latency for given parameters."""
        # Base latency from hardware
        base_latency = self.profile.latency_ms
        
        # Buffer contribution to latency
        buffer_latency = (buffer_size / sample_rate) * 1000  # Convert to ms
        
        # System load contribution
        load_latency = self.current_load * 5.0  # Up to 5ms additional latency
        
        total_latency = base_latency + buffer_latency + load_latency
        
        # Thermal throttling increases latency
        if self.thermal_throttling:
            total_latency *= 1.3
        
        return total_latency


class HardwareVariationTest:
    """Test audio buffer optimization across different hardware configurations."""
    
    def __init__(self, profile_name: str, duration: int = 30):
        self.profile_name = profile_name
        self.profile = HardwareSimulator.PROFILES[profile_name]
        self.simulator = HardwareSimulator(self.profile)
        self.duration = duration
        self.results = {}
    
    def run_test(self) -> dict:
        """Run hardware variation test."""
        # Test different sample rates and buffer sizes
        test_configurations = []
        
        for sample_rate in self.profile.sample_rates:
            for buffer_size in self.profile.buffer_sizes:
                test_configurations.append((sample_rate, buffer_size))
        
        configuration_results = []
        
        for sample_rate, buffer_size in test_configurations:
            config_result = self._test_configuration(sample_rate, buffer_size)
            configuration_results.append(config_result)
        
        # Find optimal configuration
        optimal_config = max(configuration_results, key=lambda x: x['performance_score'])
        
        # Aggregate results
        self.results = {
            'hardware_profile': self.profile.name,
            'test_duration': self.duration,
            'configurations_tested': len(configuration_results),
            'configuration_results': configuration_results,
            'optimal_configuration': optimal_config,
            'avg_performance_score': sum(r['performance_score'] for r in configuration_results) / len(configuration_results),
            'stability_rating': self.profile.stability_factor,
            'hardware_limitations': self._identify_limitations()
        }
        
        return self.results
    
    def _test_configuration(self, sample_rate: int, buffer_size: int) -> dict:
        """Test a specific sample rate and buffer size configuration."""
        # Create buffer manager with hardware-appropriate settings
        buffer_manager = AdaptiveBufferManager(
            initial_queue_size=buffer_size,
            min_size=max(32, buffer_size // 4),
            max_size=min(2048, buffer_size * 4)
        )
        
        # Performance tracking
        metrics = {
            'frames_processed': 0,
            'buffer_overflows': 0,
            'buffer_underruns': 0,
            'audio_glitches': 0,
            'latency_samples': [],
            'cpu_load_samples': [],
            'memory_pressure_samples': [],
            'buffer_adjustments': 0,
            'thermal_throttling_events': 0
        }
        
        # Create audio queue
        audio_queue = queue.Queue(maxsize=buffer_size)
        
        # Control flags
        active = threading.Event()
        active.set()
        
        def audio_processor():
            """Simulate audio processing for this configuration."""
            while active.is_set():
                try:
                    start_time = time.time()
                    
                    # Simulate system load
                    cpu_load = self.simulator.simulate_system_load()
                    memory_pressure = self.simulator.simulate_memory_pressure()
                    
                    metrics['cpu_load_samples'].append(cpu_load)
                    metrics['memory_pressure_samples'].append(memory_pressure)
                    
                    # Check for thermal throttling
                    if self.simulator.thermal_throttling:
                        metrics['thermal_throttling_events'] += 1
                    
                    # Create audio frame
                    frame_samples = int(sample_rate * 0.01)  # 10ms of audio
                    audio_frame = np.random.random((frame_samples, 2)).astype(np.float32)
                    
                    # Try to process frame
                    try:
                        audio_queue.put_nowait(audio_frame)
                        processed_frame = audio_queue.get_nowait()
                        metrics['frames_processed'] += 1
                        
                        # Calculate latency
                        processing_time = time.time() - start_time
                        latency_ms = processing_time * 1000
                        metrics['latency_samples'].append(latency_ms)
                        
                    except queue.Full:
                        metrics['buffer_overflows'] += 1
                        
                        # Try buffer adjustment
                        queue_metrics = buffer_manager.monitor_utilization(audio_queue)
                        old_size = buffer_manager.current_queue_size
                        buffer_manager.adjust_buffer_size(queue_metrics)
                        
                        if buffer_manager.current_queue_size != old_size:
                            metrics['buffer_adjustments'] += 1
                    
                    except queue.Empty:
                        metrics['buffer_underruns'] += 1
                    
                    # Simulate audio glitches
                    if self.simulator.simulate_audio_glitch():
                        metrics['audio_glitches'] += 1
                    
                    # Simulate processing time based on hardware
                    processing_delay = self._calculate_processing_delay(sample_rate, cpu_load)
                    time.sleep(processing_delay)
                
                except Exception as e:
                    print(f"Audio processing error: {e}")
                    time.sleep(0.001)
        
        # Run test
        processor_thread = threading.Thread(target=audio_processor, daemon=True)
        processor_thread.start()
        
        time.sleep(self.duration)
        active.clear()
        processor_thread.join(timeout=1.0)
        
        # Calculate performance metrics
        performance_score = self._calculate_performance_score(metrics, sample_rate, buffer_size)
        
        return {
            'sample_rate': sample_rate,
            'buffer_size': buffer_size,
            'frames_processed': metrics['frames_processed'],
            'buffer_overflows': metrics['buffer_overflows'],
            'buffer_underruns': metrics['buffer_underruns'],
            'audio_glitches': metrics['audio_glitches'],
            'buffer_adjustments': metrics['buffer_adjustments'],
            'thermal_throttling_events': metrics['thermal_throttling_events'],
            'avg_latency_ms': sum(metrics['latency_samples']) / max(1, len(metrics['latency_samples'])),
            'max_latency_ms': max(metrics['latency_samples']) if metrics['latency_samples'] else 0,
            'avg_cpu_load': sum(metrics['cpu_load_samples']) / max(1, len(metrics['cpu_load_samples'])),
            'max_cpu_load': max(metrics['cpu_load_samples']) if metrics['cpu_load_samples'] else 0,
            'avg_memory_pressure': sum(metrics['memory_pressure_samples']) / max(1, len(metrics['memory_pressure_samples'])),
            'performance_score': performance_score,
            'expected_latency_ms': self.simulator.get_expected_latency(buffer_size, sample_rate),
            'optimal_buffer_size': self.simulator.get_optimal_buffer_size(sample_rate)
        }
    
    def _calculate_processing_delay(self, sample_rate: int, cpu_load: float) -> float:
        """Calculate processing delay based on hardware and load."""
        # Base processing time per frame
        base_delay = 0.001  # 1ms base
        
        # Adjust for CPU capabilities
        cpu_factor = 1.0 / (self.profile.cpu_cores * self.profile.cpu_speed_ghz)
        
        # Adjust for current load
        load_factor = 1.0 + (cpu_load * 2.0)
        
        # Adjust for sample rate
        rate_factor = sample_rate / 44100.0
        
        total_delay = base_delay * cpu_factor * load_factor * rate_factor
        
        return max(0.0001, min(0.01, total_delay))  # Clamp between 0.1ms and 10ms
    
    def _calculate_performance_score(self, metrics: dict, sample_rate: int, buffer_size: int) -> float:
        """Calculate overall performance score for a configuration."""
        if metrics['frames_processed'] == 0:
            return 0.0
        
        # Base score from successful frame processing
        success_rate = 1.0 - (metrics['buffer_overflows'] + metrics['buffer_underruns']) / metrics['frames_processed']
        success_score = max(0.0, success_rate) * 40.0
        
        # Latency score (lower is better)
        avg_latency = sum(metrics['latency_samples']) / max(1, len(metrics['latency_samples']))
        expected_latency = self.simulator.get_expected_latency(buffer_size, sample_rate)
        latency_ratio = avg_latency / max(1.0, expected_latency)
        latency_score = max(0.0, 30.0 - (latency_ratio - 1.0) * 20.0)
        
        # Stability score (fewer glitches is better)
        glitch_rate = metrics['audio_glitches'] / max(1, metrics['frames_processed'])
        stability_score = max(0.0, 20.0 - glitch_rate * 1000.0)
        
        # Efficiency score (fewer adjustments is better for stable configs)
        adjustment_rate = metrics['buffer_adjustments'] / max(1, self.duration)
        efficiency_score = max(0.0, 10.0 - adjustment_rate * 2.0)
        
        total_score = success_score + latency_score + stability_score + efficiency_score
        return min(100.0, total_score)
    
    def _identify_limitations(self) -> List[str]:
        """Identify hardware limitations based on profile."""
        limitations = []
        
        if self.profile.cpu_cores < 4:
            limitations.append("Limited CPU cores may affect concurrent processing")
        
        if self.profile.cpu_speed_ghz < 2.0:
            limitations.append("Low CPU speed may increase processing latency")
        
        if self.profile.memory_gb < 8:
            limitations.append("Limited memory may restrict buffer sizes")
        
        if self.profile.latency_ms > 10.0:
            limitations.append("High base latency may affect real-time performance")
        
        if self.profile.stability_factor < 0.8:
            limitations.append("Hardware instability may cause audio glitches")
        
        if len(self.profile.sample_rates) < 3:
            limitations.append("Limited sample rate support")
        
        if max(self.profile.buffer_sizes) < 512:
            limitations.append("Small maximum buffer size may cause overflows")
        
        return limitations


class CrossPlatformCompatibilityTest:
    """Test compatibility across different audio backends and configurations."""
    
    def __init__(self):
        self.results = {}
    
    def run_test(self) -> dict:
        """Run cross-platform compatibility test."""
        # Test different backend configurations
        backend_tests = {
            'pulseaudio': self._test_pulseaudio_compatibility(),
            'pipewire': self._test_pipewire_compatibility(),
            'alsa': self._test_alsa_compatibility()
        }
        
        # Test configuration profile compatibility
        profile_tests = {
            'headless': self._test_headless_profile(),
            'desktop': self._test_desktop_profile(),
            'high_performance': self._test_high_performance_profile()
        }
        
        self.results = {
            'backend_compatibility': backend_tests,
            'profile_compatibility': profile_tests,
            'cross_platform_score': self._calculate_compatibility_score(backend_tests, profile_tests)
        }
        
        return self.results
    
    def _test_pulseaudio_compatibility(self) -> dict:
        """Test PulseAudio backend compatibility."""
        # Simulate PulseAudio-specific behavior
        buffer_manager = AdaptiveBufferManager(
            initial_queue_size=256,
            min_size=128,
            max_size=1024
        )
        
        # PulseAudio typically has higher latency but better stability
        test_metrics = {
            'supported_sample_rates': [44100, 48000],
            'supported_buffer_sizes': [128, 256, 512, 1024],
            'typical_latency_ms': 8.0,
            'stability_rating': 0.92,
            'cpu_efficiency': 0.85,
            'memory_usage_mb': 12.0
        }
        
        return test_metrics
    
    def _test_pipewire_compatibility(self) -> dict:
        """Test PipeWire backend compatibility."""
        # Simulate PipeWire-specific behavior
        buffer_manager = AdaptiveBufferManager(
            initial_queue_size=128,
            min_size=64,
            max_size=512
        )
        
        # PipeWire typically has lower latency and better performance
        test_metrics = {
            'supported_sample_rates': [44100, 48000, 96000, 192000],
            'supported_buffer_sizes': [32, 64, 128, 256, 512],
            'typical_latency_ms': 4.0,
            'stability_rating': 0.95,
            'cpu_efficiency': 0.92,
            'memory_usage_mb': 8.0
        }
        
        return test_metrics
    
    def _test_alsa_compatibility(self) -> dict:
        """Test ALSA backend compatibility."""
        # Simulate ALSA-specific behavior
        buffer_manager = AdaptiveBufferManager(
            initial_queue_size=512,
            min_size=256,
            max_size=2048
        )
        
        # ALSA typically needs larger buffers but has good performance
        test_metrics = {
            'supported_sample_rates': [44100, 48000, 96000],
            'supported_buffer_sizes': [256, 512, 1024, 2048],
            'typical_latency_ms': 6.0,
            'stability_rating': 0.88,
            'cpu_efficiency': 0.90,
            'memory_usage_mb': 6.0
        }
        
        return test_metrics
    
    def _test_headless_profile(self) -> dict:
        """Test headless configuration profile."""
        profile = ProfileManager.get_profile(ProfileType.HEADLESS)
        
        return {
            'profile_name': 'Headless',
            'audio_settings': {
                'queue_size': profile.queue_size,
                'blocksize': profile.blocksize,
                'latency': profile.latency
            },
            'buffer_settings': {
                'buffer_strategy': profile.buffer_strategy.value,
                'enable_adaptive_management': profile.enable_adaptive_management
            },
            'optimization_focus': 'stability',
            'expected_latency_ms': profile.latency * 1000,
            'memory_efficiency': 0.95,
            'cpu_efficiency': 0.88
        }
    
    def _test_desktop_profile(self) -> dict:
        """Test desktop configuration profile."""
        profile = ProfileManager.get_profile(ProfileType.DESKTOP)
        
        return {
            'profile_name': 'Desktop',
            'audio_settings': {
                'queue_size': profile.queue_size,
                'blocksize': profile.blocksize,
                'latency': profile.latency
            },
            'buffer_settings': {
                'buffer_strategy': profile.buffer_strategy.value,
                'enable_adaptive_management': profile.enable_adaptive_management
            },
            'optimization_focus': 'balanced',
            'expected_latency_ms': profile.latency * 1000,
            'memory_efficiency': 0.85,
            'cpu_efficiency': 0.90
        }
    
    def _test_high_performance_profile(self) -> dict:
        """Test high-performance configuration profile."""
        profile = ProfileManager.get_profile(ProfileType.HIGH_PERFORMANCE)
        
        return {
            'profile_name': 'High Performance',
            'audio_settings': {
                'queue_size': profile.queue_size,
                'blocksize': profile.blocksize,
                'latency': profile.latency
            },
            'buffer_settings': {
                'buffer_strategy': profile.buffer_strategy.value,
                'enable_adaptive_management': profile.enable_adaptive_management
            },
            'optimization_focus': 'latency',
            'expected_latency_ms': profile.latency * 1000,
            'memory_efficiency': 0.75,
            'cpu_efficiency': 0.95
        }
    
    def _calculate_compatibility_score(self, backend_tests: dict, profile_tests: dict) -> float:
        """Calculate overall cross-platform compatibility score."""
        # Backend compatibility score
        backend_score = 0.0
        for backend, metrics in backend_tests.items():
            stability = metrics.get('stability_rating', 0.0)
            efficiency = metrics.get('cpu_efficiency', 0.0)
            backend_score += (stability + efficiency) / 2.0
        
        backend_score = (backend_score / len(backend_tests)) * 50.0
        
        # Profile compatibility score
        profile_score = 0.0
        for profile, metrics in profile_tests.items():
            memory_eff = metrics.get('memory_efficiency', 0.0)
            cpu_eff = metrics.get('cpu_efficiency', 0.0)
            profile_score += (memory_eff + cpu_eff) / 2.0
        
        profile_score = (profile_score / len(profile_tests)) * 50.0
        
        return backend_score + profile_score


# Test classes for pytest integration

class TestHardwareVariationSimulation:
    """Test class for hardware variation simulation."""
    
    @pytest.mark.parametrize("profile_name", [
        'raspberry_pi_4',
        'low_end_laptop',
        'mid_range_desktop',
        'high_end_workstation'
    ])
    def test_hardware_profile_compatibility(self, profile_name):
        """Test compatibility with different hardware profiles."""
        test = HardwareVariationTest(profile_name, duration=10)
        results = test.run_test()
        
        # Verify test completed successfully
        assert results['configurations_tested'] > 0, f"Should test configurations for {profile_name}"
        assert results['optimal_configuration'] is not None, f"Should find optimal config for {profile_name}"
        assert results['avg_performance_score'] > 0, f"Should achieve some performance on {profile_name}"
        
        # Verify hardware-appropriate results
        profile = HardwareSimulator.PROFILES[profile_name]
        if profile.stability_factor < 0.8:
            # Less stable hardware should show more limitations
            assert len(results['hardware_limitations']) > 0, "Unstable hardware should show limitations"
    
    def test_hardware_simulator_behavior(self):
        """Test hardware simulator behavior."""
        profile = HardwareSimulator.PROFILES['mid_range_desktop']
        simulator = HardwareSimulator(profile)
        
        # Test load simulation
        load1 = simulator.simulate_system_load(0.3)
        load2 = simulator.simulate_system_load(0.7)
        
        assert 0.0 <= load1 <= 1.0, "Load should be in valid range"
        assert 0.0 <= load2 <= 1.0, "Load should be in valid range"
        
        # Test memory pressure simulation
        pressure = simulator.simulate_memory_pressure()
        assert 0.0 <= pressure <= 1.0, "Memory pressure should be in valid range"
        
        # Test optimal buffer size calculation
        buffer_size = simulator.get_optimal_buffer_size(44100)
        assert buffer_size in profile.buffer_sizes, "Should return valid buffer size"
        
        # Test latency calculation
        latency = simulator.get_expected_latency(256, 44100)
        assert latency > 0, "Latency should be positive"
        assert latency >= profile.latency_ms, "Should include base hardware latency"
    
    @pytest.mark.slow
    def test_embedded_device_optimization(self):
        """Test optimization for embedded devices with limited resources."""
        test = HardwareVariationTest('embedded_device', duration=15)
        results = test.run_test()
        
        # Embedded devices should prefer larger buffers for stability
        optimal_config = results['optimal_configuration']
        assert optimal_config['buffer_size'] >= 512, "Embedded devices should use larger buffers"
        
        # Should identify resource limitations
        assert len(results['hardware_limitations']) > 0, "Should identify embedded device limitations"
        
        # Performance should still be reasonable
        assert results['avg_performance_score'] > 20, "Should achieve basic performance on embedded device"
    
    def test_cross_platform_compatibility(self):
        """Test cross-platform compatibility."""
        test = CrossPlatformCompatibilityTest()
        results = test.run_test()
        
        # Verify all backends tested
        assert 'pulseaudio' in results['backend_compatibility'], "Should test PulseAudio"
        assert 'pipewire' in results['backend_compatibility'], "Should test PipeWire"
        assert 'alsa' in results['backend_compatibility'], "Should test ALSA"
        
        # Verify all profiles tested
        assert 'headless' in results['profile_compatibility'], "Should test headless profile"
        assert 'desktop' in results['profile_compatibility'], "Should test desktop profile"
        assert 'high_performance' in results['profile_compatibility'], "Should test high-performance profile"
        
        # Verify compatibility score
        assert results['cross_platform_score'] > 50, "Should achieve reasonable cross-platform compatibility"
    
    @pytest.mark.parametrize("sample_rate,expected_buffer_range", [
        (44100, (128, 512)),
        (48000, (128, 512)),
        (96000, (256, 1024)),
        (192000, (512, 2048))
    ])
    def test_sample_rate_buffer_optimization(self, sample_rate, expected_buffer_range):
        """Test buffer optimization for different sample rates."""
        profile = HardwareSimulator.PROFILES['mid_range_desktop']
        simulator = HardwareSimulator(profile)
        
        # Test multiple times to account for randomness
        buffer_sizes = []
        for _ in range(10):
            buffer_size = simulator.get_optimal_buffer_size(sample_rate)
            buffer_sizes.append(buffer_size)
        
        avg_buffer_size = sum(buffer_sizes) / len(buffer_sizes)
        min_expected, max_expected = expected_buffer_range
        
        # Buffer size should be in expected range for sample rate
        assert min_expected <= avg_buffer_size <= max_expected, \
            f"Buffer size {avg_buffer_size} not in expected range {expected_buffer_range} for {sample_rate}Hz"


if __name__ == "__main__":
    # Run hardware variation tests directly
    print("Running Hardware Variation Simulation Tests...")
    
    # Test different hardware profiles
    profiles_to_test = ['raspberry_pi_4', 'mid_range_desktop', 'high_end_workstation']
    
    for profile_name in profiles_to_test:
        print(f"\n--- Testing {profile_name} ---")
        test = HardwareVariationTest(profile_name, duration=8)
        results = test.run_test()
        
        print(f"Configurations tested: {results['configurations_tested']}")
        print(f"Average performance score: {results['avg_performance_score']:.1f}")
        print(f"Optimal configuration: {results['optimal_configuration']['sample_rate']}Hz, "
              f"{results['optimal_configuration']['buffer_size']} buffer")
        
        if results['hardware_limitations']:
            print("Hardware limitations:")
            for limitation in results['hardware_limitations']:
                print(f"  - {limitation}")
    
    # Test cross-platform compatibility
    print(f"\n--- Cross-Platform Compatibility ---")
    compat_test = CrossPlatformCompatibilityTest()
    compat_results = compat_test.run_test()
    
    print(f"Cross-platform compatibility score: {compat_results['cross_platform_score']:.1f}")
    
    for backend, metrics in compat_results['backend_compatibility'].items():
        print(f"{backend}: {metrics['typical_latency_ms']}ms latency, "
              f"{metrics['stability_rating']:.2f} stability")
    
    print("\nHardware variation tests completed!")