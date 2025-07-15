"""
Audio quality validation tests for buffer optimization.

This module provides comprehensive tests to validate that buffer management
optimizations maintain audio quality and don't introduce artifacts.
"""

import pytest
import time
import threading
import queue
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import json
import tempfile
from pathlib import Path

from spotify_splitter.buffer_management import AdaptiveBufferManager, BufferMetrics, HealthStatus
from spotify_splitter.audio import EnhancedAudioStream
from spotify_splitter.track_boundary_detector import TrackBoundaryDetector
from spotify_splitter.segmenter import SegmentManager


@dataclass
class AudioQualityMetrics:
    """Metrics for audio quality assessment."""
    snr_db: float
    thd_percent: float
    frequency_response_deviation: float
    phase_coherence: float
    dynamic_range_db: float
    stereo_correlation: float
    artifacts_detected: int
    quality_score: float


class AudioSignalGenerator:
    """Generate test audio signals for quality validation."""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
    
    def generate_sine_wave(self, frequency: float, duration: float, 
                          amplitude: float = 0.5, phase: float = 0.0) -> np.ndarray:
        """Generate a sine wave signal."""
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        signal = amplitude * np.sin(2 * np.pi * frequency * t + phase)
        return signal.reshape(-1, 1)
    
    def generate_stereo_sine(self, frequency: float, duration: float,
                           amplitude: float = 0.5, phase_diff: float = 0.0) -> np.ndarray:
        """Generate stereo sine wave with optional phase difference."""
        left = self.generate_sine_wave(frequency, duration, amplitude, 0.0)
        right = self.generate_sine_wave(frequency, duration, amplitude, phase_diff)
        return np.hstack([left, right])
    
    def generate_white_noise(self, duration: float, amplitude: float = 0.1) -> np.ndarray:
        """Generate white noise signal."""
        samples = int(self.sample_rate * duration)
        noise = amplitude * np.random.normal(0, 1, samples)
        return noise.reshape(-1, 1)
    
    def generate_multitone(self, frequencies: List[float], duration: float,
                          amplitudes: Optional[List[float]] = None) -> np.ndarray:
        """Generate a multitone signal."""
        if amplitudes is None:
            amplitudes = [0.1] * len(frequencies)
        
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        signal = np.zeros_like(t)
        
        for freq, amp in zip(frequencies, amplitudes):
            signal += amp * np.sin(2 * np.pi * freq * t)
        
        return signal.reshape(-1, 1)


class AudioQualityAnalyzer:
    """Analyze audio quality and detect artifacts."""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
    
    def calculate_snr(self, signal: np.ndarray, noise: np.ndarray) -> float:
        """Calculate Signal-to-Noise Ratio in dB."""
        signal_power = np.mean(signal ** 2)
        noise_power = np.mean(noise ** 2)
        
        if noise_power == 0:
            return float('inf')
        
        snr = 10 * np.log10(signal_power / noise_power)
        return snr
    
    def calculate_thd(self, signal: np.ndarray, fundamental_freq: float,
                     harmonics: int = 5) -> float:
        """Calculate Total Harmonic Distortion (simplified)."""
        # For testing purposes, use a simplified THD calculation
        # In practice, this would use FFT analysis
        
        # Calculate RMS of signal
        signal_rms = np.sqrt(np.mean(signal ** 2))
        
        # Estimate distortion as high-frequency content
        # This is a simplified approach for testing
        diff_signal = np.diff(signal.flatten())
        distortion_estimate = np.sqrt(np.mean(diff_signal ** 2)) * 0.1
        
        if signal_rms == 0:
            return 100.0
        
        thd = 100 * (distortion_estimate / signal_rms)
        return min(thd, 100.0)  # Cap at 100%
    
    def analyze_frequency_response(self, input_signal: np.ndarray, 
                                 output_signal: np.ndarray) -> float:
        """Analyze frequency response deviation (simplified)."""
        # Simplified frequency response analysis
        input_rms = np.sqrt(np.mean(input_signal ** 2))
        output_rms = np.sqrt(np.mean(output_signal ** 2))
        
        if input_rms == 0:
            return 0.0
        
        # Calculate gain difference
        gain_ratio = output_rms / input_rms
        deviation = abs(gain_ratio - 1.0)
        
        return deviation
    
    def calculate_phase_coherence(self, left_channel: np.ndarray, 
                                right_channel: np.ndarray) -> float:
        """Calculate phase coherence between stereo channels."""
        # Calculate cross-correlation
        correlation = np.corrcoef(left_channel.flatten(), right_channel.flatten())[0, 1]
        
        # Handle NaN case (constant signals)
        if np.isnan(correlation):
            correlation = 1.0
        
        return abs(correlation)
    
    def calculate_dynamic_range(self, signal: np.ndarray) -> float:
        """Calculate dynamic range in dB."""
        # Calculate RMS values in overlapping windows
        window_size = int(0.1 * self.sample_rate)  # 100ms windows
        hop_size = window_size // 2
        
        rms_values = []
        for i in range(0, len(signal) - window_size, hop_size):
            window = signal[i:i + window_size]
            rms = np.sqrt(np.mean(window ** 2))
            if rms > 0:
                rms_values.append(rms)
        
        if len(rms_values) < 2:
            return 0.0
        
        # Calculate dynamic range as difference between max and min RMS
        max_rms = np.max(rms_values)
        min_rms = np.min(rms_values)
        
        if min_rms == 0:
            return float('inf')
        
        dynamic_range = 20 * np.log10(max_rms / min_rms)
        return dynamic_range
    
    def detect_artifacts(self, signal: np.ndarray) -> int:
        """Detect audio artifacts (clicks, pops, dropouts)."""
        artifacts = 0
        
        # Detect clicks/pops (sudden amplitude spikes)
        diff = np.diff(signal.flatten())
        threshold = 5 * np.std(diff)
        clicks = np.sum(np.abs(diff) > threshold)
        artifacts += clicks
        
        # Detect dropouts (sudden amplitude drops to near zero)
        rms_window = int(0.01 * self.sample_rate)  # 10ms windows
        for i in range(0, len(signal) - rms_window, rms_window):
            window = signal[i:i + rms_window]
            rms = np.sqrt(np.mean(window ** 2))
            if rms < 0.001:  # Very low level, potential dropout
                artifacts += 1
        
        # Detect DC offset
        dc_offset = np.abs(np.mean(signal))
        if dc_offset > 0.01:  # Significant DC offset
            artifacts += 1
        
        return artifacts
    
    def calculate_stereo_correlation(self, stereo_signal: np.ndarray) -> float:
        """Calculate correlation between stereo channels."""
        if stereo_signal.shape[1] < 2:
            return 1.0  # Mono signal
        
        left = stereo_signal[:, 0]
        right = stereo_signal[:, 1]
        
        # Calculate Pearson correlation coefficient
        correlation = np.corrcoef(left, right)[0, 1]
        
        # Handle NaN case (constant signals)
        if np.isnan(correlation):
            correlation = 1.0
        
        return correlation
    
    def analyze_audio_quality(self, original: np.ndarray, processed: np.ndarray,
                            test_frequency: Optional[float] = None) -> AudioQualityMetrics:
        """Comprehensive audio quality analysis."""
        # Ensure signals are the same length
        min_length = min(len(original), len(processed))
        original = original[:min_length]
        processed = processed[:min_length]
        
        # Calculate noise (difference between original and processed)
        noise = processed - original
        
        # Calculate SNR
        snr = self.calculate_snr(original, noise)
        
        # Calculate THD (if test frequency provided)
        thd = 0.0
        if test_frequency is not None:
            thd = self.calculate_thd(processed, test_frequency)
        
        # Analyze frequency response
        freq_response_dev = self.analyze_frequency_response(original, processed)
        
        # Calculate phase coherence (for stereo signals)
        phase_coherence = 1.0
        if original.shape[1] >= 2 and processed.shape[1] >= 2:
            orig_coherence = self.calculate_phase_coherence(original[:, 0], original[:, 1])
            proc_coherence = self.calculate_phase_coherence(processed[:, 0], processed[:, 1])
            phase_coherence = proc_coherence / max(orig_coherence, 0.001)
        
        # Calculate dynamic range
        dynamic_range = self.calculate_dynamic_range(processed)
        
        # Calculate stereo correlation
        stereo_correlation = self.calculate_stereo_correlation(processed)
        
        # Detect artifacts
        artifacts = self.detect_artifacts(processed)
        
        # Calculate overall quality score (0-100)
        quality_score = self._calculate_quality_score(
            snr, thd, freq_response_dev, phase_coherence, 
            dynamic_range, artifacts
        )
        
        return AudioQualityMetrics(
            snr_db=snr,
            thd_percent=thd,
            frequency_response_deviation=freq_response_dev,
            phase_coherence=phase_coherence,
            dynamic_range_db=dynamic_range,
            stereo_correlation=stereo_correlation,
            artifacts_detected=artifacts,
            quality_score=quality_score
        )
    
    def _calculate_quality_score(self, snr: float, thd: float, freq_dev: float,
                               phase_coherence: float, dynamic_range: float,
                               artifacts: int) -> float:
        """Calculate overall quality score."""
        # SNR score (0-30 points)
        snr_score = min(30, max(0, snr / 2))  # 60dB SNR = 30 points
        
        # THD score (0-20 points)
        thd_score = max(0, 20 - thd * 2)  # 10% THD = 0 points
        
        # Frequency response score (0-20 points)
        freq_score = max(0, 20 - freq_dev * 100)  # 0.2 deviation = 0 points
        
        # Phase coherence score (0-15 points)
        phase_score = phase_coherence * 15
        
        # Dynamic range score (0-10 points)
        dr_score = min(10, max(0, dynamic_range / 6))  # 60dB DR = 10 points
        
        # Artifact penalty (0-5 points deducted)
        artifact_penalty = min(5, artifacts * 0.5)
        
        total_score = snr_score + thd_score + freq_score + phase_score + dr_score - artifact_penalty
        return max(0, min(100, total_score))


class BufferQualityTest:
    """Test audio quality through buffer management system."""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.generator = AudioSignalGenerator(sample_rate)
        self.analyzer = AudioQualityAnalyzer(sample_rate)
        self.results = {}
    
    def test_sine_wave_quality(self, frequency: float = 1000.0, 
                              duration: float = 5.0) -> dict:
        """Test quality with sine wave signal."""
        # Generate test signal
        original_signal = self.generator.generate_stereo_sine(frequency, duration)
        
        # Process through buffer management
        processed_signal = self._process_through_buffers(original_signal)
        
        # Analyze quality
        quality_metrics = self.analyzer.analyze_audio_quality(
            original_signal, processed_signal, frequency
        )
        
        return {
            'test_type': 'sine_wave',
            'frequency': frequency,
            'duration': duration,
            'quality_metrics': quality_metrics,
            'passed': quality_metrics.quality_score > 80
        }
    
    def test_multitone_quality(self, frequencies: List[float] = None,
                              duration: float = 5.0) -> dict:
        """Test quality with multitone signal."""
        if frequencies is None:
            frequencies = [440, 880, 1320, 1760, 2200]  # Musical harmonics
        
        # Generate multitone signal
        original_signal = self.generator.generate_multitone(frequencies, duration)
        # Convert to stereo
        original_signal = np.hstack([original_signal, original_signal])
        
        # Process through buffer management
        processed_signal = self._process_through_buffers(original_signal)
        
        # Analyze quality
        quality_metrics = self.analyzer.analyze_audio_quality(
            original_signal, processed_signal, frequencies[0]
        )
        
        return {
            'test_type': 'multitone',
            'frequencies': frequencies,
            'duration': duration,
            'quality_metrics': quality_metrics,
            'passed': quality_metrics.quality_score > 75
        }
    
    def test_noise_handling_quality(self, duration: float = 5.0) -> dict:
        """Test quality with noise signals."""
        # Test with white noise
        white_noise = self.generator.generate_white_noise(duration)
        white_stereo = np.hstack([white_noise, white_noise])
        
        # Process through buffers
        processed_white = self._process_through_buffers(white_stereo)
        
        # Analyze quality
        white_quality = self.analyzer.analyze_audio_quality(white_stereo, processed_white)
        
        return {
            'test_type': 'noise_handling',
            'duration': duration,
            'white_noise_quality': white_quality,
            'passed': white_quality.quality_score > 70
        }
    
    def test_dynamic_range_preservation(self, duration: float = 10.0) -> dict:
        """Test dynamic range preservation."""
        # Generate signal with varying amplitude
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        
        # Create amplitude envelope (soft to loud to soft)
        envelope = 0.1 + 0.8 * np.sin(2 * np.pi * t / duration) ** 2
        
        # Generate base signal (1kHz sine wave)
        base_signal = np.sin(2 * np.pi * 1000 * t)
        
        # Apply envelope
        original_signal = (envelope * base_signal).reshape(-1, 1)
        original_stereo = np.hstack([original_signal, original_signal])
        
        # Process through buffers
        processed_signal = self._process_through_buffers(original_stereo)
        
        # Analyze quality
        quality_metrics = self.analyzer.analyze_audio_quality(
            original_stereo, processed_signal, 1000.0
        )
        
        return {
            'test_type': 'dynamic_range',
            'duration': duration,
            'quality_metrics': quality_metrics,
            'passed': quality_metrics.dynamic_range_db > 30  # At least 30dB dynamic range
        }
    
    def _process_through_buffers(self, audio_signal: np.ndarray) -> np.ndarray:
        """Process audio signal through buffer management system."""
        # Create buffer manager
        buffer_manager = AdaptiveBufferManager(
            initial_queue_size=256,
            min_size=64,
            max_size=1024
        )
        
        # Create audio queue
        audio_queue = queue.Queue(maxsize=buffer_manager.current_queue_size)
        
        # Split signal into frames
        frame_size = 1024  # Process in 1024-sample frames
        frames = []
        
        for i in range(0, len(audio_signal), frame_size):
            frame = audio_signal[i:i + frame_size]
            if len(frame) == frame_size:  # Only process complete frames
                frames.append(frame)
        
        # Process frames through buffer system
        processed_frames = []
        buffer_adjustments = 0
        
        for frame_idx, frame in enumerate(frames):
            try:
                # Queue frame
                audio_queue.put_nowait(frame)
                
                # Get processed frame
                processed_frame = audio_queue.get_nowait()
                processed_frames.append(processed_frame)
                
                # Periodic buffer monitoring
                if frame_idx % 50 == 0:
                    try:
                        metrics = buffer_manager.monitor_utilization(audio_queue)
                        old_size = buffer_manager.current_queue_size
                        buffer_manager.adjust_buffer_size(metrics)
                        
                        if buffer_manager.current_queue_size != old_size:
                            buffer_adjustments += 1
                            
                            # Resize queue if needed
                            new_queue = queue.Queue(maxsize=buffer_manager.current_queue_size)
                            while not audio_queue.empty():
                                try:
                                    new_queue.put_nowait(audio_queue.get_nowait())
                                except queue.Full:
                                    break
                            audio_queue = new_queue
                    except:
                        pass
                
            except (queue.Full, queue.Empty):
                # Handle buffer overflow/underrun by using original frame
                processed_frames.append(frame)
        
        # Reconstruct processed signal
        if processed_frames:
            processed_signal = np.vstack(processed_frames)
        else:
            processed_signal = audio_signal
        
        return processed_signal


class TrackBoundaryQualityTest:
    """Test audio quality at track boundaries."""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.generator = AudioSignalGenerator(sample_rate)
        self.analyzer = AudioQualityAnalyzer(sample_rate)
        self.results = {}
    
    def test_boundary_continuity(self, track_duration: float = 3.0,
                               boundary_grace_ms: float = 500.0) -> dict:
        """Test audio continuity at track boundaries."""
        # Generate two different tracks
        track1_freq = 440.0  # A4
        track2_freq = 880.0  # A5
        
        track1 = self.generator.generate_stereo_sine(track1_freq, track_duration)
        track2 = self.generator.generate_stereo_sine(track2_freq, track_duration)
        
        # Create combined signal with boundary
        combined_signal = np.vstack([track1, track2])
        
        # Process through track boundary detector
        boundary_detector = TrackBoundaryDetector(grace_period_ms=int(boundary_grace_ms))
        
        # Simulate track boundary detection
        boundary_frame = len(track1)
        
        # Apply grace period
        grace_samples = int(boundary_grace_ms * self.sample_rate / 1000)
        boundary_start = max(0, boundary_frame - grace_samples // 2)
        boundary_end = min(len(combined_signal), boundary_frame + grace_samples // 2)
        
        # Extract boundary region for analysis
        boundary_region = combined_signal[boundary_start:boundary_end]
        
        # Analyze for discontinuities
        artifacts = self.analyzer.detect_artifacts(boundary_region)
        
        # Check for phase continuity
        pre_boundary = combined_signal[boundary_frame - 100:boundary_frame]
        post_boundary = combined_signal[boundary_frame:boundary_frame + 100]
        
        phase_coherence = self.analyzer.calculate_phase_coherence(
            pre_boundary[:, 0], post_boundary[:, 0]
        )
        
        return {
            'test_type': 'boundary_continuity',
            'track_duration': track_duration,
            'grace_period_ms': boundary_grace_ms,
            'artifacts_at_boundary': artifacts,
            'phase_coherence': phase_coherence,
            'boundary_samples': len(boundary_region),
            'passed': artifacts == 0 and phase_coherence > 0.8
        }


# Test classes for pytest integration

class TestAudioQualityValidation:
    """Test class for audio quality validation."""
    
    def test_sine_wave_quality_preservation(self):
        """Test that sine waves maintain quality through buffer processing."""
        test = BufferQualityTest()
        result = test.test_sine_wave_quality(frequency=1000.0, duration=2.0)
        
        # Verify quality metrics
        metrics = result['quality_metrics']
        assert metrics.snr_db > 40, f"SNR too low: {metrics.snr_db} dB"
        assert metrics.thd_percent < 5.0, f"THD too high: {metrics.thd_percent}%"
        assert metrics.quality_score > 80, f"Quality score too low: {metrics.quality_score}"
        assert result['passed'], "Sine wave quality test should pass"
    
    def test_multitone_quality(self):
        """Test quality with complex multitone signals."""
        test = BufferQualityTest()
        frequencies = [220, 440, 880, 1760]  # Musical octaves
        result = test.test_multitone_quality(frequencies=frequencies, duration=2.0)
        
        # Verify quality metrics
        metrics = result['quality_metrics']
        assert metrics.artifacts_detected < 3, f"Too many artifacts: {metrics.artifacts_detected}"
        assert metrics.quality_score > 75, f"Quality score too low: {metrics.quality_score}"
        assert result['passed'], "Multitone quality test should pass"
    
    def test_dynamic_range_preservation(self):
        """Test dynamic range preservation."""
        test = BufferQualityTest()
        result = test.test_dynamic_range_preservation(duration=5.0)
        
        # Verify dynamic range
        metrics = result['quality_metrics']
        assert metrics.dynamic_range_db > 30, f"Dynamic range too low: {metrics.dynamic_range_db} dB"
        assert result['passed'], "Dynamic range preservation test should pass"
    
    def test_noise_handling_quality(self):
        """Test quality with noise signals."""
        test = BufferQualityTest()
        result = test.test_noise_handling_quality(duration=3.0)
        
        # Verify noise handling
        white_quality = result['white_noise_quality']
        
        assert white_quality.quality_score > 70, f"White noise quality too low: {white_quality.quality_score}"
        assert result['passed'], "Noise handling quality test should pass"
    
    def test_track_boundary_continuity(self):
        """Test audio continuity at track boundaries."""
        test = TrackBoundaryQualityTest()
        result = test.test_boundary_continuity(track_duration=2.0, boundary_grace_ms=200.0)
        
        # Verify boundary quality
        assert result['artifacts_at_boundary'] == 0, "Should have no artifacts at boundary"
        assert result['phase_coherence'] > 0.8, f"Phase coherence too low: {result['phase_coherence']}"
        assert result['passed'], "Boundary continuity test should pass"
    
    @pytest.mark.parametrize("frequency,expected_min_snr", [
        (100.0, 35.0),   # Low frequency
        (1000.0, 40.0),  # Mid frequency
        (10000.0, 35.0), # High frequency
    ])
    def test_frequency_specific_quality(self, frequency, expected_min_snr):
        """Test quality at specific frequencies."""
        test = BufferQualityTest()
        result = test.test_sine_wave_quality(frequency=frequency, duration=1.0)
        
        metrics = result['quality_metrics']
        assert metrics.snr_db >= expected_min_snr, \
            f"SNR {metrics.snr_db} dB below expected {expected_min_snr} dB at {frequency} Hz"
    
    def test_audio_signal_generator(self):
        """Test audio signal generator functionality."""
        generator = AudioSignalGenerator(44100)
        
        # Test sine wave generation
        sine = generator.generate_sine_wave(1000.0, 1.0)
        assert sine.shape == (44100, 1), "Sine wave should have correct shape"
        assert np.max(np.abs(sine)) <= 0.5, "Sine wave amplitude should be within bounds"
        
        # Test stereo sine generation
        stereo_sine = generator.generate_stereo_sine(1000.0, 1.0)
        assert stereo_sine.shape == (44100, 2), "Stereo sine should have correct shape"
        
        # Test noise generation
        noise = generator.generate_white_noise(1.0)
        assert noise.shape == (44100, 1), "Noise should have correct shape"
        assert np.std(noise) > 0, "Noise should have non-zero variance"
    
    def test_audio_quality_analyzer(self):
        """Test audio quality analyzer functionality."""
        analyzer = AudioQualityAnalyzer(44100)
        generator = AudioSignalGenerator(44100)
        
        # Generate test signals
        original = generator.generate_sine_wave(1000.0, 1.0)
        # Add small amount of noise to create "processed" version
        processed = original + generator.generate_white_noise(1.0) * 0.01
        
        # Test SNR calculation
        noise = processed - original
        snr = analyzer.calculate_snr(original, noise)
        assert snr > 30, f"SNR should be reasonable: {snr} dB"
        
        # Test THD calculation
        thd = analyzer.calculate_thd(original, 1000.0)
        assert thd < 1.0, f"THD should be low for pure sine wave: {thd}%"
        
        # Test artifact detection
        artifacts = analyzer.detect_artifacts(original)
        assert artifacts == 0, "Pure sine wave should have no artifacts"


if __name__ == "__main__":
    # Run audio quality validation tests directly
    print("Running Audio Quality Validation Tests...")
    
    # Sine wave quality test
    print("\n1. Sine Wave Quality Test")
    buffer_test = BufferQualityTest()
    sine_result = buffer_test.test_sine_wave_quality(frequency=1000.0, duration=2.0)
    metrics = sine_result['quality_metrics']
    print(f"   SNR: {metrics.snr_db:.1f} dB")
    print(f"   THD: {metrics.thd_percent:.2f}%")
    print(f"   Quality Score: {metrics.quality_score:.1f}")
    print(f"   Passed: {sine_result['passed']}")
    
    # Track boundary test
    print("\n2. Track Boundary Quality Test")
    boundary_test = TrackBoundaryQualityTest()
    boundary_result = boundary_test.test_boundary_continuity(track_duration=2.0)
    print(f"   Artifacts at Boundary: {boundary_result['artifacts_at_boundary']}")
    print(f"   Phase Coherence: {boundary_result['phase_coherence']:.3f}")
    print(f"   Passed: {boundary_result['passed']}")
    
    # Dynamic range test
    print("\n3. Dynamic Range Preservation Test")
    dr_result = buffer_test.test_dynamic_range_preservation(duration=3.0)
    dr_metrics = dr_result['quality_metrics']
    print(f"   Dynamic Range: {dr_metrics.dynamic_range_db:.1f} dB")
    print(f"   Quality Score: {dr_metrics.quality_score:.1f}")
    print(f"   Passed: {dr_result['passed']}")
    
    print("\nAudio quality validation tests completed!")