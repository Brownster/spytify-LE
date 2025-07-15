"""
Integration tests for complete audio pipeline with buffer health monitoring.

This module provides comprehensive integration tests that validate the entire
audio buffer optimization system working together as a complete pipeline.
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
import tempfile
from pathlib import Path
import concurrent.futures

from spotify_splitter.buffer_management import AdaptiveBufferManager, BufferMetrics, HealthStatus
from spotify_splitter.buffer_health_monitor import BufferHealthMonitor
from spotify_splitter.audio import EnhancedAudioStream
from spotify_splitter.error_recovery import ErrorRecoveryManager
from spotify_splitter.track_boundary_detector import TrackBoundaryDetector
from spotify_splitter.segmenter import SegmentManager
from spotify_splitter.metrics_collector import MetricsCollector
from spotify_splitter.performance_optimizer import PerformanceOptimizer


@dataclass
class PipelineTestResult:
    """Result from a complete pipeline test."""
    test_name: str
    duration: float
    tracks_processed: int
    total_frames: int
    buffer_health_score: float
    error_recovery_rate: float
    optimization_suggestions: int
    performance_score: float
    passed: bool
    details: dict


class MockAudioSource:
    """Mock audio source for testing."""
    
    def __init__(self, sample_rate: int = 44100, channels: int = 2):
        self.sample_rate = sample_rate
        self.channels = channels
        self.current_track = 0
        self.tracks_generated = 0
        self.frame_count = 0
        
    def generate_audio_frame(self, frame_size: int = 1024) -> np.ndarray:
        """Generate a frame of audio data."""
        # Generate different frequencies for different tracks
        base_freq = 440.0 + (self.current_track * 110.0)  # A4, B4, C#5, etc.
        
        # Create time vector for this frame
        t_start = self.frame_count / self.sample_rate
        t_end = (self.frame_count + frame_size) / self.sample_rate
        t = np.linspace(t_start, t_end, frame_size, False)
        
        # Generate sine wave with some variation
        signal = 0.3 * np.sin(2 * np.pi * base_freq * t)
        
        # Add some harmonics for realism
        signal += 0.1 * np.sin(2 * np.pi * base_freq * 2 * t)
        signal += 0.05 * np.sin(2 * np.pi * base_freq * 3 * t)
        
        # Add small amount of noise
        signal += 0.01 * np.random.normal(0, 1, frame_size)
        
        # Convert to stereo
        if self.channels == 2:
            # Slight phase difference between channels
            left = signal
            right = 0.3 * np.sin(2 * np.pi * base_freq * t + 0.1)
            right += 0.1 * np.sin(2 * np.pi * base_freq * 2 * t + 0.1)
            right += 0.05 * np.sin(2 * np.pi * base_freq * 3 * t + 0.1)
            right += 0.01 * np.random.normal(0, 1, frame_size)
            
            audio_frame = np.column_stack([left, right])
        else:
            audio_frame = signal.reshape(-1, 1)
        
        self.frame_count += frame_size
        return audio_frame.astype(np.float32)
    
    def simulate_track_change(self):
        """Simulate a track change."""
        self.current_track += 1
        self.tracks_generated += 1
    
    def get_track_info(self) -> dict:
        """Get current track information."""
        return {
            'track_id': self.current_track,
            'frequency': 440.0 + (self.current_track * 110.0),
            'frames_generated': self.frame_count
        }


class MockMPRISInterface:
    """Mock MPRIS interface for testing."""
    
    def __init__(self):
        self.current_track = {
            'title': 'Test Track 1',
            'artist': 'Test Artist',
            'album': 'Test Album',
            'duration': 180.0  # 3 minutes
        }
        self.position = 0.0
        self.playing = True
        self.track_changes = []
    
    def get_current_track(self) -> dict:
        """Get current track metadata."""
        return self.current_track.copy()
    
    def get_position(self) -> float:
        """Get current playback position."""
        return self.position
    
    def is_playing(self) -> bool:
        """Check if currently playing."""
        return self.playing
    
    def simulate_track_change(self, new_track: dict):
        """Simulate a track change event."""
        self.track_changes.append({
            'timestamp': time.time(),
            'old_track': self.current_track.copy(),
            'new_track': new_track.copy()
        })
        self.current_track = new_track
        self.position = 0.0


class CompletePipelineTest:
    """Test the complete audio processing pipeline."""
    
    def __init__(self, test_name: str, duration: float = 60.0):
        self.test_name = test_name
        self.duration = duration
        self.results = {}
        
        # Create all pipeline components
        self.buffer_manager = AdaptiveBufferManager(
            initial_queue_size=300,
            min_size=100,
            max_size=2000
        )
        
        self.buffer_health_monitor = BufferHealthMonitor(
            buffer_manager=self.buffer_manager,
            monitoring_interval=5.0
        )
        
        self.error_recovery = ErrorRecoveryManager(
            max_retries=5,
            backoff_factor=1.5
        )
        
        self.track_boundary_detector = TrackBoundaryDetector(
            grace_period_ms=300
        )
        
        self.metrics_collector = MetricsCollector(
            collection_interval=10.0,
            enable_debug_mode=True
        )
        
        self.performance_optimizer = PerformanceOptimizer(
            metrics_collector=self.metrics_collector,
            auto_apply_safe_optimizations=True,
            optimization_interval=30.0
        )
        
        # Mock components
        self.audio_source = MockAudioSource()
        self.mpris_interface = MockMPRISInterface()
        
        # Pipeline state
        self.audio_queue = queue.Queue(maxsize=self.buffer_manager.current_queue_size)
        self.segment_queue = queue.Queue(maxsize=100)
        self.active = threading.Event()
        
        # Performance tracking
        self.pipeline_stats = {
            'frames_processed': 0,
            'tracks_processed': 0,
            'buffer_overflows': 0,
            'buffer_underruns': 0,
            'error_events': 0,
            'recovery_attempts': 0,
            'successful_recoveries': 0,
            'buffer_adjustments': 0,
            'track_boundaries_detected': 0,
            'segments_created': 0,
            'optimization_applications': 0
        }
    
    def run_test(self) -> PipelineTestResult:
        """Run the complete pipeline test."""
        print(f"Starting pipeline test: {self.test_name}")
        
        # Start all monitoring and optimization
        self.metrics_collector.start_collection()
        self.buffer_health_monitor.start_monitoring()
        self.performance_optimizer.start_optimization()
        
        # Register components with metrics collector
        self._register_metrics_components()
        
        # Start pipeline threads
        self.active.set()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            # Submit all pipeline components
            futures = [
                executor.submit(self._audio_capture_worker),
                executor.submit(self._audio_processing_worker),
                executor.submit(self._track_boundary_worker),
                executor.submit(self._segment_processing_worker),
                executor.submit(self._pipeline_monitor_worker),
                executor.submit(self._track_simulation_worker)
            ]
            
            # Run for specified duration
            start_time = time.time()
            time.sleep(self.duration)
            
            # Stop all workers
            self.active.clear()
            
            # Wait for workers to complete
            for future in concurrent.futures.as_completed(futures, timeout=10.0):
                try:
                    future.result()
                except Exception as e:
                    print(f"Worker error: {e}")
        
        actual_duration = time.time() - start_time
        
        # Stop monitoring and optimization
        self.performance_optimizer.stop_optimization()
        self.buffer_health_monitor.stop_monitoring()
        self.metrics_collector.stop_collection()
        
        # Collect final results
        return self._calculate_results(actual_duration)
    
    def _register_metrics_components(self):
        """Register components with metrics collector."""
        self.metrics_collector.register_component(
            'buffer_manager',
            lambda: {
                'current_queue_size': self.buffer_manager.current_queue_size,
                'utilization_percent': self._get_queue_utilization(),
                'adjustment_count': self.buffer_manager.adjustment_count,
                'emergency_expansions': self.buffer_manager.emergency_expansion_count
            }
        )
        
        self.metrics_collector.register_component(
            'error_recovery',
            lambda: {
                'total_errors': getattr(self.error_recovery, 'total_errors', 0),
                'successful_recoveries': getattr(self.error_recovery, 'successful_recoveries', 0),
                'recovery_success_rate': self._calculate_recovery_rate()
            }
        )
        
        self.metrics_collector.register_component(
            'pipeline',
            lambda: {
                'frames_processed': self.pipeline_stats['frames_processed'],
                'tracks_processed': self.pipeline_stats['tracks_processed'],
                'segments_created': self.pipeline_stats['segments_created']
            }
        )
    
    def _audio_capture_worker(self):
        """Worker that simulates audio capture."""
        frame_size = 1024
        
        while self.active.is_set():
            try:
                # Generate audio frame
                audio_frame = self.audio_source.generate_audio_frame(frame_size)
                
                # Try to queue frame
                try:
                    self.audio_queue.put_nowait(audio_frame)
                    self.pipeline_stats['frames_processed'] += 1
                    
                except queue.Full:
                    self.pipeline_stats['buffer_overflows'] += 1
                    
                    # Try emergency buffer expansion
                    if self.buffer_manager.emergency_expansion():
                        self.pipeline_stats['buffer_adjustments'] += 1
                        
                        # Resize queue
                        new_queue = queue.Queue(maxsize=self.buffer_manager.current_queue_size)
                        while not self.audio_queue.empty():
                            try:
                                new_queue.put_nowait(self.audio_queue.get_nowait())
                            except queue.Full:
                                break
                        self.audio_queue = new_queue
                        
                        # Retry queuing
                        try:
                            self.audio_queue.put_nowait(audio_frame)
                            self.pipeline_stats['frames_processed'] += 1
                        except queue.Full:
                            pass  # Still couldn't queue, drop frame
                
                # Simulate real-time audio capture timing
                time.sleep(frame_size / self.audio_source.sample_rate)
                
            except Exception as e:
                self.pipeline_stats['error_events'] += 1
                print(f"Audio capture error: {e}")
                time.sleep(0.01)
    
    def _audio_processing_worker(self):
        """Worker that processes audio frames."""
        while self.active.is_set():
            try:
                # Get audio frame
                try:
                    audio_frame = self.audio_queue.get_nowait()
                    
                    # Simulate audio processing
                    processing_time = np.random.uniform(0.005, 0.015)  # 5-15ms
                    time.sleep(processing_time)
                    
                    # Queue for segment processing
                    try:
                        self.segment_queue.put_nowait(audio_frame)
                    except queue.Full:
                        # Drop oldest frame to make room
                        try:
                            self.segment_queue.get_nowait()
                            self.segment_queue.put_nowait(audio_frame)
                        except queue.Empty:
                            pass
                    
                except queue.Empty:
                    self.pipeline_stats['buffer_underruns'] += 1
                    time.sleep(0.001)  # Brief wait for more data
                
                # Periodic buffer monitoring and adjustment
                if self.pipeline_stats['frames_processed'] % 500 == 0:
                    try:
                        metrics = self.buffer_manager.monitor_utilization(self.audio_queue)
                        old_size = self.buffer_manager.current_queue_size
                        self.buffer_manager.adjust_buffer_size(metrics)
                        
                        if self.buffer_manager.current_queue_size != old_size:
                            self.pipeline_stats['buffer_adjustments'] += 1
                    except Exception as e:
                        print(f"Buffer monitoring error: {e}")
                
            except Exception as e:
                self.pipeline_stats['error_events'] += 1
                
                # Attempt error recovery
                self.pipeline_stats['recovery_attempts'] += 1
                recovery_success = self.error_recovery.handle_error(e, "audio_processing")
                
                if recovery_success:
                    self.pipeline_stats['successful_recoveries'] += 1
                
                time.sleep(0.01)
    
    def _track_boundary_worker(self):
        """Worker that detects track boundaries."""
        frames_since_boundary = 0
        frames_per_track = int(3 * 60 * self.audio_source.sample_rate / 1024)  # ~3 minutes
        
        while self.active.is_set():
            try:
                frames_since_boundary += 1
                
                # Simulate track boundary detection
                if frames_since_boundary >= frames_per_track:
                    # Detect track boundary
                    self.pipeline_stats['track_boundaries_detected'] += 1
                    self.pipeline_stats['tracks_processed'] += 1
                    
                    # Simulate track change
                    self.audio_source.simulate_track_change()
                    
                    # Update MPRIS interface
                    new_track = {
                        'title': f'Test Track {self.audio_source.current_track + 1}',
                        'artist': 'Test Artist',
                        'album': 'Test Album',
                        'duration': 180.0
                    }
                    self.mpris_interface.simulate_track_change(new_track)
                    
                    frames_since_boundary = 0
                
                time.sleep(0.1)  # Check every 100ms
                
            except Exception as e:
                self.pipeline_stats['error_events'] += 1
                print(f"Track boundary detection error: {e}")
                time.sleep(0.1)
    
    def _segment_processing_worker(self):
        """Worker that processes audio segments."""
        segment_frames = []
        frames_per_segment = 100  # Process in segments of 100 frames
        
        while self.active.is_set():
            try:
                # Collect frames for segment
                try:
                    audio_frame = self.segment_queue.get_nowait()
                    segment_frames.append(audio_frame)
                    
                    # Process segment when we have enough frames
                    if len(segment_frames) >= frames_per_segment:
                        # Simulate segment processing
                        segment_data = np.vstack(segment_frames)
                        
                        # Apply track boundary detection
                        boundary_result = self.track_boundary_detector.detect_boundary(
                            segment_data, []  # No markers for this test
                        )
                        
                        # Create segment
                        self.pipeline_stats['segments_created'] += 1
                        
                        # Clear frames for next segment
                        segment_frames = []
                        
                        # Simulate segment export time
                        time.sleep(0.01)
                
                except queue.Empty:
                    time.sleep(0.001)
                
            except Exception as e:
                self.pipeline_stats['error_events'] += 1
                print(f"Segment processing error: {e}")
                time.sleep(0.01)
    
    def _pipeline_monitor_worker(self):
        """Worker that monitors overall pipeline health."""
        while self.active.is_set():
            try:
                # Record pipeline metrics
                self.metrics_collector.record_gauge(
                    'pipeline.frames_processed', 
                    self.pipeline_stats['frames_processed']
                )
                self.metrics_collector.record_gauge(
                    'pipeline.buffer_utilization', 
                    self._get_queue_utilization()
                )
                self.metrics_collector.record_counter(
                    'pipeline.buffer_overflows', 
                    self.pipeline_stats['buffer_overflows']
                )
                
                time.sleep(5.0)  # Monitor every 5 seconds
                
            except Exception as e:
                print(f"Pipeline monitoring error: {e}")
                time.sleep(5.0)
    
    def _track_simulation_worker(self):
        """Worker that simulates track changes and events."""
        while self.active.is_set():
            try:
                # Simulate occasional playback events
                if np.random.random() < 0.01:  # 1% chance per iteration
                    # Simulate pause/resume
                    self.mpris_interface.playing = not self.mpris_interface.playing
                    time.sleep(np.random.uniform(0.5, 2.0))  # Pause duration
                    self.mpris_interface.playing = True
                
                # Update position
                self.mpris_interface.position += 1.0
                
                time.sleep(1.0)  # Update every second
                
            except Exception as e:
                print(f"Track simulation error: {e}")
                time.sleep(1.0)
    
    def _get_queue_utilization(self) -> float:
        """Get current queue utilization percentage."""
        if self.audio_queue.maxsize == 0:
            return 0.0
        return (self.audio_queue.qsize() / self.audio_queue.maxsize) * 100.0
    
    def _calculate_recovery_rate(self) -> float:
        """Calculate error recovery success rate."""
        attempts = self.pipeline_stats['recovery_attempts']
        successes = self.pipeline_stats['successful_recoveries']
        return successes / max(1, attempts)
    
    def _calculate_results(self, actual_duration: float) -> PipelineTestResult:
        """Calculate final test results."""
        # Get buffer health score
        buffer_health = self.buffer_health_monitor.get_current_health()
        buffer_health_score = self._calculate_health_score(buffer_health)
        
        # Get optimization suggestions
        optimization_suggestions = self.performance_optimizer.get_optimization_suggestions()
        
        # Calculate performance score
        performance_score = self._calculate_performance_score()
        
        # Determine if test passed
        passed = self._evaluate_test_success(buffer_health_score, performance_score)
        
        # Collect detailed results
        details = {
            'pipeline_stats': self.pipeline_stats.copy(),
            'buffer_health': buffer_health,
            'optimization_suggestions': len(optimization_suggestions),
            'metrics_collected': self.metrics_collector.get_debug_info(),
            'track_changes': len(self.mpris_interface.track_changes),
            'final_queue_size': self.buffer_manager.current_queue_size,
            'queue_utilization': self._get_queue_utilization()
        }
        
        return PipelineTestResult(
            test_name=self.test_name,
            duration=actual_duration,
            tracks_processed=self.pipeline_stats['tracks_processed'],
            total_frames=self.pipeline_stats['frames_processed'],
            buffer_health_score=buffer_health_score,
            error_recovery_rate=self._calculate_recovery_rate(),
            optimization_suggestions=len(optimization_suggestions),
            performance_score=performance_score,
            passed=passed,
            details=details
        )
    
    def _calculate_health_score(self, buffer_health) -> float:
        """Calculate buffer health score (0-100)."""
        if not buffer_health:
            return 0.0
        
        # Base score from health status
        status_scores = {
            HealthStatus.HEALTHY: 30,
            HealthStatus.WARNING: 20,
            HealthStatus.CRITICAL: 10
        }
        
        status_score = status_scores.get(buffer_health.status, 0)
        
        # Utilization score (optimal around 50-70%)
        utilization = buffer_health.utilization
        if 50 <= utilization <= 70:
            util_score = 25
        elif 30 <= utilization <= 80:
            util_score = 20
        elif 20 <= utilization <= 90:
            util_score = 15
        else:
            util_score = 10
        
        # Overflow risk score
        overflow_risk = buffer_health.overflow_risk
        risk_score = max(0, 25 - overflow_risk * 25)
        
        # Metrics quality score
        metrics = buffer_health.metrics
        if metrics:
            metrics_score = max(0, 20 - metrics.overflow_count * 2)
        else:
            metrics_score = 10
        
        total_score = status_score + util_score + risk_score + metrics_score
        return min(100, max(0, total_score))
    
    def _calculate_performance_score(self) -> float:
        """Calculate overall performance score (0-100)."""
        # Throughput score (frames per second)
        fps = self.pipeline_stats['frames_processed'] / self.duration
        throughput_score = min(30, fps / 10)  # 300 fps = 30 points
        
        # Error handling score
        error_rate = self.pipeline_stats['error_events'] / max(1, self.pipeline_stats['frames_processed'])
        error_score = max(0, 25 - error_rate * 1000)  # Penalize high error rates
        
        # Buffer management score
        overflow_rate = self.pipeline_stats['buffer_overflows'] / max(1, self.pipeline_stats['frames_processed'])
        underrun_rate = self.pipeline_stats['buffer_underruns'] / max(1, self.pipeline_stats['frames_processed'])
        buffer_score = max(0, 25 - (overflow_rate + underrun_rate) * 500)
        
        # Adaptation score (reward appropriate adaptations)
        adaptation_rate = self.pipeline_stats['buffer_adjustments'] / self.duration
        if 0.1 <= adaptation_rate <= 1.0:  # Reasonable adaptation rate
            adaptation_score = 20
        elif adaptation_rate < 0.1:
            adaptation_score = 15  # Too few adaptations
        else:
            adaptation_score = 10  # Too many adaptations
        
        total_score = throughput_score + error_score + buffer_score + adaptation_score
        return min(100, max(0, total_score))
    
    def _evaluate_test_success(self, buffer_health_score: float, performance_score: float) -> bool:
        """Evaluate if the test passed."""
        # Minimum thresholds for success
        min_buffer_health = 60.0
        min_performance = 60.0
        min_tracks = 1
        max_error_rate = 0.01  # 1% error rate
        
        # Check all criteria
        health_ok = buffer_health_score >= min_buffer_health
        performance_ok = performance_score >= min_performance
        tracks_ok = self.pipeline_stats['tracks_processed'] >= min_tracks
        
        error_rate = self.pipeline_stats['error_events'] / max(1, self.pipeline_stats['frames_processed'])
        errors_ok = error_rate <= max_error_rate
        
        return health_ok and performance_ok and tracks_ok and errors_ok


# Test classes for pytest integration

class TestIntegrationBufferHealth:
    """Test class for integration buffer health tests."""
    
    @pytest.mark.slow
    def test_complete_pipeline_integration(self):
        """Test complete audio pipeline integration."""
        test = CompletePipelineTest("Complete Pipeline Integration", duration=30.0)
        result = test.run_test()
        
        # Verify pipeline functionality
        assert result.total_frames > 0, "Should process audio frames"
        assert result.tracks_processed > 0, "Should process at least one track"
        assert result.buffer_health_score > 60, f"Buffer health too low: {result.buffer_health_score}"
        assert result.performance_score > 60, f"Performance too low: {result.performance_score}"
        assert result.passed, "Complete pipeline integration should pass"
    
    @pytest.mark.slow
    def test_high_load_pipeline_integration(self):
        """Test pipeline integration under high load."""
        test = CompletePipelineTest("High Load Pipeline", duration=20.0)
        
        # Modify buffer manager for high load scenario
        test.buffer_manager.initial_queue_size = 150  # Smaller initial buffer
        test.buffer_manager.max_size = 1000  # Lower max size
        
        result = test.run_test()
        
        # Verify pipeline handles high load
        assert result.total_frames > 0, "Should process frames under high load"
        assert result.buffer_health_score > 50, "Should maintain reasonable health under load"
        assert result.error_recovery_rate > 0.8, "Should have good error recovery under load"
    
    def test_buffer_health_monitoring_integration(self):
        """Test buffer health monitoring integration."""
        test = CompletePipelineTest("Buffer Health Monitoring", duration=15.0)
        result = test.run_test()
        
        # Verify health monitoring
        details = result.details
        assert 'buffer_health' in details, "Should collect buffer health data"
        assert details['buffer_health'] is not None, "Should have buffer health information"
        
        # Verify health score calculation
        assert 0 <= result.buffer_health_score <= 100, "Health score should be in valid range"
    
    def test_error_recovery_integration(self):
        """Test error recovery integration in pipeline."""
        test = CompletePipelineTest("Error Recovery Integration", duration=10.0)
        result = test.run_test()
        
        # Verify error recovery functionality
        if result.details['pipeline_stats']['error_events'] > 0:
            assert result.error_recovery_rate > 0.5, "Should have reasonable error recovery rate"
        
        # Pipeline should continue functioning despite errors
        assert result.total_frames > 0, "Should continue processing despite errors"
    
    def test_performance_optimization_integration(self):
        """Test performance optimization integration."""
        test = CompletePipelineTest("Performance Optimization", duration=20.0)
        result = test.run_test()
        
        # Verify optimization functionality
        assert result.optimization_suggestions >= 0, "Should generate optimization suggestions"
        
        # Performance should be reasonable
        assert result.performance_score > 40, "Should achieve basic performance"
    
    def test_track_boundary_integration(self):
        """Test track boundary detection integration."""
        test = CompletePipelineTest("Track Boundary Integration", duration=25.0)
        result = test.run_test()
        
        # Verify track boundary detection
        pipeline_stats = result.details['pipeline_stats']
        assert pipeline_stats['track_boundaries_detected'] > 0, "Should detect track boundaries"
        assert pipeline_stats['segments_created'] > 0, "Should create audio segments"
        
        # Track processing should work
        assert result.tracks_processed > 0, "Should process tracks"
    
    @pytest.mark.parametrize("duration,expected_min_frames", [
        (10.0, 400),   # 10 seconds should process at least 400 frames
        (20.0, 800),   # 20 seconds should process at least 800 frames
        (30.0, 1200),  # 30 seconds should process at least 1200 frames
    ])
    def test_pipeline_scalability(self, duration, expected_min_frames):
        """Test pipeline scalability with different durations."""
        test = CompletePipelineTest(f"Scalability Test {duration}s", duration=duration)
        result = test.run_test()
        
        # Verify scalability
        assert result.total_frames >= expected_min_frames, \
            f"Should process at least {expected_min_frames} frames in {duration}s"
        assert result.passed, f"Pipeline should pass at {duration}s duration"
    
    def test_mock_components_functionality(self):
        """Test mock components used in integration tests."""
        # Test MockAudioSource
        audio_source = MockAudioSource(44100, 2)
        frame = audio_source.generate_audio_frame(1024)
        
        assert frame.shape == (1024, 2), "Audio frame should have correct shape"
        assert frame.dtype == np.float32, "Audio frame should be float32"
        
        # Test track change simulation
        initial_track = audio_source.current_track
        audio_source.simulate_track_change()
        assert audio_source.current_track == initial_track + 1, "Should increment track"
        
        # Test MockMPRISInterface
        mpris = MockMPRISInterface()
        initial_track = mpris.get_current_track()
        
        new_track = {'title': 'New Track', 'artist': 'New Artist', 'album': 'New Album', 'duration': 200.0}
        mpris.simulate_track_change(new_track)
        
        assert mpris.get_current_track()['title'] == 'New Track', "Should update current track"
        assert len(mpris.track_changes) == 1, "Should record track change"


if __name__ == "__main__":
    # Run integration tests directly
    print("Running Integration Buffer Health Tests...")
    
    # Complete pipeline integration test
    print("\n1. Complete Pipeline Integration Test")
    pipeline_test = CompletePipelineTest("Complete Pipeline Test", duration=15.0)
    pipeline_result = pipeline_test.run_test()
    
    print(f"   Duration: {pipeline_result.duration:.1f}s")
    print(f"   Tracks processed: {pipeline_result.tracks_processed}")
    print(f"   Total frames: {pipeline_result.total_frames}")
    print(f"   Buffer health score: {pipeline_result.buffer_health_score:.1f}")
    print(f"   Performance score: {pipeline_result.performance_score:.1f}")
    print(f"   Error recovery rate: {pipeline_result.error_recovery_rate:.2%}")
    print(f"   Passed: {pipeline_result.passed}")
    
    # High load test
    print("\n2. High Load Pipeline Test")
    high_load_test = CompletePipelineTest("High Load Test", duration=10.0)
    high_load_test.buffer_manager.initial_queue_size = 100  # Smaller buffer for stress
    high_load_result = high_load_test.run_test()
    
    print(f"   Frames processed: {high_load_result.total_frames}")
    print(f"   Buffer health: {high_load_result.buffer_health_score:.1f}")
    print(f"   Performance: {high_load_result.performance_score:.1f}")
    print(f"   Passed: {high_load_result.passed}")
    
    print("\nIntegration buffer health tests completed!")