"""
Long-running session stability tests for audio buffer optimization.

This module provides tests for extended recording sessions to validate
system stability, memory management, and performance consistency over time.
"""

import pytest
import time
import threading
import queue
import numpy as np
import gc
import psutil
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import json
import tempfile
from pathlib import Path

from spotify_splitter.buffer_management import AdaptiveBufferManager, BufferMetrics, HealthStatus
from spotify_splitter.audio import EnhancedAudioStream
from spotify_splitter.error_recovery import ErrorRecoveryManager
from spotify_splitter.metrics_collector import MetricsCollector
from spotify_splitter.segmenter import SegmentManager


@dataclass
class SessionMetrics:
    """Metrics collected during a long-running session."""
    timestamp: datetime
    frames_processed: int
    memory_usage_mb: float
    cpu_usage_percent: float
    buffer_utilization: float
    error_count: int
    recovery_count: int
    queue_size: int
    latency_ms: float


class LongRunningSessionMonitor:
    """Monitor for long-running audio sessions."""
    
    def __init__(self, collection_interval: float = 30.0):
        self.collection_interval = collection_interval
        self.metrics_history: List[SessionMetrics] = []
        self.active = False
        self.monitor_thread = None
        
        # Components to monitor
        self.buffer_manager = None
        self.error_recovery = None
        self.audio_queue = None
        
        # Tracking variables
        self.total_frames = 0
        self.total_errors = 0
        self.total_recoveries = 0
        self.session_start_time = None
    
    def start_monitoring(self, buffer_manager: AdaptiveBufferManager, 
                        error_recovery: ErrorRecoveryManager,
                        audio_queue: queue.Queue):
        """Start monitoring a long-running session."""
        self.buffer_manager = buffer_manager
        self.error_recovery = error_recovery
        self.audio_queue = audio_queue
        self.session_start_time = datetime.now()
        
        self.active = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop monitoring."""
        self.active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.active:
            try:
                # Collect current metrics
                current_metrics = self._collect_metrics()
                self.metrics_history.append(current_metrics)
                
                # Limit history size to prevent memory growth
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-500:]
                
                time.sleep(self.collection_interval)
                
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(self.collection_interval)
    
    def _collect_metrics(self) -> SessionMetrics:
        """Collect current session metrics."""
        try:
            # Memory usage
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            # CPU usage
            cpu_percent = process.cpu_percent()
            
            # Buffer metrics
            buffer_utilization = 0.0
            queue_size = 0
            if self.audio_queue and self.buffer_manager:
                try:
                    buffer_metrics = self.buffer_manager.monitor_utilization(self.audio_queue)
                    buffer_utilization = buffer_metrics.utilization_percent
                    queue_size = self.buffer_manager.current_queue_size
                except:
                    pass
            
            # Error metrics
            error_count = getattr(self.error_recovery, 'total_errors', 0) if self.error_recovery else 0
            recovery_count = getattr(self.error_recovery, 'successful_recoveries', 0) if self.error_recovery else 0
            
            return SessionMetrics(
                timestamp=datetime.now(),
                frames_processed=self.total_frames,
                memory_usage_mb=memory_mb,
                cpu_usage_percent=cpu_percent,
                buffer_utilization=buffer_utilization,
                error_count=error_count,
                recovery_count=recovery_count,
                queue_size=queue_size,
                latency_ms=0.0  # Would be calculated from actual audio processing
            )
            
        except Exception as e:
            print(f"Error collecting metrics: {e}")
            return SessionMetrics(
                timestamp=datetime.now(),
                frames_processed=self.total_frames,
                memory_usage_mb=0.0,
                cpu_usage_percent=0.0,
                buffer_utilization=0.0,
                error_count=0,
                recovery_count=0,
                queue_size=0,
                latency_ms=0.0
            )
    
    def get_session_summary(self) -> dict:
        """Get summary of the monitoring session."""
        if not self.metrics_history:
            return {}
        
        # Calculate session duration
        session_duration = (datetime.now() - self.session_start_time).total_seconds()
        
        # Memory analysis
        memory_values = [m.memory_usage_mb for m in self.metrics_history]
        memory_growth = memory_values[-1] - memory_values[0] if len(memory_values) > 1 else 0
        
        # CPU analysis
        cpu_values = [m.cpu_usage_percent for m in self.metrics_history]
        avg_cpu = sum(cpu_values) / len(cpu_values)
        
        # Buffer analysis
        buffer_values = [m.buffer_utilization for m in self.metrics_history]
        avg_buffer_util = sum(buffer_values) / len(buffer_values)
        
        # Error analysis
        final_errors = self.metrics_history[-1].error_count
        final_recoveries = self.metrics_history[-1].recovery_count
        
        return {
            'session_duration_hours': session_duration / 3600,
            'total_frames_processed': self.total_frames,
            'memory_start_mb': memory_values[0] if memory_values else 0,
            'memory_end_mb': memory_values[-1] if memory_values else 0,
            'memory_growth_mb': memory_growth,
            'memory_growth_rate_mb_per_hour': memory_growth / (session_duration / 3600) if session_duration > 0 else 0,
            'avg_cpu_usage_percent': avg_cpu,
            'max_cpu_usage_percent': max(cpu_values) if cpu_values else 0,
            'avg_buffer_utilization': avg_buffer_util,
            'max_buffer_utilization': max(buffer_values) if buffer_values else 0,
            'total_errors': final_errors,
            'total_recoveries': final_recoveries,
            'error_recovery_rate': final_recoveries / max(1, final_errors),
            'metrics_collected': len(self.metrics_history),
            'stability_score': self._calculate_stability_score()
        }
    
    def _calculate_stability_score(self) -> float:
        """Calculate overall stability score (0-100)."""
        if not self.metrics_history:
            return 0.0
        
        # Memory stability (penalize growth)
        memory_values = [m.memory_usage_mb for m in self.metrics_history]
        if len(memory_values) > 1:
            memory_growth_rate = (memory_values[-1] - memory_values[0]) / len(memory_values)
            memory_score = max(0, 30 - memory_growth_rate * 10)  # Penalize growth
        else:
            memory_score = 30
        
        # CPU stability (penalize high usage)
        cpu_values = [m.cpu_usage_percent for m in self.metrics_history]
        avg_cpu = sum(cpu_values) / len(cpu_values)
        cpu_score = max(0, 30 - (avg_cpu - 50) * 0.6) if avg_cpu > 50 else 30
        
        # Buffer stability (penalize high utilization)
        buffer_values = [m.buffer_utilization for m in self.metrics_history]
        avg_buffer = sum(buffer_values) / len(buffer_values)
        buffer_score = max(0, 25 - (avg_buffer - 70) * 0.5) if avg_buffer > 70 else 25
        
        # Error handling (reward good recovery)
        final_errors = self.metrics_history[-1].error_count
        final_recoveries = self.metrics_history[-1].recovery_count
        if final_errors > 0:
            recovery_rate = final_recoveries / final_errors
            error_score = recovery_rate * 15
        else:
            error_score = 15  # No errors is perfect
        
        return memory_score + cpu_score + buffer_score + error_score


class ExtendedRecordingSessionTest:
    """Test for extended recording sessions (multiple hours)."""
    
    def __init__(self, duration_hours: float = 2.0, track_count: int = 100):
        self.duration_hours = duration_hours
        self.track_count = track_count
        self.duration_seconds = duration_hours * 3600
        self.results = {}
    
    def run_test(self) -> dict:
        """Run extended recording session test."""
        # Create components
        buffer_manager = AdaptiveBufferManager(
            initial_queue_size=300,
            min_size=100,
            max_size=2000
        )
        
        error_recovery = ErrorRecoveryManager(
            max_retries=5,
            backoff_factor=1.5
        )
        
        metrics_collector = MetricsCollector(
            collection_interval=60.0,  # Collect every minute
            enable_debug_mode=True
        )
        
        # Create audio processing queue
        audio_queue = queue.Queue(maxsize=buffer_manager.current_queue_size)
        
        # Start monitoring
        monitor = LongRunningSessionMonitor(collection_interval=60.0)
        monitor.start_monitoring(buffer_manager, error_recovery, audio_queue)
        
        # Performance tracking
        session_stats = {
            'tracks_processed': 0,
            'total_frames': 0,
            'buffer_overflows': 0,
            'buffer_underruns': 0,
            'error_events': 0,
            'recovery_attempts': 0,
            'successful_recoveries': 0,
            'buffer_adjustments': 0,
            'memory_cleanups': 0
        }
        
        # Control flags
        active = threading.Event()
        active.set()
        
        def audio_producer():
            """Simulate continuous audio production."""
            track_id = 0
            frames_in_track = 0
            target_frames_per_track = 44100 * 180  # 3 minutes per track at 44.1kHz
            
            while active.is_set() and track_id < self.track_count:
                try:
                    # Create audio frame (simulate 10ms of audio)
                    frame_samples = 441  # 10ms at 44.1kHz
                    audio_frame = np.random.random((frame_samples, 2)).astype(np.float32)
                    
                    # Try to queue frame
                    try:
                        audio_queue.put_nowait(audio_frame)
                        session_stats['total_frames'] += 1
                        monitor.total_frames += 1
                        frames_in_track += frame_samples
                        
                        # Check if track is complete
                        if frames_in_track >= target_frames_per_track:
                            session_stats['tracks_processed'] += 1
                            track_id += 1
                            frames_in_track = 0
                            
                            # Simulate track boundary processing
                            time.sleep(0.1)  # Brief pause for track processing
                    
                    except queue.Full:
                        session_stats['buffer_overflows'] += 1
                        
                        # Try emergency buffer expansion
                        if buffer_manager.emergency_expansion():
                            session_stats['buffer_adjustments'] += 1
                            
                            # Resize queue
                            new_queue = queue.Queue(maxsize=buffer_manager.current_queue_size)
                            while not audio_queue.empty():
                                try:
                                    new_queue.put_nowait(audio_queue.get_nowait())
                                except queue.Full:
                                    break
                            audio_queue = new_queue
                    
                    # Simulate real-time audio (10ms per frame)
                    time.sleep(0.01)
                    
                except Exception as e:
                    session_stats['error_events'] += 1
                    print(f"Producer error: {e}")
                    time.sleep(0.01)
        
        def audio_consumer():
            """Simulate continuous audio consumption and processing."""
            while active.is_set():
                try:
                    # Get audio frame
                    try:
                        audio_frame = audio_queue.get_nowait()
                        
                        # Simulate processing time
                        processing_time = np.random.uniform(0.005, 0.015)  # 5-15ms
                        time.sleep(processing_time)
                        
                    except queue.Empty:
                        session_stats['buffer_underruns'] += 1
                        time.sleep(0.001)  # Brief wait for more data
                    
                    # Periodic buffer monitoring and adjustment
                    if session_stats['total_frames'] % 1000 == 0:
                        try:
                            metrics = buffer_manager.monitor_utilization(audio_queue)
                            old_size = buffer_manager.current_queue_size
                            buffer_manager.adjust_buffer_size(metrics)
                            
                            if buffer_manager.current_queue_size != old_size:
                                session_stats['buffer_adjustments'] += 1
                        except Exception as e:
                            print(f"Buffer monitoring error: {e}")
                    
                    # Periodic memory cleanup
                    if session_stats['total_frames'] % 10000 == 0:
                        gc.collect()
                        session_stats['memory_cleanups'] += 1
                    
                    # Simulate occasional errors
                    if np.random.random() < 0.0001:  # 0.01% error rate
                        session_stats['error_events'] += 1
                        session_stats['recovery_attempts'] += 1
                        
                        # Simulate recovery
                        recovery_success = error_recovery.handle_error(
                            Exception("Simulated processing error"),
                            "audio_consumer"
                        )
                        
                        if recovery_success:
                            session_stats['successful_recoveries'] += 1
                        
                        time.sleep(0.1)  # Recovery delay
                
                except Exception as e:
                    session_stats['error_events'] += 1
                    print(f"Consumer error: {e}")
                    time.sleep(0.01)
        
        # Start metrics collection
        metrics_collector.start_collection()
        
        # Start audio processing threads
        producer_thread = threading.Thread(target=audio_producer, daemon=True)
        consumer_thread = threading.Thread(target=audio_consumer, daemon=True)
        
        producer_thread.start()
        consumer_thread.start()
        
        # Run for specified duration
        start_time = time.time()
        try:
            time.sleep(self.duration_seconds)
        except KeyboardInterrupt:
            print("Test interrupted by user")
        
        actual_duration = time.time() - start_time
        
        # Stop all activity
        active.clear()
        
        # Wait for threads to finish
        producer_thread.join(timeout=5.0)
        consumer_thread.join(timeout=5.0)
        
        # Stop monitoring and metrics collection
        monitor.stop_monitoring()
        metrics_collector.stop_collection()
        
        # Get final results
        session_summary = monitor.get_session_summary()
        debug_info = metrics_collector.get_debug_info()
        
        # Calculate final metrics
        self.results = {
            'planned_duration_hours': self.duration_hours,
            'actual_duration_hours': actual_duration / 3600,
            'planned_tracks': self.track_count,
            'tracks_processed': session_stats['tracks_processed'],
            'total_frames_processed': session_stats['total_frames'],
            'frames_per_second': session_stats['total_frames'] / actual_duration,
            'buffer_overflows': session_stats['buffer_overflows'],
            'buffer_underruns': session_stats['buffer_underruns'],
            'buffer_adjustments': session_stats['buffer_adjustments'],
            'error_events': session_stats['error_events'],
            'recovery_attempts': session_stats['recovery_attempts'],
            'successful_recoveries': session_stats['successful_recoveries'],
            'recovery_success_rate': session_stats['successful_recoveries'] / max(1, session_stats['recovery_attempts']),
            'memory_cleanups': session_stats['memory_cleanups'],
            'overflow_rate_per_hour': session_stats['buffer_overflows'] / (actual_duration / 3600),
            'underrun_rate_per_hour': session_stats['buffer_underruns'] / (actual_duration / 3600),
            'error_rate_per_hour': session_stats['error_events'] / (actual_duration / 3600),
            'session_monitoring': session_summary,
            'metrics_collection': debug_info,
            'final_buffer_size': buffer_manager.current_queue_size,
            'completion_rate': session_stats['tracks_processed'] / self.track_count
        }
        
        return self.results


class MemoryLeakDetectionTest:
    """Test for detecting memory leaks in long-running sessions."""
    
    def __init__(self, duration_minutes: int = 60, sample_interval: int = 30):
        self.duration_minutes = duration_minutes
        self.sample_interval = sample_interval
        self.results = {}
    
    def run_test(self) -> dict:
        """Run memory leak detection test."""
        # Create components
        buffer_manager = AdaptiveBufferManager(
            initial_queue_size=200,
            min_size=50,
            max_size=1000
        )
        
        metrics_collector = MetricsCollector(
            collection_interval=10.0,
            enable_debug_mode=True
        )
        
        # Memory tracking
        memory_samples = []
        object_counts = []
        
        # Control flags
        active = threading.Event()
        active.set()
        
        def memory_monitor():
            """Monitor memory usage over time."""
            while active.is_set():
                try:
                    # Get memory info
                    process = psutil.Process()
                    memory_info = process.memory_info()
                    
                    memory_sample = {
                        'timestamp': time.time(),
                        'rss_mb': memory_info.rss / 1024 / 1024,
                        'vms_mb': memory_info.vms / 1024 / 1024,
                        'percent': process.memory_percent()
                    }
                    
                    # Get object counts (simplified)
                    gc.collect()  # Force garbage collection
                    object_count = len(gc.get_objects())
                    
                    memory_samples.append(memory_sample)
                    object_counts.append({
                        'timestamp': time.time(),
                        'object_count': object_count
                    })
                    
                    time.sleep(self.sample_interval)
                    
                except Exception as e:
                    print(f"Memory monitoring error: {e}")
                    time.sleep(self.sample_interval)
        
        def workload_generator():
            """Generate continuous workload to test for leaks."""
            audio_queue = queue.Queue(maxsize=buffer_manager.current_queue_size)
            frame_count = 0
            
            while active.is_set():
                try:
                    # Create and process audio frames
                    for _ in range(100):  # Process in batches
                        frame_size = np.random.randint(256, 1024)
                        audio_frame = np.random.random((frame_size, 2)).astype(np.float32)
                        
                        try:
                            audio_queue.put_nowait(audio_frame)
                            processed_frame = audio_queue.get_nowait()
                            frame_count += 1
                            
                            # Simulate some processing
                            _ = np.mean(processed_frame)
                            
                        except (queue.Full, queue.Empty):
                            pass
                    
                    # Monitor buffer utilization
                    if frame_count % 1000 == 0:
                        try:
                            metrics = buffer_manager.monitor_utilization(audio_queue)
                            buffer_manager.adjust_buffer_size(metrics)
                        except:
                            pass
                    
                    # Record some metrics
                    metrics_collector.record_counter('frames_processed', frame_count)
                    metrics_collector.record_gauge('queue_size', audio_queue.qsize())
                    
                    time.sleep(0.01)  # Brief pause
                    
                except Exception as e:
                    print(f"Workload error: {e}")
                    time.sleep(0.01)
        
        # Start monitoring and workload
        metrics_collector.start_collection()
        
        monitor_thread = threading.Thread(target=memory_monitor, daemon=True)
        workload_thread = threading.Thread(target=workload_generator, daemon=True)
        
        monitor_thread.start()
        workload_thread.start()
        
        # Run for specified duration
        time.sleep(self.duration_minutes * 60)
        
        # Stop all activity
        active.clear()
        metrics_collector.stop_collection()
        
        # Wait for threads
        monitor_thread.join(timeout=2.0)
        workload_thread.join(timeout=2.0)
        
        # Analyze memory usage
        if len(memory_samples) < 2:
            return {'error': 'Insufficient memory samples collected'}
        
        # Calculate memory growth
        initial_memory = memory_samples[0]['rss_mb']
        final_memory = memory_samples[-1]['rss_mb']
        memory_growth = final_memory - initial_memory
        
        # Calculate growth rate
        duration_hours = self.duration_minutes / 60.0
        growth_rate_mb_per_hour = memory_growth / duration_hours
        
        # Analyze object count growth
        initial_objects = object_counts[0]['object_count'] if object_counts else 0
        final_objects = object_counts[-1]['object_count'] if object_counts else 0
        object_growth = final_objects - initial_objects
        
        # Detect potential leaks
        leak_detected = False
        leak_severity = "none"
        
        if growth_rate_mb_per_hour > 10:  # More than 10MB/hour growth
            leak_detected = True
            if growth_rate_mb_per_hour > 50:
                leak_severity = "severe"
            elif growth_rate_mb_per_hour > 25:
                leak_severity = "moderate"
            else:
                leak_severity = "minor"
        
        self.results = {
            'test_duration_minutes': self.duration_minutes,
            'memory_samples_collected': len(memory_samples),
            'initial_memory_mb': initial_memory,
            'final_memory_mb': final_memory,
            'memory_growth_mb': memory_growth,
            'memory_growth_rate_mb_per_hour': growth_rate_mb_per_hour,
            'initial_object_count': initial_objects,
            'final_object_count': final_objects,
            'object_count_growth': object_growth,
            'leak_detected': leak_detected,
            'leak_severity': leak_severity,
            'max_memory_mb': max(s['rss_mb'] for s in memory_samples),
            'min_memory_mb': min(s['rss_mb'] for s in memory_samples),
            'memory_variance': np.var([s['rss_mb'] for s in memory_samples]),
            'samples_per_hour': len(memory_samples) / duration_hours
        }
        
        return self.results


# Test classes for pytest integration

class TestLongRunningSessionStability:
    """Test class for long-running session stability."""
    
    @pytest.mark.slow
    @pytest.mark.timeout(7200)  # 2 hour timeout
    def test_extended_recording_session(self):
        """Test extended recording session (2 hours)."""
        test = ExtendedRecordingSessionTest(duration_hours=0.5, track_count=10)  # Shorter for testing
        results = test.run_test()
        
        # Verify session completed successfully
        assert results['completion_rate'] > 0.5, "Should complete at least half the planned tracks"
        assert results['total_frames_processed'] > 0, "Should process audio frames"
        
        # Verify stability metrics
        assert results['recovery_success_rate'] > 0.8, "Should have high recovery success rate"
        assert results['overflow_rate_per_hour'] < 100, "Should have reasonable overflow rate"
        
        # Verify memory stability
        session_monitoring = results.get('session_monitoring', {})
        if session_monitoring:
            assert session_monitoring.get('memory_growth_rate_mb_per_hour', 0) < 50, \
                "Memory growth should be reasonable"
            assert session_monitoring.get('stability_score', 0) > 50, \
                "Should maintain reasonable stability"
    
    @pytest.mark.slow
    def test_memory_leak_detection(self):
        """Test memory leak detection over extended period."""
        test = MemoryLeakDetectionTest(duration_minutes=30, sample_interval=10)  # Shorter for testing
        results = test.run_test()
        
        # Verify test completed
        assert results.get('memory_samples_collected', 0) > 0, "Should collect memory samples"
        
        # Check for memory leaks
        growth_rate = results.get('memory_growth_rate_mb_per_hour', 0)
        assert growth_rate < 100, f"Memory growth rate too high: {growth_rate} MB/hour"
        
        # Severe leaks should not be detected in normal operation
        assert results.get('leak_severity', 'none') != 'severe', "Should not detect severe memory leaks"
    
    def test_session_monitor_functionality(self):
        """Test session monitoring functionality."""
        buffer_manager = AdaptiveBufferManager()
        error_recovery = ErrorRecoveryManager()
        audio_queue = queue.Queue(maxsize=200)
        
        monitor = LongRunningSessionMonitor(collection_interval=1.0)
        
        # Test monitoring startup and shutdown
        monitor.start_monitoring(buffer_manager, error_recovery, audio_queue)
        time.sleep(2)  # Let it collect a few samples
        monitor.stop_monitoring()
        
        # Verify monitoring worked
        summary = monitor.get_session_summary()
        assert summary.get('session_duration_hours', 0) > 0, "Should record session duration"
        assert summary.get('metrics_collected', 0) > 0, "Should collect metrics"


if __name__ == "__main__":
    # Run long-running session tests directly
    print("Running Long-Running Session Stability Tests...")
    
    # Extended recording session test (shortened for demo)
    print("\n1. Extended Recording Session Test")
    session_test = ExtendedRecordingSessionTest(duration_hours=0.1, track_count=5)
    session_results = session_test.run_test()
    print(f"   Tracks processed: {session_results['tracks_processed']}/5")
    print(f"   Total frames: {session_results['total_frames_processed']}")
    print(f"   Recovery success rate: {session_results['recovery_success_rate']:.2%}")
    
    # Memory leak detection test
    print("\n2. Memory Leak Detection Test")
    memory_test = MemoryLeakDetectionTest(duration_minutes=5, sample_interval=10)
    memory_results = memory_test.run_test()
    print(f"   Memory growth: {memory_results['memory_growth_mb']:.2f} MB")
    print(f"   Growth rate: {memory_results['memory_growth_rate_mb_per_hour']:.2f} MB/hour")
    print(f"   Leak detected: {memory_results['leak_detected']}")
    
    print("\nLong-running session tests completed!")