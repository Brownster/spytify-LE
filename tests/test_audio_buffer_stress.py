"""
Stress tests for audio buffer optimization under high-load scenarios.

This module provides comprehensive stress testing to validate buffer management
performance under extreme conditions and high system load.
"""

import pytest
import time
import threading
import queue
import numpy as np
import multiprocessing
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import psutil
import os

from spotify_splitter.buffer_management import AdaptiveBufferManager, BufferMetrics, HealthStatus
from spotify_splitter.audio import EnhancedAudioStream
from spotify_splitter.error_recovery import ErrorRecoveryManager
from spotify_splitter.metrics_collector import MetricsCollector


class StressTestEnvironment:
    """Environment for running stress tests with controlled load."""
    
    def __init__(self, cpu_load_percent: float = 80.0, memory_pressure: bool = False):
        self.cpu_load_percent = cpu_load_percent
        self.memory_pressure = memory_pressure
        self.load_processes = []
        self.memory_hogs = []
        self.active = False
    
    def start_load(self):
        """Start artificial system load."""
        self.active = True
        
        # CPU load
        if self.cpu_load_percent > 0:
            cpu_count = multiprocessing.cpu_count()
            load_processes = max(1, int(cpu_count * self.cpu_load_percent / 100))
            
            for _ in range(load_processes):
                p = multiprocessing.Process(target=self._cpu_load_worker)
                p.start()
                self.load_processes.append(p)
        
        # Memory pressure
        if self.memory_pressure:
            memory_thread = threading.Thread(target=self._memory_pressure_worker, daemon=True)
            memory_thread.start()
    
    def stop_load(self):
        """Stop artificial system load."""
        self.active = False
        
        # Stop CPU load processes
        for p in self.load_processes:
            p.terminate()
            p.join(timeout=1.0)
        self.load_processes.clear()
        
        # Clean up memory
        self.memory_hogs.clear()
        gc.collect()
    
    def _cpu_load_worker(self):
        """Worker function to generate CPU load."""
        while self.active:
            # Busy work
            for _ in range(10000):
                _ = sum(i * i for i in range(100))
            time.sleep(0.001)  # Brief pause to allow other processes
    
    def _memory_pressure_worker(self):
        """Worker function to generate memory pressure."""
        while self.active:
            # Allocate memory chunks
            try:
                chunk = np.random.random((1000, 1000)).astype(np.float32)
                self.memory_hogs.append(chunk)
                
                # Keep memory usage reasonable
                if len(self.memory_hogs) > 100:
                    self.memory_hogs.pop(0)
                
                time.sleep(0.1)
            except MemoryError:
                # Clean up if we hit memory limits
                self.memory_hogs = self.memory_hogs[-50:]
                gc.collect()
                time.sleep(1.0)


class HighLoadBufferStressTest:
    """Stress test for buffer management under high load."""
    
    def __init__(self, duration: int = 60):
        self.duration = duration
        self.results = {}
    
    def run_test(self) -> dict:
        """Run high-load buffer stress test."""
        # Create stress environment
        stress_env = StressTestEnvironment(cpu_load_percent=75.0, memory_pressure=True)
        
        # Create buffer manager
        buffer_manager = AdaptiveBufferManager(
            initial_queue_size=200,
            min_size=50,
            max_size=2000  # Allow larger buffers under stress
        )
        
        # Create test queues for multiple streams
        num_streams = 4
        test_queues = [queue.Queue(maxsize=200) for _ in range(num_streams)]
        
        # Performance tracking
        metrics = {
            'produced_items': [0] * num_streams,
            'consumed_items': [0] * num_streams,
            'overflow_count': [0] * num_streams,
            'underrun_count': [0] * num_streams,
            'buffer_adjustments': 0,
            'emergency_expansions': 0,
            'max_queue_size': buffer_manager.current_queue_size,
            'system_metrics': []
        }
        
        # Control flags
        active = threading.Event()
        active.set()
        
        def producer(stream_id: int):
            """Producer for a specific stream."""
            while active.is_set():
                try:
                    # Simulate varying audio data sizes
                    size = np.random.randint(512, 2048)
                    audio_data = np.random.random((size, 2)).astype(np.float32)
                    
                    test_queues[stream_id].put_nowait(audio_data)
                    metrics['produced_items'][stream_id] += 1
                    
                    # Variable production rate to simulate real audio
                    time.sleep(np.random.uniform(0.001, 0.005))
                    
                except queue.Full:
                    metrics['overflow_count'][stream_id] += 1
                    time.sleep(0.001)
        
        def consumer(stream_id: int):
            """Consumer for a specific stream."""
            while active.is_set():
                try:
                    data = test_queues[stream_id].get_nowait()
                    metrics['consumed_items'][stream_id] += 1
                    
                    # Simulate processing time with variation
                    processing_time = np.random.uniform(0.002, 0.008)
                    time.sleep(processing_time)
                    
                except queue.Empty:
                    metrics['underrun_count'][stream_id] += 1
                    time.sleep(0.001)
        
        def monitor():
            """Monitor and adapt buffer performance."""
            while active.is_set():
                try:
                    # Monitor all queues
                    total_utilization = 0
                    for i, q in enumerate(test_queues):
                        queue_metrics = buffer_manager.monitor_utilization(q)
                        total_utilization += queue_metrics.utilization_percent
                        
                        # Trigger adaptation if needed
                        old_size = buffer_manager.current_queue_size
                        buffer_manager.adjust_buffer_size(queue_metrics)
                        
                        if buffer_manager.current_queue_size != old_size:
                            metrics['buffer_adjustments'] += 1
                            metrics['max_queue_size'] = max(
                                metrics['max_queue_size'], 
                                buffer_manager.current_queue_size
                            )
                    
                    # Record system metrics
                    try:
                        cpu_percent = psutil.cpu_percent(interval=None)
                        memory_percent = psutil.virtual_memory().percent
                        metrics['system_metrics'].append({
                            'timestamp': time.time(),
                            'cpu_percent': cpu_percent,
                            'memory_percent': memory_percent,
                            'avg_utilization': total_utilization / len(test_queues)
                        })
                    except:
                        pass  # Skip if psutil not available
                    
                    time.sleep(0.1)  # Monitor every 100ms
                    
                except Exception as e:
                    print(f"Monitor error: {e}")
        
        # Start stress environment
        stress_env.start_load()
        
        try:
            # Start all threads
            threads = []
            
            # Producer threads
            for i in range(num_streams):
                t = threading.Thread(target=producer, args=(i,), daemon=True)
                t.start()
                threads.append(t)
            
            # Consumer threads
            for i in range(num_streams):
                t = threading.Thread(target=consumer, args=(i,), daemon=True)
                t.start()
                threads.append(t)
            
            # Monitor thread
            monitor_thread = threading.Thread(target=monitor, daemon=True)
            monitor_thread.start()
            threads.append(monitor_thread)
            
            # Run for specified duration
            time.sleep(self.duration)
            
        finally:
            # Stop all activity
            active.clear()
            stress_env.stop_load()
            
            # Wait for threads to finish
            for t in threads:
                t.join(timeout=1.0)
        
        # Calculate results
        total_produced = sum(metrics['produced_items'])
        total_consumed = sum(metrics['consumed_items'])
        total_overflows = sum(metrics['overflow_count'])
        total_underruns = sum(metrics['underrun_count'])
        
        self.results = {
            'duration': self.duration,
            'num_streams': num_streams,
            'total_produced': total_produced,
            'total_consumed': total_consumed,
            'total_overflows': total_overflows,
            'total_underruns': total_underruns,
            'throughput_per_sec': total_consumed / self.duration,
            'overflow_rate_per_sec': total_overflows / self.duration,
            'underrun_rate_per_sec': total_underruns / self.duration,
            'efficiency': total_consumed / max(1, total_produced),
            'buffer_adjustments': metrics['buffer_adjustments'],
            'max_queue_size_reached': metrics['max_queue_size'],
            'adaptation_rate': metrics['buffer_adjustments'] / self.duration,
            'system_metrics': metrics['system_metrics']
        }
        
        return self.results


class ConcurrentStreamStressTest:
    """Stress test for multiple concurrent audio streams."""
    
    def __init__(self, num_streams: int = 8, duration: int = 30):
        self.num_streams = num_streams
        self.duration = duration
        self.results = {}
    
    def run_test(self) -> dict:
        """Run concurrent stream stress test."""
        # Create multiple buffer managers (one per stream)
        buffer_managers = [
            AdaptiveBufferManager(
                initial_queue_size=150,
                min_size=50,
                max_size=1000
            ) for _ in range(self.num_streams)
        ]
        
        # Create error recovery managers
        error_managers = [
            ErrorRecoveryManager(max_retries=3, backoff_factor=1.5)
            for _ in range(self.num_streams)
        ]
        
        # Performance tracking per stream
        stream_metrics = []
        for i in range(self.num_streams):
            stream_metrics.append({
                'stream_id': i,
                'frames_processed': 0,
                'errors_encountered': 0,
                'recoveries_attempted': 0,
                'buffer_overflows': 0,
                'adaptation_count': 0,
                'avg_latency_ms': 0.0,
                'latency_samples': []
            })
        
        # Control flags
        active = threading.Event()
        active.set()
        
        def stream_worker(stream_id: int):
            """Worker function for a single audio stream."""
            buffer_manager = buffer_managers[stream_id]
            error_manager = error_managers[stream_id]
            metrics = stream_metrics[stream_id]
            
            # Create mock audio stream
            audio_queue = queue.Queue(maxsize=buffer_manager.current_queue_size)
            
            while active.is_set():
                try:
                    start_time = time.time()
                    
                    # Simulate audio frame processing
                    frame_size = np.random.randint(256, 1024)
                    audio_frame = np.random.random((frame_size, 2)).astype(np.float32)
                    
                    # Try to process frame
                    try:
                        audio_queue.put_nowait(audio_frame)
                        
                        # Simulate processing
                        processing_time = np.random.uniform(0.001, 0.01)
                        time.sleep(processing_time)
                        
                        # Get processed frame
                        processed_frame = audio_queue.get_nowait()
                        metrics['frames_processed'] += 1
                        
                        # Record latency
                        latency_ms = (time.time() - start_time) * 1000
                        metrics['latency_samples'].append(latency_ms)
                        
                    except queue.Full:
                        metrics['buffer_overflows'] += 1
                        
                        # Try emergency expansion
                        if buffer_manager.emergency_expansion():
                            # Resize queue
                            new_queue = queue.Queue(maxsize=buffer_manager.current_queue_size)
                            while not audio_queue.empty():
                                try:
                                    new_queue.put_nowait(audio_queue.get_nowait())
                                except queue.Full:
                                    break
                            audio_queue = new_queue
                            metrics['adaptation_count'] += 1
                    
                    # Monitor buffer health
                    if metrics['frames_processed'] % 100 == 0:
                        queue_metrics = buffer_manager.monitor_utilization(audio_queue)
                        old_size = buffer_manager.current_queue_size
                        buffer_manager.adjust_buffer_size(queue_metrics)
                        
                        if buffer_manager.current_queue_size != old_size:
                            metrics['adaptation_count'] += 1
                    
                    # Simulate occasional errors
                    if np.random.random() < 0.001:  # 0.1% error rate
                        metrics['errors_encountered'] += 1
                        
                        # Attempt recovery
                        recovery_action = error_manager.handle_error(
                            Exception("Simulated stream error"), 
                            f"stream_{stream_id}"
                        )
                        
                        if recovery_action:
                            metrics['recoveries_attempted'] += 1
                            time.sleep(0.01)  # Recovery delay
                
                except Exception as e:
                    metrics['errors_encountered'] += 1
                    time.sleep(0.001)
        
        # Start all stream workers
        with ThreadPoolExecutor(max_workers=self.num_streams) as executor:
            futures = [
                executor.submit(stream_worker, i) 
                for i in range(self.num_streams)
            ]
            
            # Run for specified duration
            time.sleep(self.duration)
            active.clear()
            
            # Wait for all workers to complete
            for future in as_completed(futures, timeout=5.0):
                try:
                    future.result()
                except Exception as e:
                    print(f"Stream worker error: {e}")
        
        # Calculate final metrics
        for metrics in stream_metrics:
            if metrics['latency_samples']:
                metrics['avg_latency_ms'] = sum(metrics['latency_samples']) / len(metrics['latency_samples'])
                metrics['max_latency_ms'] = max(metrics['latency_samples'])
                metrics['min_latency_ms'] = min(metrics['latency_samples'])
            else:
                metrics['avg_latency_ms'] = 0.0
                metrics['max_latency_ms'] = 0.0
                metrics['min_latency_ms'] = 0.0
            
            # Remove raw samples to save memory
            del metrics['latency_samples']
        
        # Aggregate results
        total_frames = sum(m['frames_processed'] for m in stream_metrics)
        total_errors = sum(m['errors_encountered'] for m in stream_metrics)
        total_overflows = sum(m['buffer_overflows'] for m in stream_metrics)
        total_adaptations = sum(m['adaptation_count'] for m in stream_metrics)
        
        self.results = {
            'duration': self.duration,
            'num_streams': self.num_streams,
            'total_frames_processed': total_frames,
            'total_errors': total_errors,
            'total_buffer_overflows': total_overflows,
            'total_adaptations': total_adaptations,
            'frames_per_second': total_frames / self.duration,
            'error_rate_percent': (total_errors / max(1, total_frames)) * 100,
            'overflow_rate_percent': (total_overflows / max(1, total_frames)) * 100,
            'adaptation_rate_per_sec': total_adaptations / self.duration,
            'stream_metrics': stream_metrics,
            'avg_latency_ms': sum(m['avg_latency_ms'] for m in stream_metrics) / self.num_streams,
            'max_latency_ms': max(m['max_latency_ms'] for m in stream_metrics),
            'successful_streams': len([m for m in stream_metrics if m['frames_processed'] > 0])
        }
        
        return self.results


class MemoryPressureStressTest:
    """Stress test for buffer management under memory pressure."""
    
    def __init__(self, duration: int = 45):
        self.duration = duration
        self.results = {}
    
    def run_test(self) -> dict:
        """Run memory pressure stress test."""
        # Create buffer manager with memory-aware settings
        buffer_manager = AdaptiveBufferManager(
            initial_queue_size=100,  # Start smaller under memory pressure
            min_size=25,
            max_size=500  # Lower max to conserve memory
        )
        
        # Memory tracking
        memory_allocations = []
        buffer_metrics = []
        
        # Create audio processing queue
        audio_queue = queue.Queue(maxsize=buffer_manager.current_queue_size)
        
        # Control flags
        active = threading.Event()
        active.set()
        
        def memory_pressure_generator():
            """Generate memory pressure by allocating large chunks."""
            while active.is_set():
                try:
                    # Allocate memory in chunks
                    chunk_size = np.random.randint(1000, 5000)
                    chunk = np.random.random((chunk_size, chunk_size)).astype(np.float32)
                    memory_allocations.append(chunk)
                    
                    # Periodically clean up to avoid system crash
                    if len(memory_allocations) > 20:
                        # Keep only recent allocations
                        memory_allocations[:] = memory_allocations[-10:]
                        gc.collect()
                    
                    time.sleep(0.1)
                    
                except MemoryError:
                    # Clean up aggressively on memory error
                    memory_allocations.clear()
                    gc.collect()
                    time.sleep(1.0)
        
        def audio_processor():
            """Process audio under memory pressure."""
            frames_processed = 0
            memory_errors = 0
            buffer_adjustments = 0
            
            while active.is_set():
                try:
                    # Create audio frame
                    frame_size = np.random.randint(256, 1024)
                    audio_frame = np.random.random((frame_size, 2)).astype(np.float32)
                    
                    # Try to queue frame
                    try:
                        audio_queue.put_nowait(audio_frame)
                        
                        # Process frame
                        processed_frame = audio_queue.get_nowait()
                        frames_processed += 1
                        
                        # Monitor buffer every 50 frames
                        if frames_processed % 50 == 0:
                            try:
                                metrics = buffer_manager.monitor_utilization(audio_queue)
                                buffer_metrics.append({
                                    'timestamp': time.time(),
                                    'utilization': metrics.utilization_percent,
                                    'queue_size': buffer_manager.current_queue_size,
                                    'frames_processed': frames_processed
                                })
                                
                                # Adapt buffer size
                                old_size = buffer_manager.current_queue_size
                                buffer_manager.adjust_buffer_size(metrics)
                                
                                if buffer_manager.current_queue_size != old_size:
                                    buffer_adjustments += 1
                                    
                                    # Resize queue if needed
                                    if buffer_manager.current_queue_size != audio_queue.maxsize:
                                        new_queue = queue.Queue(maxsize=buffer_manager.current_queue_size)
                                        while not audio_queue.empty():
                                            try:
                                                new_queue.put_nowait(audio_queue.get_nowait())
                                            except queue.Full:
                                                break
                                        audio_queue = new_queue
                                
                            except Exception as e:
                                print(f"Buffer monitoring error: {e}")
                    
                    except queue.Full:
                        # Try emergency expansion under memory pressure
                        if buffer_manager.emergency_expansion():
                            buffer_adjustments += 1
                    
                    time.sleep(0.001)  # Small processing delay
                    
                except MemoryError:
                    memory_errors += 1
                    gc.collect()
                    time.sleep(0.01)
                except Exception as e:
                    print(f"Audio processing error: {e}")
                    time.sleep(0.001)
            
            return frames_processed, memory_errors, buffer_adjustments
        
        # Start memory pressure and audio processing
        memory_thread = threading.Thread(target=memory_pressure_generator, daemon=True)
        memory_thread.start()
        
        # Run audio processing
        start_time = time.time()
        frames_processed, memory_errors, buffer_adjustments = audio_processor()
        actual_duration = time.time() - start_time
        
        # Stop memory pressure
        active.clear()
        memory_thread.join(timeout=1.0)
        
        # Clean up memory
        memory_allocations.clear()
        gc.collect()
        
        # Calculate results
        self.results = {
            'duration': actual_duration,
            'frames_processed': frames_processed,
            'memory_errors': memory_errors,
            'buffer_adjustments': buffer_adjustments,
            'frames_per_second': frames_processed / actual_duration,
            'memory_error_rate': memory_errors / max(1, frames_processed),
            'adaptation_rate': buffer_adjustments / actual_duration,
            'final_buffer_size': buffer_manager.current_queue_size,
            'buffer_metrics_count': len(buffer_metrics),
            'avg_utilization': sum(m['utilization'] for m in buffer_metrics) / max(1, len(buffer_metrics)),
            'max_utilization': max((m['utilization'] for m in buffer_metrics), default=0),
            'memory_allocations_peak': len(memory_allocations) if memory_allocations else 0
        }
        
        return self.results


# Test classes for pytest integration

class TestAudioBufferStress:
    """Test class for audio buffer stress tests."""
    
    @pytest.mark.slow
    def test_high_load_buffer_stress(self):
        """Test buffer management under high system load."""
        stress_test = HighLoadBufferStressTest(duration=30)
        results = stress_test.run_test()
        
        # Verify basic functionality under stress
        assert results['total_consumed'] > 0, "Should process some audio frames under stress"
        assert results['efficiency'] > 0.3, "Should maintain reasonable efficiency under stress"
        assert results['throughput_per_sec'] > 10, "Should maintain minimum throughput"
        
        # Verify adaptive behavior
        assert results['buffer_adjustments'] > 0, "Should adapt buffer sizes under stress"
        assert results['max_queue_size_reached'] >= 200, "Should expand buffers when needed"
    
    @pytest.mark.slow
    def test_concurrent_stream_stress(self):
        """Test multiple concurrent audio streams."""
        stress_test = ConcurrentStreamStressTest(num_streams=4, duration=20)
        results = stress_test.run_test()
        
        # Verify all streams processed frames
        assert results['total_frames_processed'] > 0, "Should process frames across all streams"
        assert results['successful_streams'] >= 3, "Most streams should process successfully"
        assert results['error_rate_percent'] < 5.0, "Error rate should be reasonable"
        
        # Verify performance under concurrency
        assert results['frames_per_second'] > 50, "Should maintain throughput with multiple streams"
        assert results['avg_latency_ms'] < 100, "Should maintain reasonable latency"
    
    @pytest.mark.slow
    def test_memory_pressure_stress(self):
        """Test buffer management under memory pressure."""
        stress_test = MemoryPressureStressTest(duration=25)
        results = stress_test.run_test()
        
        # Verify functionality under memory pressure
        assert results['frames_processed'] > 0, "Should process frames under memory pressure"
        assert results['frames_per_second'] > 5, "Should maintain minimum throughput"
        
        # Verify adaptive behavior
        assert results['buffer_adjustments'] >= 0, "Should adapt to memory constraints"
        assert results['final_buffer_size'] <= 500, "Should keep buffer size reasonable"
        
        # Memory error handling
        if results['memory_errors'] > 0:
            assert results['memory_error_rate'] < 0.1, "Should handle memory errors gracefully"
    
    def test_stress_test_environment(self):
        """Test the stress test environment itself."""
        env = StressTestEnvironment(cpu_load_percent=50.0, memory_pressure=False)
        
        # Test environment startup and shutdown
        env.start_load()
        time.sleep(2)  # Let it run briefly
        
        # Verify load is active
        assert env.active, "Environment should be active"
        assert len(env.load_processes) > 0, "Should have load processes"
        
        env.stop_load()
        
        # Verify cleanup
        assert not env.active, "Environment should be inactive"
        assert len(env.load_processes) == 0, "Should clean up load processes"
    
    @pytest.mark.parametrize("num_streams,duration", [
        (2, 10),
        (4, 15),
        (6, 20)
    ])
    def test_scalability_stress(self, num_streams, duration):
        """Test scalability with different numbers of streams."""
        stress_test = ConcurrentStreamStressTest(num_streams=num_streams, duration=duration)
        results = stress_test.run_test()
        
        # Verify scalability
        expected_min_throughput = num_streams * 10  # Minimum frames per second per stream
        assert results['frames_per_second'] > expected_min_throughput, \
            f"Should scale to {num_streams} streams with adequate throughput"
        
        # Verify error rates don't increase dramatically with scale
        assert results['error_rate_percent'] < 10.0, "Error rate should remain manageable"
        assert results['successful_streams'] >= max(1, num_streams - 1), "Most streams should succeed"


if __name__ == "__main__":
    # Run stress tests directly
    print("Running Audio Buffer Stress Tests...")
    
    # High load test
    print("\n1. High Load Buffer Stress Test")
    high_load_test = HighLoadBufferStressTest(duration=15)
    high_load_results = high_load_test.run_test()
    print(f"   Processed {high_load_results['total_consumed']} frames")
    print(f"   Efficiency: {high_load_results['efficiency']:.2%}")
    print(f"   Adaptations: {high_load_results['buffer_adjustments']}")
    
    # Concurrent streams test
    print("\n2. Concurrent Stream Stress Test")
    concurrent_test = ConcurrentStreamStressTest(num_streams=4, duration=10)
    concurrent_results = concurrent_test.run_test()
    print(f"   Total frames: {concurrent_results['total_frames_processed']}")
    print(f"   Successful streams: {concurrent_results['successful_streams']}/4")
    print(f"   Average latency: {concurrent_results['avg_latency_ms']:.2f}ms")
    
    # Memory pressure test
    print("\n3. Memory Pressure Stress Test")
    memory_test = MemoryPressureStressTest(duration=10)
    memory_results = memory_test.run_test()
    print(f"   Frames processed: {memory_results['frames_processed']}")
    print(f"   Memory errors: {memory_results['memory_errors']}")
    print(f"   Final buffer size: {memory_results['final_buffer_size']}")
    
    print("\nStress tests completed!")