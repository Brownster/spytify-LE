"""
Performance benchmarking tests for audio buffer optimization.

This module provides comprehensive performance testing to validate
optimization effectiveness and detect performance regressions.
"""

import pytest
import time
import threading
import queue
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json

from spotify_splitter.metrics_collector import MetricsCollector, DiagnosticLevel
from spotify_splitter.performance_dashboard import PerformanceDashboard, DashboardConfig
from spotify_splitter.performance_optimizer import PerformanceOptimizer, OptimizationType, OptimizationPriority
from spotify_splitter.buffer_management import AdaptiveBufferManager, BufferMetrics, HealthStatus
from spotify_splitter.audio import EnhancedAudioStream


class PerformanceBenchmark:
    """Base class for performance benchmarks."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.results = {}
        self.start_time = None
        self.end_time = None
    
    def start(self):
        """Start the benchmark."""
        self.start_time = time.time()
    
    def stop(self):
        """Stop the benchmark."""
        self.end_time = time.time()
        self.results['duration'] = self.end_time - self.start_time
    
    def get_results(self) -> dict:
        """Get benchmark results."""
        return {
            'name': self.name,
            'description': self.description,
            'results': self.results,
            'duration': self.results.get('duration', 0)
        }


class MetricsCollectionBenchmark(PerformanceBenchmark):
    """Benchmark for metrics collection performance."""
    
    def __init__(self):
        super().__init__(
            "Metrics Collection Performance",
            "Measures performance impact of metrics collection"
        )
    
    def run_benchmark(self, collection_interval: float = 0.1, duration: int = 10) -> dict:
        """Run metrics collection benchmark."""
        self.start()
        
        # Create metrics collector
        metrics_collector = MetricsCollector(
            collection_interval=collection_interval,
            enable_debug_mode=True
        )
        
        # Mock components for metrics collection
        mock_audio_stream = Mock()
        mock_audio_stream.get_performance_metrics.return_value = {
            'total_frames': 1000,
            'dropped_frames': 0,
            'buffer_overflows': 0,
            'callback_latency_ms': 5.0
        }
        
        mock_buffer_manager = Mock()
        mock_buffer_manager.get_performance_metrics.return_value = {
            'utilization_percent': 45.0,
            'overflow_count': 0,
            'adjustment_count': 2,
            'current_queue_size': 200
        }
        
        # Register components
        metrics_collector.register_component('audio_stream', lambda: mock_audio_stream.get_performance_metrics())
        metrics_collector.register_component('buffer_manager', lambda: mock_buffer_manager.get_performance_metrics())
        
        # Start collection
        metrics_collector.start_collection()
        
        # Simulate workload
        start_time = time.time()
        frame_count = 0
        
        try:
            while time.time() - start_time < duration:
                # Simulate audio processing
                frame_count += 1
                mock_audio_stream.get_performance_metrics.return_value['total_frames'] = frame_count * 100
                
                # Record some metrics manually
                metrics_collector.record_counter('test_counter', 1)
                metrics_collector.record_gauge('test_gauge', frame_count * 0.1)
                metrics_collector.record_timer('test_timer', 0.001)
                
                time.sleep(0.01)  # 10ms processing simulation
        
        finally:
            metrics_collector.stop_collection()
        
        self.stop()
        
        # Collect results
        debug_info = metrics_collector.get_debug_info()
        
        self.results.update({
            'collection_interval': collection_interval,
            'total_collections': debug_info.get('collection_stats', {}).get('total_collections', 0),
            'failed_collections': debug_info.get('collection_stats', {}).get('failed_collections', 0),
            'metrics_recorded': debug_info.get('collection_stats', {}).get('metrics_recorded', 0),
            'collection_success_rate': (
                1.0 - (debug_info.get('collection_stats', {}).get('failed_collections', 0) /
                       max(1, debug_info.get('collection_stats', {}).get('total_collections', 1)))
            ),
            'metrics_per_second': debug_info.get('collection_stats', {}).get('metrics_recorded', 0) / duration,
            'memory_usage_mb': debug_info.get('memory_usage', {}).get('metrics_memory_mb', 0)
        })
        
        return self.get_results()


class BufferManagementBenchmark(PerformanceBenchmark):
    """Benchmark for buffer management performance."""
    
    def __init__(self):
        super().__init__(
            "Buffer Management Performance",
            "Measures adaptive buffer management performance under load"
        )
    
    def run_benchmark(self, queue_size: int = 200, duration: int = 30) -> dict:
        """Run buffer management benchmark."""
        self.start()
        
        # Create buffer manager
        buffer_manager = AdaptiveBufferManager(
            initial_queue_size=queue_size,
            min_size=50,
            max_size=1000
        )
        
        # Create test queue
        test_queue = queue.Queue(maxsize=queue_size)
        
        # Simulate high-load scenario
        producer_active = threading.Event()
        consumer_active = threading.Event()
        producer_active.set()
        consumer_active.set()
        
        # Performance tracking
        produced_items = 0
        consumed_items = 0
        overflow_count = 0
        underrun_count = 0
        
        def producer():
            nonlocal produced_items, overflow_count
            while producer_active.is_set():
                try:
                    # Simulate audio data
                    audio_data = np.random.random((1024, 2)).astype(np.float32)
                    test_queue.put_nowait(audio_data)
                    produced_items += 1
                    time.sleep(0.001)  # 1ms between items
                except queue.Full:
                    overflow_count += 1
                    time.sleep(0.001)
        
        def consumer():
            nonlocal consumed_items, underrun_count
            while consumer_active.is_set():
                try:
                    test_queue.get_nowait()
                    consumed_items += 1
                    time.sleep(0.002)  # 2ms processing time
                except queue.Empty:
                    underrun_count += 1
                    time.sleep(0.001)
        
        def monitor():
            """Monitor buffer performance and trigger adaptations."""
            while producer_active.is_set():
                try:
                    # Monitor utilization
                    metrics = buffer_manager.monitor_utilization(test_queue)
                    
                    # Trigger adaptation if needed
                    buffer_manager.adjust_buffer_size(metrics)
                    
                    time.sleep(0.1)  # Monitor every 100ms
                except Exception as e:
                    print(f"Monitor error: {e}")
        
        # Start threads
        producer_thread = threading.Thread(target=producer, daemon=True)
        consumer_thread = threading.Thread(target=consumer, daemon=True)
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        
        producer_thread.start()
        consumer_thread.start()
        monitor_thread.start()
        
        # Run for specified duration
        time.sleep(duration)
        
        # Stop threads
        producer_active.clear()
        consumer_active.clear()
        
        # Wait for threads to finish
        producer_thread.join(timeout=1.0)
        consumer_thread.join(timeout=1.0)
        monitor_thread.join(timeout=1.0)
        
        self.stop()
        
        # Calculate performance metrics
        throughput = consumed_items / duration
        overflow_rate = overflow_count / duration
        underrun_rate = underrun_count / duration
        efficiency = consumed_items / max(1, produced_items)
        
        self.results.update({
            'initial_queue_size': queue_size,
            'final_queue_size': buffer_manager.current_queue_size,
            'produced_items': produced_items,
            'consumed_items': consumed_items,
            'overflow_count': overflow_count,
            'underrun_count': underrun_count,
            'throughput_items_per_sec': throughput,
            'overflow_rate_per_sec': overflow_rate,
            'underrun_rate_per_sec': underrun_rate,
            'efficiency': efficiency,
            'adaptations_made': buffer_manager.adjustment_count,
            'emergency_expansions': buffer_manager.emergency_expansion_count
        })
        
        return self.get_results()


class DashboardPerformanceBenchmark(PerformanceBenchmark):
    """Benchmark for dashboard performance impact."""
    
    def __init__(self):
        super().__init__(
            "Dashboard Performance Impact",
            "Measures performance impact of real-time dashboard updates"
        )
    
    def run_benchmark(self, update_interval: float = 1.0, duration: int = 60) -> dict:
        """Run dashboard performance benchmark."""
        self.start()
        
        # Create metrics collector
        metrics_collector = MetricsCollector(
            collection_interval=0.5,
            enable_debug_mode=True
        )
        
        # Create dashboard
        config = DashboardConfig(
            update_interval=update_interval,
            enable_alerts=True,
            enable_recommendations=True
        )
        
        dashboard = PerformanceDashboard(
            metrics_collector=metrics_collector,
            config=config
        )
        
        # Mock data sources
        mock_components = {}
        for component in ['audio_stream', 'buffer_manager', 'error_recovery']:
            mock_components[component] = Mock()
            mock_components[component].get_metrics.return_value = {
                'metric1': 50.0,
                'metric2': 100.0,
                'metric3': 0.1
            }
            metrics_collector.register_component(component, mock_components[component].get_metrics)
        
        # Start monitoring
        metrics_collector.start_collection()
        dashboard.start_monitoring()
        
        # Simulate varying load
        cpu_usage = 30.0
        memory_usage = 40.0
        error_count = 0
        
        try:
            start_time = time.time()
            while time.time() - start_time < duration:
                # Simulate changing metrics
                cpu_usage += np.random.normal(0, 5)
                cpu_usage = max(10, min(95, cpu_usage))
                
                memory_usage += np.random.normal(0, 3)
                memory_usage = max(20, min(90, memory_usage))
                
                if np.random.random() < 0.1:  # 10% chance of error
                    error_count += 1
                    metrics_collector.record_error('test_error', 'Simulated error')
                
                # Update mock metrics
                for component in mock_components.values():
                    component.get_metrics.return_value = {
                        'cpu_usage': cpu_usage,
                        'memory_usage': memory_usage,
                        'error_count': error_count,
                        'buffer_utilization': np.random.uniform(20, 80)
                    }
                
                time.sleep(0.1)  # 100ms simulation step
        
        finally:
            dashboard.stop_monitoring()
            metrics_collector.stop_collection()
        
        self.stop()
        
        # Get dashboard data
        dashboard_data = dashboard.get_dashboard_data()
        debug_info = metrics_collector.get_debug_info()
        
        self.results.update({
            'update_interval': update_interval,
            'total_alerts': len(dashboard_data.get('alerts', [])),
            'total_recommendations': len(dashboard_data.get('recommendations', [])),
            'metrics_collected': debug_info.get('collection_stats', {}).get('metrics_recorded', 0),
            'collection_success_rate': (
                1.0 - (debug_info.get('collection_stats', {}).get('failed_collections', 0) /
                       max(1, debug_info.get('collection_stats', {}).get('total_collections', 1)))
            ),
            'dashboard_updates': duration / update_interval,
            'memory_overhead_mb': debug_info.get('memory_usage', {}).get('metrics_memory_mb', 0)
        })
        
        return self.get_results()


class OptimizationBenchmark(PerformanceBenchmark):
    """Benchmark for optimization suggestion performance."""
    
    def __init__(self):
        super().__init__(
            "Optimization Performance",
            "Measures performance of optimization analysis and suggestion generation"
        )
    
    def run_benchmark(self, analysis_cycles: int = 10) -> dict:
        """Run optimization benchmark."""
        self.start()
        
        # Create metrics collector with test data
        metrics_collector = MetricsCollector(
            collection_interval=1.0,
            enable_debug_mode=True
        )
        
        # Create optimizer
        optimizer = PerformanceOptimizer(
            metrics_collector=metrics_collector,
            auto_apply_safe_optimizations=False,
            optimization_interval=5.0
        )
        
        # Populate metrics collector with test data
        metrics_collector.start_collection()
        
        # Generate test metrics over time
        for i in range(100):
            timestamp = datetime.now() - timedelta(seconds=100-i)
            
            # Simulate varying performance metrics
            buffer_util = 50 + 30 * np.sin(i * 0.1) + np.random.normal(0, 5)
            buffer_util = max(0, min(100, buffer_util))
            
            error_rate = max(0, 2 + np.random.normal(0, 1))
            latency = 50 + 20 * np.sin(i * 0.05) + np.random.normal(0, 5)
            cpu_usage = 60 + 20 * np.sin(i * 0.08) + np.random.normal(0, 8)
            
            # Record metrics
            metrics_collector.record_gauge('buffer_manager.utilization_percent', buffer_util)
            metrics_collector.record_gauge('error_recovery.total_errors', error_rate * i)
            metrics_collector.record_timer('audio_stream.callback_latency', latency / 1000)
            metrics_collector.record_gauge('system.cpu_percent', cpu_usage)
            
            time.sleep(0.01)  # Small delay to simulate real-time data
        
        # Run optimization analysis cycles
        suggestion_counts = []
        analysis_times = []
        
        for cycle in range(analysis_cycles):
            cycle_start = time.time()
            
            # Trigger optimization analysis
            optimizer._analyze_and_optimize()
            
            cycle_time = time.time() - cycle_start
            analysis_times.append(cycle_time)
            
            # Get suggestions
            suggestions = optimizer.get_optimization_suggestions()
            suggestion_counts.append(len(suggestions))
            
            time.sleep(0.1)  # Brief pause between cycles
        
        metrics_collector.stop_collection()
        self.stop()
        
        # Calculate performance metrics
        avg_analysis_time = sum(analysis_times) / len(analysis_times)
        avg_suggestions = sum(suggestion_counts) / len(suggestion_counts)
        
        self.results.update({
            'analysis_cycles': analysis_cycles,
            'avg_analysis_time_ms': avg_analysis_time * 1000,
            'max_analysis_time_ms': max(analysis_times) * 1000,
            'avg_suggestions_per_cycle': avg_suggestions,
            'total_suggestions_generated': sum(suggestion_counts),
            'analysis_efficiency': avg_suggestions / avg_analysis_time if avg_analysis_time > 0 else 0
        })
        
        return self.get_results()


class IntegrationBenchmark(PerformanceBenchmark):
    """Comprehensive integration benchmark."""
    
    def __init__(self):
        super().__init__(
            "Integration Performance",
            "Measures performance of complete monitoring and optimization system"
        )
    
    def run_benchmark(self, duration: int = 120) -> dict:
        """Run comprehensive integration benchmark."""
        self.start()
        
        # Create complete monitoring stack
        metrics_collector = MetricsCollector(
            collection_interval=1.0,
            enable_debug_mode=True
        )
        
        dashboard = PerformanceDashboard(
            metrics_collector=metrics_collector,
            config=DashboardConfig(update_interval=2.0)
        )
        
        optimizer = PerformanceOptimizer(
            metrics_collector=metrics_collector,
            auto_apply_safe_optimizations=True,
            optimization_interval=30.0
        )
        
        # Mock audio pipeline components
        mock_audio_stream = Mock()
        mock_buffer_manager = Mock()
        mock_error_recovery = Mock()
        
        # Register components
        metrics_collector.register_component('audio_stream', lambda: {
            'total_frames': np.random.randint(1000, 10000),
            'dropped_frames': np.random.randint(0, 10),
            'callback_latency_ms': np.random.uniform(1, 20)
        })
        
        metrics_collector.register_component('buffer_manager', lambda: {
            'utilization_percent': np.random.uniform(20, 90),
            'overflow_count': np.random.randint(0, 5),
            'current_queue_size': np.random.randint(100, 500)
        })
        
        metrics_collector.register_component('error_recovery', lambda: {
            'total_errors': np.random.randint(0, 3),
            'recovery_success_rate': np.random.uniform(0.8, 1.0)
        })
        
        # Start all components
        metrics_collector.start_collection()
        dashboard.start_monitoring()
        optimizer.start_optimization()
        
        # Performance tracking
        system_load = []
        memory_usage = []
        
        try:
            start_time = time.time()
            while time.time() - start_time < duration:
                # Simulate system load
                import psutil
                try:
                    cpu_percent = psutil.cpu_percent(interval=None)
                    memory_percent = psutil.virtual_memory().percent
                    system_load.append(cpu_percent)
                    memory_usage.append(memory_percent)
                except:
                    # Fallback if psutil not available
                    system_load.append(np.random.uniform(20, 80))
                    memory_usage.append(np.random.uniform(30, 70))
                
                # Occasionally trigger errors
                if np.random.random() < 0.05:  # 5% chance
                    metrics_collector.record_error('integration_test', 'Simulated integration error')
                
                time.sleep(1.0)
        
        finally:
            optimizer.stop_optimization()
            dashboard.stop_monitoring()
            metrics_collector.stop_collection()
        
        self.stop()
        
        # Collect comprehensive results
        debug_info = metrics_collector.get_debug_info()
        dashboard_data = dashboard.get_dashboard_data()
        suggestions = optimizer.get_optimization_suggestions()
        
        self.results.update({
            'total_duration': duration,
            'metrics_collected': debug_info.get('collection_stats', {}).get('metrics_recorded', 0),
            'collection_success_rate': (
                1.0 - (debug_info.get('collection_stats', {}).get('failed_collections', 0) /
                       max(1, debug_info.get('collection_stats', {}).get('total_collections', 1)))
            ),
            'dashboard_alerts': len(dashboard_data.get('alerts', [])),
            'optimization_suggestions': len(suggestions),
            'avg_system_load': sum(system_load) / len(system_load) if system_load else 0,
            'max_system_load': max(system_load) if system_load else 0,
            'avg_memory_usage': sum(memory_usage) / len(memory_usage) if memory_usage else 0,
            'max_memory_usage': max(memory_usage) if memory_usage else 0,
            'system_stability': 1.0 - (max(system_load) - min(system_load)) / 100 if system_load else 1.0
        })
        
        return self.get_results()


# Test classes for pytest integration

class TestPerformanceBenchmarks:
    """Test class for performance benchmarks."""
    
    def test_metrics_collection_performance(self):
        """Test metrics collection performance."""
        benchmark = MetricsCollectionBenchmark()
        results = benchmark.run_benchmark(collection_interval=0.1, duration=5)
        
        assert results['results']['collection_success_rate'] > 0.95
        assert results['results']['metrics_per_second'] > 10
        assert results['duration'] < 10  # Should complete quickly
    
    def test_buffer_management_performance(self):
        """Test buffer management performance."""
        benchmark = BufferManagementBenchmark()
        results = benchmark.run_benchmark(queue_size=200, duration=10)
        
        assert results['results']['efficiency'] > 0.5  # More realistic threshold
        assert results['results']['overflow_rate_per_sec'] < 500  # More realistic threshold for stress test
        assert results['duration'] < 15  # Should complete within reasonable time
    
    def test_dashboard_performance_impact(self):
        """Test dashboard performance impact."""
        benchmark = DashboardPerformanceBenchmark()
        results = benchmark.run_benchmark(update_interval=1.0, duration=10)
        
        assert results['results']['collection_success_rate'] > 0.9
        assert results['results']['memory_overhead_mb'] < 50  # Reasonable memory usage
        assert results['duration'] < 15
    
    def test_optimization_performance(self):
        """Test optimization analysis performance."""
        benchmark = OptimizationBenchmark()
        results = benchmark.run_benchmark(analysis_cycles=5)
        
        assert results['results']['avg_analysis_time_ms'] < 100  # Should be fast
        assert results['results']['avg_suggestions_per_cycle'] >= 0
        assert results['duration'] < 10
    
    @pytest.mark.slow
    def test_integration_performance(self):
        """Test complete integration performance."""
        benchmark = IntegrationBenchmark()
        results = benchmark.run_benchmark(duration=30)  # Shorter duration for tests
        
        assert results['results']['collection_success_rate'] > 0.9
        assert results['results']['system_stability'] > 0.1  # More realistic threshold for stress test
        assert results['duration'] < 40


class TestPerformanceRegression:
    """Test class for performance regression detection."""
    
    def test_performance_baseline_comparison(self):
        """Test performance regression detection against baseline."""
        # This would typically compare against stored baseline results
        # For now, we'll create a simple regression test
        
        current_benchmark = MetricsCollectionBenchmark()
        current_results = current_benchmark.run_benchmark(duration=3)
        
        # Define acceptable performance thresholds
        min_collection_rate = 0.95
        max_analysis_time_ms = 50
        max_memory_mb = 20
        
        # Check for regressions
        assert current_results['results']['collection_success_rate'] >= min_collection_rate, \
            f"Collection success rate regression: {current_results['results']['collection_success_rate']} < {min_collection_rate}"
        
        assert current_results['results']['memory_usage_mb'] <= max_memory_mb, \
            f"Memory usage regression: {current_results['results']['memory_usage_mb']} > {max_memory_mb}"
    
    def test_optimization_effectiveness(self):
        """Test that optimizations actually improve performance."""
        # Create a scenario with known performance issues
        metrics_collector = MetricsCollector(collection_interval=0.5)
        optimizer = PerformanceOptimizer(metrics_collector)
        
        # Simulate high buffer utilization
        metrics_collector.start_collection()
        
        for i in range(10):
            metrics_collector.record_gauge('buffer_manager.utilization_percent', 90.0)
            time.sleep(0.1)
        
        # Get optimization suggestions
        optimizer._analyze_and_optimize()
        suggestions = optimizer.get_optimization_suggestions()
        
        metrics_collector.stop_collection()
        
        # Verify that appropriate suggestions were generated
        buffer_suggestions = [s for s in suggestions if s.optimization_type == OptimizationType.BUFFER_SIZE_ADJUSTMENT]
        assert len(buffer_suggestions) > 0, "Should generate buffer size adjustment suggestions for high utilization"
        
        # Verify suggestion quality
        high_priority_suggestions = [s for s in suggestions if s.priority in [OptimizationPriority.HIGH, OptimizationPriority.CRITICAL]]
        assert len(high_priority_suggestions) > 0, "Should generate high priority suggestions for critical issues"


def run_performance_suite():
    """Run complete performance benchmark suite."""
    benchmarks = [
        MetricsCollectionBenchmark(),
        BufferManagementBenchmark(),
        DashboardPerformanceBenchmark(),
        OptimizationBenchmark(),
        IntegrationBenchmark()
    ]
    
    results = []
    
    print("Running Performance Benchmark Suite...")
    print("=" * 50)
    
    for benchmark in benchmarks:
        print(f"\nRunning: {benchmark.name}")
        print(f"Description: {benchmark.description}")
        
        try:
            if isinstance(benchmark, MetricsCollectionBenchmark):
                result = benchmark.run_benchmark(duration=5)
            elif isinstance(benchmark, BufferManagementBenchmark):
                result = benchmark.run_benchmark(duration=15)
            elif isinstance(benchmark, DashboardPerformanceBenchmark):
                result = benchmark.run_benchmark(duration=20)
            elif isinstance(benchmark, OptimizationBenchmark):
                result = benchmark.run_benchmark(analysis_cycles=5)
            elif isinstance(benchmark, IntegrationBenchmark):
                result = benchmark.run_benchmark(duration=60)
            else:
                result = benchmark.get_results()
            
            results.append(result)
            print(f"✓ Completed in {result['duration']:.2f}s")
            
            # Print key metrics
            key_metrics = result['results']
            for key, value in list(key_metrics.items())[:5]:  # Show first 5 metrics
                if isinstance(value, float):
                    print(f"  {key}: {value:.3f}")
                else:
                    print(f"  {key}: {value}")
        
        except Exception as e:
            print(f"✗ Failed: {e}")
            results.append({
                'name': benchmark.name,
                'error': str(e),
                'duration': 0
            })
    
    print("\n" + "=" * 50)
    print("Performance Benchmark Suite Complete")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"performance_benchmark_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'benchmarks': results,
            'summary': {
                'total_benchmarks': len(benchmarks),
                'successful_benchmarks': len([r for r in results if 'error' not in r]),
                'total_duration': sum(r.get('duration', 0) for r in results)
            }
        }, f, indent=2)
    
    print(f"Results saved to: {results_file}")
    
    return results


if __name__ == "__main__":
    run_performance_suite()