"""
Tests for the MetricsCollector class and related functionality.

Tests cover metrics collection, performance tracking, error diagnostics,
and debug mode functionality.
"""

import json
import tempfile
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from spotify_splitter.metrics_collector import (
    MetricsCollector,
    MetricType,
    DiagnosticLevel,
    MetricValue,
    PerformanceSnapshot,
    DiagnosticReport,
    TimerContext,
    timer
)


class TestMetricValue:
    """Test MetricValue data class."""
    
    def test_metric_value_creation(self):
        """Test creating a MetricValue."""
        timestamp = datetime.now()
        value = MetricValue(timestamp=timestamp, value=42.5, labels={'type': 'test'})
        
        assert value.timestamp == timestamp
        assert value.value == 42.5
        assert value.labels == {'type': 'test'}
    
    def test_metric_value_to_dict(self):
        """Test converting MetricValue to dictionary."""
        timestamp = datetime.now()
        value = MetricValue(timestamp=timestamp, value=100, labels={'component': 'audio'})
        
        result = value.to_dict()
        
        assert result['timestamp'] == timestamp.isoformat()
        assert result['value'] == 100
        assert result['labels'] == {'component': 'audio'}


class TestMetricsCollector:
    """Test MetricsCollector functionality."""
    
    @pytest.fixture
    def collector(self):
        """Create a MetricsCollector for testing."""
        return MetricsCollector(
            collection_interval=0.1,  # Fast collection for testing
            history_size=100,
            enable_debug_mode=False
        )
    
    @pytest.fixture
    def debug_collector(self):
        """Create a MetricsCollector with debug mode enabled."""
        return MetricsCollector(
            collection_interval=0.1,
            history_size=100,
            enable_debug_mode=True
        )
    
    def test_collector_initialization(self):
        """Test MetricsCollector initialization."""
        collector = MetricsCollector(
            collection_interval=2.0,
            history_size=500,
            enable_debug_mode=True
        )
        
        assert collector.collection_interval == 2.0
        assert collector.history_size == 500
        assert collector.enable_debug_mode is True
        assert not collector._collecting
        assert len(collector.metrics) == 0
        assert len(collector.performance_snapshots) == 0
    
    def test_counter_recording(self, collector):
        """Test recording counter metrics."""
        collector.record_counter('test_counter', 5)
        collector.record_counter('test_counter', 3)
        
        assert collector.counters['test_counter'] == 8
        
        # Check metric values
        values = collector.get_metric_values('test_counter')
        assert len(values) == 2
        assert values[0].value == 5
        assert values[1].value == 8
    
    def test_gauge_recording(self, collector):
        """Test recording gauge metrics."""
        collector.record_gauge('cpu_usage', 45.2)
        collector.record_gauge('cpu_usage', 52.8)
        
        assert collector.gauges['cpu_usage'] == 52.8
        
        values = collector.get_metric_values('cpu_usage')
        assert len(values) == 2
        assert values[0].value == 45.2
        assert values[1].value == 52.8
    
    def test_timer_recording(self, collector):
        """Test recording timer metrics."""
        collector.record_timer('operation_duration', 0.125)
        collector.record_timer('operation_duration', 0.089)
        collector.record_timer('operation_duration', 0.156)
        
        assert len(collector.timers['operation_duration']) == 3
        
        stats = collector.get_timer_statistics('operation_duration')
        assert stats['count'] == 3
        assert stats['min'] == 0.089
        assert stats['max'] == 0.156
        assert abs(stats['average'] - 0.123333) < 0.001
    
    def test_error_recording(self, collector):
        """Test recording error metrics."""
        collector.record_error('buffer_overflow', 'Queue is full', {'queue_size': 200})
        collector.record_error('buffer_overflow', 'Queue is full again')
        collector.record_error('stream_error', 'Connection lost')
        
        assert collector.error_counts['buffer_overflow'] == 2
        assert collector.error_counts['stream_error'] == 1
        
        error_summary = collector.get_error_summary()
        assert error_summary['total_errors'] == 3
        assert error_summary['unique_error_types'] == 2
        assert error_summary['error_type_counts']['buffer_overflow'] == 2
    
    def test_component_registration(self, collector):
        """Test registering components for metrics collection."""
        def mock_audio_metrics():
            return {
                'frames_processed': 1000,
                'buffer_utilization': 0.75,
                'latency_ms': 45.2
            }
        
        def mock_buffer_metrics():
            return {
                'queue_size': 150,
                'overflow_count': 2
            }
        
        collector.register_component('audio_stream', mock_audio_metrics)
        collector.register_component('buffer_manager', mock_buffer_metrics)
        
        assert 'audio_stream' in collector._registered_components
        assert 'buffer_manager' in collector._registered_components
        
        # Test unregistration
        collector.unregister_component('audio_stream')
        assert 'audio_stream' not in collector._registered_components
    
    def test_metric_summary(self, collector):
        """Test getting metric summaries."""
        # Record some test data
        for i in range(10):
            collector.record_gauge('test_metric', i * 10)
        
        summary = collector.get_metric_summary('test_metric')
        
        assert summary['count'] == 10
        assert summary['min'] == 0
        assert summary['max'] == 90
        assert summary['average'] == 45.0
        assert summary['latest'] == 90
    
    def test_metric_summary_with_time_filter(self, collector):
        """Test getting metric summaries with time filtering."""
        # Record metrics with different timestamps
        now = datetime.now()
        old_time = now - timedelta(minutes=10)
        
        # Manually add old metric
        old_metric = MetricValue(timestamp=old_time, value=100)
        collector.metrics['test_metric'].append(old_metric)
        
        # Add recent metric
        collector.record_gauge('test_metric', 200)
        
        # Get summary since 5 minutes ago
        since = now - timedelta(minutes=5)
        summary = collector.get_metric_summary('test_metric', since=since)
        
        assert summary['count'] == 1  # Only recent metric
        assert summary['latest'] == 200
    
    def test_timer_statistics(self, collector):
        """Test timer statistics calculation."""
        # Record timer values
        durations = [0.1, 0.2, 0.15, 0.3, 0.25, 0.18, 0.22, 0.12, 0.28, 0.16]
        for duration in durations:
            collector.record_timer('test_timer', duration)
        
        stats = collector.get_timer_statistics('test_timer')
        
        assert stats['count'] == 10
        assert stats['min'] == 0.1
        assert stats['max'] == 0.3
        assert abs(stats['average'] - 0.196) < 0.001
        # Median of sorted values: [0.1, 0.12, 0.15, 0.16, 0.18, 0.2, 0.22, 0.25, 0.28, 0.3]
        # With 10 values, median is average of 5th and 6th values: (0.18 + 0.2) / 2 = 0.19
        assert abs(stats['median'] - 0.19) < 0.01
    
    def test_automatic_collection(self, collector):
        """Test automatic metrics collection."""
        # Register a mock component
        call_count = 0
        
        def mock_metrics():
            nonlocal call_count
            call_count += 1
            return {'call_count': call_count}
        
        collector.register_component('test_component', mock_metrics)
        
        # Start collection
        collector.start_collection()
        
        try:
            # Wait for a few collection cycles
            time.sleep(0.3)
            
            # Check that metrics were collected
            assert collector.collection_stats['total_collections'] > 0
            assert len(collector.performance_snapshots) > 0
            
            # Check that component was called
            values = collector.get_metric_values('test_component.call_count')
            assert len(values) > 0
            
        finally:
            collector.stop_collection()
    
    def test_collection_start_stop(self, collector):
        """Test starting and stopping collection."""
        assert not collector._collecting
        
        collector.start_collection()
        assert collector._collecting
        assert collector._collection_thread is not None
        
        collector.stop_collection()
        assert not collector._collecting
    
    def test_double_start_collection(self, collector):
        """Test that starting collection twice doesn't cause issues."""
        collector.start_collection()
        
        # Starting again should log warning but not crash
        collector.start_collection()
        
        collector.stop_collection()
    
    def test_system_metrics_collection(self, debug_collector):
        """Test system metrics collection in debug mode."""
        # Mock psutil at the module level where it's imported
        with patch('psutil.cpu_percent', return_value=45.2), \
             patch('psutil.virtual_memory', return_value=Mock(percent=60.5, available=2048*1024*1024)), \
             patch('psutil.Process') as mock_process_class:
            
            # Mock process instance
            mock_process = Mock()
            mock_process.memory_info.return_value = Mock(rss=100*1024*1024, vms=200*1024*1024)
            mock_process.cpu_percent.return_value = 25.3
            mock_process.num_threads.return_value = 8
            mock_process_class.return_value = mock_process
            
            # Trigger system metrics collection
            debug_collector._collect_system_metrics(datetime.now())
            
            # Check that system metrics were recorded
            assert len(debug_collector.get_metric_values('system.cpu_percent')) == 1
            assert len(debug_collector.get_metric_values('system.memory_percent')) == 1
            assert len(debug_collector.get_metric_values('process.memory_rss_mb')) == 1
    
    def test_diagnostic_report_generation(self, collector):
        """Test generating diagnostic reports."""
        # Add some test data
        collector.record_counter('test_counter', 10)
        collector.record_gauge('test_gauge', 75.5)
        collector.record_error('test_error', 'Test error message')
        
        # Generate basic report
        basic_report = collector.generate_diagnostic_report(DiagnosticLevel.BASIC)
        
        assert basic_report.level == DiagnosticLevel.BASIC
        assert isinstance(basic_report.timestamp, datetime)
        assert 'total_metrics' in basic_report.summary
        assert basic_report.error_analysis['total_errors'] == 1
        
        # Generate detailed report
        detailed_report = collector.generate_diagnostic_report(DiagnosticLevel.DETAILED)
        
        assert detailed_report.level == DiagnosticLevel.DETAILED
        assert len(detailed_report.metrics) >= len(basic_report.metrics)
    
    def test_recommendations_generation(self, collector):
        """Test performance recommendations generation."""
        # Create conditions that should trigger recommendations
        
        # High error rate
        for i in range(15):
            collector.record_error('test_error', f'Error {i}')
        
        # Buffer overflows
        for i in range(5):
            collector.record_gauge('buffer_manager.overflow_count', i)
        
        report = collector.generate_diagnostic_report()
        
        assert len(report.recommendations) > 0
        assert any('error rate' in rec.lower() for rec in report.recommendations)
    
    def test_metrics_export(self, collector):
        """Test exporting metrics to file."""
        # Add some test data
        collector.record_counter('export_test', 42)
        collector.record_gauge('export_gauge', 3.14)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            export_path = Path(f.name)
        
        try:
            collector.export_metrics(export_path, 'json')
            
            # Verify file was created and contains data
            assert export_path.exists()
            
            with open(export_path, 'r') as f:
                data = json.load(f)
            
            assert 'timestamp' in data
            assert 'metrics' in data
            assert 'summary' in data
            
        finally:
            export_path.unlink(missing_ok=True)
    
    def test_metrics_clearing(self, collector):
        """Test clearing all metrics."""
        # Add some data
        collector.record_counter('test_counter', 5)
        collector.record_gauge('test_gauge', 10.5)
        collector.record_error('test_error', 'Test error')
        
        assert len(collector.metrics) > 0
        assert len(collector.error_history) > 0
        
        collector.clear_metrics()
        
        assert len(collector.metrics) == 0
        assert len(collector.counters) == 0
        assert len(collector.gauges) == 0
        assert len(collector.error_history) == 0
        assert collector.collection_stats['metrics_recorded'] == 0
    
    def test_debug_info(self, collector):
        """Test getting debug information."""
        collector.register_component('test_comp', lambda: {'value': 1})
        collector.record_counter('test_counter', 1)
        
        debug_info = collector.get_debug_info()
        
        assert 'collecting' in debug_info
        assert 'registered_components' in debug_info
        assert 'test_comp' in debug_info['registered_components']
        assert 'metrics_count' in debug_info
        assert debug_info['metrics_count'] > 0
    
    def test_context_manager(self, collector):
        """Test using MetricsCollector as context manager."""
        with collector as c:
            assert c._collecting
            c.record_counter('context_test', 1)
        
        assert not collector._collecting
        assert len(collector.get_metric_values('context_test')) == 1


class TestTimerContext:
    """Test TimerContext functionality."""
    
    def test_timer_context(self):
        """Test timing operations with context manager."""
        collector = MetricsCollector()
        
        with TimerContext(collector, 'test_operation'):
            time.sleep(0.01)  # Small delay
        
        values = collector.get_metric_values('test_operation')
        assert len(values) == 1
        assert values[0].value >= 0.01  # Should be at least 10ms
    
    def test_timer_context_with_labels(self):
        """Test timer context with labels."""
        collector = MetricsCollector()
        labels = {'operation': 'test', 'component': 'audio'}
        
        with TimerContext(collector, 'labeled_operation', labels):
            time.sleep(0.005)
        
        values = collector.get_metric_values('labeled_operation')
        assert len(values) == 1
        assert values[0].labels == labels
    
    def test_timer_decorator(self):
        """Test timer decorator functionality."""
        collector = MetricsCollector()
        
        @timer(collector, 'decorated_function')
        def test_function():
            time.sleep(0.005)
            return 42
        
        result = test_function()
        
        assert result == 42
        values = collector.get_metric_values('decorated_function')
        assert len(values) == 1
        assert values[0].value >= 0.005


class TestPerformanceSnapshot:
    """Test PerformanceSnapshot functionality."""
    
    def test_snapshot_creation(self):
        """Test creating a performance snapshot."""
        timestamp = datetime.now()
        snapshot = PerformanceSnapshot(
            timestamp=timestamp,
            audio_pipeline={'frames': 1000},
            buffer_management={'utilization': 0.75},
            error_recovery={'errors': 2},
            system_resources={'cpu': 45.2}
        )
        
        assert snapshot.timestamp == timestamp
        assert snapshot.audio_pipeline['frames'] == 1000
        assert snapshot.buffer_management['utilization'] == 0.75
    
    def test_snapshot_to_dict(self):
        """Test converting snapshot to dictionary."""
        timestamp = datetime.now()
        snapshot = PerformanceSnapshot(
            timestamp=timestamp,
            audio_pipeline={'test': 1},
            buffer_management={'test': 2},
            error_recovery={'test': 3},
            system_resources={'test': 4}
        )
        
        result = snapshot.to_dict()
        
        assert result['timestamp'] == timestamp.isoformat()
        assert result['audio_pipeline']['test'] == 1
        assert result['buffer_management']['test'] == 2


class TestDiagnosticReport:
    """Test DiagnosticReport functionality."""
    
    def test_report_creation(self):
        """Test creating a diagnostic report."""
        timestamp = datetime.now()
        metrics = {'test_metric': [MetricValue(timestamp, 42)]}
        snapshots = [PerformanceSnapshot(
            timestamp, {}, {}, {}, {}
        )]
        
        report = DiagnosticReport(
            timestamp=timestamp,
            level=DiagnosticLevel.BASIC,
            summary={'test': 'value'},
            metrics=metrics,
            performance_snapshots=snapshots,
            error_analysis={'errors': 0},
            recommendations=['Test recommendation']
        )
        
        assert report.level == DiagnosticLevel.BASIC
        assert report.summary['test'] == 'value'
        assert len(report.metrics) == 1
        assert len(report.recommendations) == 1
    
    def test_report_to_dict(self):
        """Test converting report to dictionary."""
        timestamp = datetime.now()
        report = DiagnosticReport(
            timestamp=timestamp,
            level=DiagnosticLevel.DETAILED,
            summary={},
            metrics={},
            performance_snapshots=[],
            error_analysis={},
            recommendations=[]
        )
        
        result = report.to_dict()
        
        assert result['timestamp'] == timestamp.isoformat()
        assert result['level'] == 'detailed'
        assert 'summary' in result
        assert 'metrics' in result


class TestIntegration:
    """Integration tests for metrics collection."""
    
    def test_full_metrics_pipeline(self):
        """Test complete metrics collection pipeline."""
        collector = MetricsCollector(collection_interval=0.05, enable_debug_mode=False)
        
        # Register mock components
        def audio_metrics():
            return {
                'frames_processed': 1000,
                'buffer_utilization': 0.65,
                'latency_ms': 25.5
            }
        
        def buffer_metrics():
            return {
                'queue_size': 150,
                'overflow_count': 1,
                'utilization_percent': 75.0
            }
        
        collector.register_component('audio_stream', audio_metrics)
        collector.register_component('buffer_manager', buffer_metrics)
        
        try:
            # Start collection
            collector.start_collection()
            
            # Let it collect for a short time
            time.sleep(0.2)
            
            # Add some manual metrics
            collector.record_counter('manual_counter', 5)
            collector.record_error('test_error', 'Integration test error')
            
            # Generate comprehensive report
            report = collector.generate_diagnostic_report(DiagnosticLevel.DETAILED)
            
            # Verify report contents
            assert report.summary['total_metrics'] > 0
            assert len(report.performance_snapshots) > 0
            assert report.error_analysis['total_errors'] >= 1
            
            # Verify component metrics were collected
            audio_frames = collector.get_metric_values('audio_stream.frames_processed')
            buffer_queue = collector.get_metric_values('buffer_manager.queue_size')
            
            assert len(audio_frames) > 0
            assert len(buffer_queue) > 0
            
        finally:
            collector.stop_collection()
    
    def test_error_handling_during_collection(self):
        """Test error handling during metrics collection."""
        collector = MetricsCollector(collection_interval=0.05)
        
        # Register a component that raises an exception
        def failing_metrics():
            raise ValueError("Simulated collection error")
        
        def working_metrics():
            return {'working_metric': 42}
        
        collector.register_component('failing_component', failing_metrics)
        collector.register_component('working_component', working_metrics)
        
        try:
            collector.start_collection()
            time.sleep(0.15)  # Let it try to collect a few times
            
            # Should have recorded errors but continued collecting
            assert collector.collection_stats['failed_collections'] == 0  # Component errors don't fail collection
            assert collector.collection_stats['total_collections'] > 0
            
            # Working component should still have data
            working_values = collector.get_metric_values('working_component.working_metric')
            assert len(working_values) > 0
            
        finally:
            collector.stop_collection()


if __name__ == '__main__':
    pytest.main([__file__])