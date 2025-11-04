#!/usr/bin/env python3
"""
Integration test for buffer health monitoring system.
Tests the complete workflow from buffer management to health monitoring.
"""

import time
from queue import Queue
from spotify_splitter.buffer_management import AdaptiveBufferManager
from spotify_splitter.buffer_health_monitor import BufferHealthMonitor, AlertLevel

def test_integration():
    """Test complete buffer health monitoring integration."""
    print("Starting buffer health monitoring integration test...")
    
    # Create buffer manager
    buffer_manager = AdaptiveBufferManager(
        initial_queue_size=100,
        min_size=50,
        max_size=500,
        adjustment_threshold=0.7,
        emergency_threshold=0.9
    )
    
    # Create alert callback
    alerts_received = []
    def alert_callback(alert):
        alerts_received.append(alert)
        print(f"ALERT: {alert.level.value.upper()} - {alert.message}")
    
    # Create health monitor
    monitor = BufferHealthMonitor(
        buffer_manager=buffer_manager,
        monitoring_interval=0.2,
        alert_callback=alert_callback
    )
    
    # Create test queue
    audio_queue = Queue(maxsize=100)
    
    try:
        # Start monitoring
        monitor.start_monitoring(audio_queue)
        print("Monitoring started...")
        
        # Test 1: Normal operation (low utilization)
        print("\nTest 1: Normal operation")
        for i in range(20):
            audio_queue.put(f"audio_data_{i}")
        time.sleep(0.5)
        
        current_health = monitor.get_current_health()
        print(f"Current health status: {current_health.status.value}")
        print(f"Current utilization: {current_health.utilization:.1%}")
        
        # Test 2: High utilization (should trigger warning)
        print("\nTest 2: High utilization")
        for i in range(60):  # Fill to ~80%
            audio_queue.put(f"audio_data_high_{i}")
        time.sleep(0.8)
        
        current_health = monitor.get_current_health()
        print(f"High load health status: {current_health.status.value}")
        print(f"High load utilization: {current_health.utilization:.1%}")
        
        # Test 3: Critical utilization (should trigger critical alert)
        print("\nTest 3: Critical utilization")
        for i in range(15):  # Fill to ~95%
            audio_queue.put(f"audio_data_critical_{i}")
        time.sleep(0.8)
        
        current_health = monitor.get_current_health()
        print(f"Critical health status: {current_health.status.value}")
        print(f"Critical utilization: {current_health.utilization:.1%}")
        
        # Generate health report
        print("\nGenerating health report...")
        report = monitor.generate_health_report()
        print(f"Report timestamp: {report.timestamp}")
        print(f"Total health checks: {report.performance_summary['total_checks']}")
        print(f"Alerts generated: {report.performance_summary['alerts_sent']}")
        print(f"Recent alerts: {len(report.alerts)}")
        
        if report.trend_analysis.get("status") == "analyzed":
            trend = report.trend_analysis["utilization_trend"]
            print(f"Utilization trend: {trend['direction']}")
            print(f"Average utilization: {trend['average']:.1%}")
        
        # Test buffer manager integration
        print("\nTesting buffer manager integration...")
        stats = buffer_manager.get_stats()
        print(f"Current buffer size: {stats['current_queue_size']}")
        print(f"Average utilization: {stats['average_utilization']:.1f}%")
        
        print(f"\nTotal alerts received: {len(alerts_received)}")
        for alert in alerts_received:
            print(f"  - {alert.level.value}: {alert.message}")
        
        print("\nIntegration test completed successfully!")
        
    finally:
        # Clean up
        monitor.stop_monitoring()
        print("Monitoring stopped.")

if __name__ == "__main__":
    test_integration()