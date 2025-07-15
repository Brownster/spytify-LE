"""
Debug mode utilities for comprehensive diagnostics and monitoring.

This module provides debug mode functionality with exposed performance metrics,
real-time monitoring displays, and comprehensive diagnostic reporting.
"""

import logging
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List
import json

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich.layout import Layout

from .metrics_collector import MetricsCollector, DiagnosticLevel, DiagnosticReport
from .buffer_management import AdaptiveBufferManager
from .buffer_health_monitor import BufferHealthMonitor
from .error_recovery import ErrorRecoveryManager

logger = logging.getLogger(__name__)


class DebugModeManager:
    """
    Manages debug mode functionality with real-time monitoring and diagnostics.
    
    Provides comprehensive debug information display, metrics monitoring,
    and diagnostic report generation for troubleshooting audio issues.
    """
    
    def __init__(
        self,
        metrics_collector: MetricsCollector,
        buffer_manager: Optional[AdaptiveBufferManager] = None,
        health_monitor: Optional[BufferHealthMonitor] = None,
        error_recovery: Optional[ErrorRecoveryManager] = None,
        update_interval: float = 1.0,
        enable_live_display: bool = True
    ):
        """
        Initialize debug mode manager.
        
        Args:
            metrics_collector: The metrics collector instance
            buffer_manager: Optional buffer manager for detailed metrics
            health_monitor: Optional health monitor for buffer status
            error_recovery: Optional error recovery manager for error stats
            update_interval: Display update interval in seconds
            enable_live_display: Enable real-time display updates
        """
        self.metrics_collector = metrics_collector
        self.buffer_manager = buffer_manager
        self.health_monitor = health_monitor
        self.error_recovery = error_recovery
        self.update_interval = update_interval
        self.enable_live_display = enable_live_display
        
        # Display components
        self.console = Console()
        self.live_display: Optional[Live] = None
        self.display_thread: Optional[threading.Thread] = None
        self.stop_display = threading.Event()
        
        # Debug state
        self.debug_active = False
        self.start_time = None
        self.last_report_time = None
        
        logger.debug("DebugModeManager initialized")
    
    def start_debug_mode(self) -> None:
        """Start debug mode with real-time monitoring."""
        if self.debug_active:
            logger.warning("Debug mode already active")
            return
        
        self.debug_active = True
        self.start_time = datetime.now()
        self.stop_display.clear()
        
        logger.info("Debug mode started")
        
        # Start metrics collection if not already running
        if not self.metrics_collector._collecting:
            self.metrics_collector.start_collection()
        
        # Start live display if enabled
        if self.enable_live_display:
            self._start_live_display()
        
        # Print initial debug information
        self._print_debug_header()
    
    def stop_debug_mode(self) -> None:
        """Stop debug mode and generate final report."""
        if not self.debug_active:
            return
        
        self.debug_active = False
        self.stop_display.set()
        
        # Stop live display
        if self.live_display:
            self.live_display.stop()
            self.live_display = None
        
        # Wait for display thread to finish
        if self.display_thread and self.display_thread.is_alive():
            self.display_thread.join(timeout=2.0)
        
        # Generate final debug report
        self._generate_final_report()
        
        logger.info("Debug mode stopped")
    
    def _start_live_display(self) -> None:
        """Start the live display for real-time monitoring."""
        try:
            layout = self._create_debug_layout()
            self.live_display = Live(
                layout,
                refresh_per_second=1/self.update_interval,
                console=self.console
            )
            
            self.display_thread = threading.Thread(
                target=self._display_loop,
                daemon=True,
                name="DebugDisplay"
            )
            
            self.live_display.start()
            self.display_thread.start()
            
        except Exception as e:
            logger.error("Error starting live display: %s", e)
            self.enable_live_display = False
    
    def _display_loop(self) -> None:
        """Main loop for updating the live display."""
        while not self.stop_display.is_set() and self.live_display:
            try:
                # Update the display layout
                updated_layout = self._create_debug_layout()
                self.live_display.update(updated_layout)
                
                # Wait for next update
                self.stop_display.wait(self.update_interval)
                
            except Exception as e:
                logger.error("Error in display loop: %s", e)
                break
    
    def _create_debug_layout(self) -> Layout:
        """Create the debug information layout."""
        layout = Layout()
        
        # Split into main sections
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3)
        )
        
        # Split main section
        layout["main"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )
        
        # Split left section
        layout["left"].split_column(
            Layout(name="metrics", ratio=2),
            Layout(name="buffer", ratio=1)
        )
        
        # Split right section
        layout["right"].split_column(
            Layout(name="errors", ratio=1),
            Layout(name="system", ratio=1)
        )
        
        # Populate sections
        layout["header"].update(self._create_header_panel())
        layout["metrics"].update(self._create_metrics_panel())
        layout["buffer"].update(self._create_buffer_panel())
        layout["errors"].update(self._create_errors_panel())
        layout["system"].update(self._create_system_panel())
        layout["footer"].update(self._create_footer_panel())
        
        return layout
    
    def _create_header_panel(self) -> Panel:
        """Create the header panel with session information."""
        if self.start_time:
            uptime = datetime.now() - self.start_time
            uptime_str = str(uptime).split('.')[0]  # Remove microseconds
        else:
            uptime_str = "Unknown"
        
        header_text = Text()
        header_text.append("🔍 Spotify Splitter Debug Mode", style="bold cyan")
        header_text.append(f" | Uptime: {uptime_str}", style="dim")
        header_text.append(f" | Updated: {datetime.now().strftime('%H:%M:%S')}", style="dim")
        
        return Panel(header_text, border_style="cyan")
    
    def _create_metrics_panel(self) -> Panel:
        """Create the metrics panel with performance statistics."""
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        table.add_column("Status", style="green")
        
        try:
            # Get metrics from collector
            debug_info = self.metrics_collector.get_debug_info()
            
            # Collection statistics
            table.add_row(
                "Collection Active",
                "✓" if debug_info['collecting'] else "✗",
                "OK" if debug_info['collecting'] else "STOPPED"
            )
            
            table.add_row(
                "Total Metrics",
                str(debug_info['metrics_count']),
                "OK" if debug_info['metrics_count'] > 0 else "NO DATA"
            )
            
            table.add_row(
                "Collection Rate",
                f"{1/self.metrics_collector.collection_interval:.1f}/sec",
                "OK"
            )
            
            # Performance metrics
            collection_stats = debug_info['collection_stats']
            if collection_stats['total_collections'] > 0:
                success_rate = 1 - (collection_stats['failed_collections'] / collection_stats['total_collections'])
                table.add_row(
                    "Collection Success",
                    f"{success_rate:.1%}",
                    "OK" if success_rate > 0.95 else "WARNING"
                )
            
            # Component metrics
            table.add_row(
                "Registered Components",
                str(len(debug_info['registered_components'])),
                "OK" if debug_info['registered_components'] else "NONE"
            )
            
        except Exception as e:
            table.add_row("Error", str(e), "ERROR")
        
        return Panel(table, title="📊 Metrics Collection", border_style="magenta")
    
    def _create_buffer_panel(self) -> Panel:
        """Create the buffer status panel."""
        table = Table(show_header=True, header_style="bold blue")
        table.add_column("Buffer Metric", style="cyan")
        table.add_column("Value", style="white")
        table.add_column("Status", style="green")
        
        try:
            if self.buffer_manager:
                stats = self.buffer_manager.get_stats()
                
                table.add_row(
                    "Current Size",
                    str(stats['current_queue_size']),
                    "OK"
                )
                
                table.add_row(
                    "Avg Utilization",
                    f"{stats['average_utilization']:.1f}%",
                    "OK" if stats['average_utilization'] < 80 else "HIGH"
                )
                
                table.add_row(
                    "Overflows",
                    str(stats['overflow_count']),
                    "OK" if stats['overflow_count'] == 0 else "WARNING"
                )
                
                table.add_row(
                    "Emergency Expansions",
                    str(stats['emergency_expansions']),
                    "OK" if stats['emergency_expansions'] == 0 else "WARNING"
                )
                
            else:
                table.add_row("Buffer Manager", "Not Available", "N/A")
            
            # Health monitor status
            if self.health_monitor:
                health_stats = self.health_monitor.get_statistics()
                table.add_row(
                    "Health Monitoring",
                    "Active" if health_stats['monitoring_active'] else "Inactive",
                    "OK" if health_stats['monitoring_active'] else "OFF"
                )
            
        except Exception as e:
            table.add_row("Error", str(e), "ERROR")
        
        return Panel(table, title="🔧 Buffer Status", border_style="blue")
    
    def _create_errors_panel(self) -> Panel:
        """Create the errors panel with error statistics."""
        table = Table(show_header=True, header_style="bold red")
        table.add_column("Error Metric", style="cyan")
        table.add_column("Value", style="white")
        table.add_column("Status", style="green")
        
        try:
            if self.error_recovery:
                stats = self.error_recovery.get_statistics()
                
                table.add_row(
                    "Total Errors",
                    str(stats['total_errors']),
                    "OK" if stats['total_errors'] == 0 else "WARNING"
                )
                
                table.add_row(
                    "Recent Errors (1h)",
                    str(stats['recent_error_count']),
                    "OK" if stats['recent_error_count'] == 0 else "WARNING"
                )
                
                # Recovery statistics
                recovery_stats = stats.get('recovery_stats', {})
                total_attempts = sum(s.get('attempted', 0) for s in recovery_stats.values())
                total_successful = sum(s.get('successful', 0) for s in recovery_stats.values())
                
                if total_attempts > 0:
                    success_rate = total_successful / total_attempts
                    table.add_row(
                        "Recovery Success",
                        f"{success_rate:.1%}",
                        "OK" if success_rate > 0.8 else "LOW"
                    )
                
            else:
                table.add_row("Error Recovery", "Not Available", "N/A")
            
            # Metrics collector errors
            error_summary = self.metrics_collector.get_error_summary()
            table.add_row(
                "Metrics Errors",
                str(error_summary['total_errors']),
                "OK" if error_summary['total_errors'] == 0 else "WARNING"
            )
            
        except Exception as e:
            table.add_row("Error", str(e), "ERROR")
        
        return Panel(table, title="⚠️  Error Status", border_style="red")
    
    def _create_system_panel(self) -> Panel:
        """Create the system information panel."""
        table = Table(show_header=True, header_style="bold yellow")
        table.add_column("System Metric", style="cyan")
        table.add_column("Value", style="white")
        table.add_column("Status", style="green")
        
        try:
            # Memory usage estimation
            debug_info = self.metrics_collector.get_debug_info()
            memory_info = debug_info.get('memory_usage', {})
            
            if 'metrics_memory_mb' in memory_info:
                table.add_row(
                    "Metrics Memory",
                    f"{memory_info['metrics_memory_mb']:.1f} MB",
                    "OK" if memory_info['metrics_memory_mb'] < 100 else "HIGH"
                )
            
            # Collection performance
            collection_stats = debug_info['collection_stats']
            if collection_stats['total_collections'] > 0:
                table.add_row(
                    "Collections",
                    str(collection_stats['total_collections']),
                    "OK"
                )
                
                table.add_row(
                    "Failed Collections",
                    str(collection_stats['failed_collections']),
                    "OK" if collection_stats['failed_collections'] == 0 else "WARNING"
                )
            
            # Debug mode specific
            table.add_row(
                "Debug Mode",
                "Active",
                "OK"
            )
            
            table.add_row(
                "Live Display",
                "Enabled" if self.enable_live_display else "Disabled",
                "OK"
            )
            
        except Exception as e:
            table.add_row("Error", str(e), "ERROR")
        
        return Panel(table, title="💻 System Info", border_style="yellow")
    
    def _create_footer_panel(self) -> Panel:
        """Create the footer panel with controls and status."""
        footer_text = Text()
        footer_text.append("Press Ctrl+C to stop debug mode", style="dim")
        footer_text.append(" | ", style="dim")
        footer_text.append("Logs: Check console output for detailed information", style="dim")
        
        return Panel(footer_text, border_style="dim")
    
    def _print_debug_header(self) -> None:
        """Print initial debug information."""
        self.console.print("\n" + "="*80, style="cyan")
        self.console.print("🔍 SPOTIFY SPLITTER DEBUG MODE ACTIVATED", style="bold cyan", justify="center")
        self.console.print("="*80, style="cyan")
        
        # Print configuration
        config_table = Table(show_header=False, box=None)
        config_table.add_column("Setting", style="cyan")
        config_table.add_column("Value", style="white")
        
        config_table.add_row("Metrics Collection Interval", f"{self.metrics_collector.collection_interval}s")
        config_table.add_row("Debug Mode Enabled", "✓")
        config_table.add_row("Live Display", "✓" if self.enable_live_display else "✗")
        config_table.add_row("Buffer Manager", "✓" if self.buffer_manager else "✗")
        config_table.add_row("Health Monitor", "✓" if self.health_monitor else "✗")
        config_table.add_row("Error Recovery", "✓" if self.error_recovery else "✗")
        
        self.console.print(Panel(config_table, title="Debug Configuration", border_style="cyan"))
        self.console.print()
    
    def generate_diagnostic_report(self, level: DiagnosticLevel = DiagnosticLevel.DEBUG) -> DiagnosticReport:
        """Generate a comprehensive diagnostic report."""
        return self.metrics_collector.generate_diagnostic_report(level)
    
    def save_diagnostic_report(self, filepath: Path, level: DiagnosticLevel = DiagnosticLevel.DEBUG) -> None:
        """Save a diagnostic report to file."""
        try:
            report = self.generate_diagnostic_report(level)
            
            with open(filepath, 'w') as f:
                json.dump(report.to_dict(), f, indent=2, default=str)
            
            logger.info("Diagnostic report saved to %s", filepath)
            self.console.print(f"📄 Diagnostic report saved to: {filepath}", style="green")
            
        except Exception as e:
            logger.error("Error saving diagnostic report: %s", e)
            self.console.print(f"❌ Error saving report: {e}", style="red")
    
    def _generate_final_report(self) -> None:
        """Generate and display final debug report."""
        try:
            self.console.print("\n" + "="*80, style="cyan")
            self.console.print("📊 FINAL DEBUG REPORT", style="bold cyan", justify="center")
            self.console.print("="*80, style="cyan")
            
            # Generate comprehensive report
            report = self.generate_diagnostic_report(DiagnosticLevel.DETAILED)
            
            # Display summary
            summary_table = Table(show_header=False, box=None)
            summary_table.add_column("Metric", style="cyan")
            summary_table.add_column("Value", style="white")
            
            summary_table.add_row("Session Duration", str(datetime.now() - self.start_time).split('.')[0])
            summary_table.add_row("Total Metrics Collected", str(report.summary.get('total_metrics', 0)))
            summary_table.add_row("Performance Snapshots", str(len(report.performance_snapshots)))
            summary_table.add_row("Errors Recorded", str(report.error_analysis.get('total_errors', 0)))
            
            self.console.print(Panel(summary_table, title="Session Summary", border_style="green"))
            
            # Display recommendations
            if report.recommendations:
                self.console.print("\n📋 Recommendations:", style="bold yellow")
                for i, rec in enumerate(report.recommendations, 1):
                    self.console.print(f"  {i}. {rec}", style="yellow")
            
            self.console.print("\n" + "="*80, style="cyan")
            
        except Exception as e:
            logger.error("Error generating final report: %s", e)
            self.console.print(f"❌ Error generating final report: {e}", style="red")
    
    def print_current_metrics(self) -> None:
        """Print current metrics to console (non-live mode)."""
        if self.enable_live_display:
            return  # Live display handles this
        
        try:
            self.console.clear()
            layout = self._create_debug_layout()
            self.console.print(layout)
            
        except Exception as e:
            logger.error("Error printing current metrics: %s", e)
    
    def __enter__(self):
        """Context manager entry."""
        self.start_debug_mode()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_debug_mode()