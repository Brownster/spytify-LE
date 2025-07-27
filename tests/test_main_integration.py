"""
Integration tests for main application flow with adaptive buffer management.

Tests the integration of adaptive buffer management, configuration profiles,
system capability detection, and CLI argument handling in the main application.
"""

import pytest
import threading
import time
import queue
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import os

from spotify_splitter.main import app
from spotify_splitter.config_profiles import ProfileManager, ProfileType, SystemCapabilityDetector
from spotify_splitter.buffer_management import AdaptiveBufferManager, BufferStrategy
from spotify_splitter.util import StreamInfo
from typer.testing import CliRunner


class TestMainIntegration:
    """Integration tests for main application with adaptive buffer management."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock stream info
        self.mock_stream_info = StreamInfo(
            monitor_name="test_monitor",
            samplerate=44100,
            channels=2
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('spotify_splitter.main.get_spotify_stream_info')
    @patch('spotify_splitter.main.track_events')
    @patch('spotify_splitter.main.AudioStream')
    @patch('spotify_splitter.main.EnhancedAudioStream')
    @patch('spotify_splitter.main.SegmentManager')
    def test_basic_recording_with_adaptive_management(
        self, mock_segment_manager, mock_enhanced_stream, mock_basic_stream, 
        mock_track_events, mock_get_stream_info
    ):
        """Test basic recording functionality with adaptive management enabled."""
        # Setup mocks
        mock_get_stream_info.return_value = self.mock_stream_info
        
        mock_stream_instance = Mock()
        mock_enhanced_stream.return_value = mock_stream_instance
        mock_stream_instance.__enter__ = Mock(return_value=mock_stream_instance)
        mock_stream_instance.__exit__ = Mock(return_value=None)
        
        mock_manager_instance = Mock()
        mock_segment_manager.return_value = mock_manager_instance
        mock_manager_instance.flush_cache = Mock()
        mock_manager_instance.run = Mock()
        mock_manager_instance.shutdown_cleanup = Mock()
        
        # Mock track_events to simulate quick exit
        def mock_track_events_func(*args, **kwargs):
            time.sleep(0.1)  # Brief simulation
            raise KeyboardInterrupt()
        
        mock_track_events.side_effect = mock_track_events_func
        
        # Test with adaptive management enabled (default)
        result = self.runner.invoke(app, [
            'record',
            '--output', self.temp_dir,
            '--profile', 'desktop',
            '--adaptive'
        ])
        
        # Verify the command completed
        assert result.exit_code == 0
        
        # Verify enhanced audio stream was used
        mock_enhanced_stream.assert_called_once()
        mock_basic_stream.assert_not_called()
        
        # Verify segment manager was initialized
        mock_segment_manager.assert_called_once()
        
        # Verify cleanup was called
        mock_manager_instance.shutdown_cleanup.assert_called_once()
    
    @patch('spotify_splitter.main.get_spotify_stream_info')
    @patch('spotify_splitter.main.track_events')
    @patch('spotify_splitter.main.AudioStream')
    @patch('spotify_splitter.main.EnhancedAudioStream')
    @patch('spotify_splitter.main.SegmentManager')
    def test_recording_without_adaptive_management(
        self, mock_segment_manager, mock_enhanced_stream, mock_basic_stream,
        mock_track_events, mock_get_stream_info
    ):
        """Test recording functionality with adaptive management disabled."""
        # Setup mocks
        mock_get_stream_info.return_value = self.mock_stream_info
        
        mock_stream_instance = Mock()
        mock_basic_stream.return_value = mock_stream_instance
        mock_stream_instance.__enter__ = Mock(return_value=mock_stream_instance)
        mock_stream_instance.__exit__ = Mock(return_value=None)
        
        mock_manager_instance = Mock()
        mock_segment_manager.return_value = mock_manager_instance
        mock_manager_instance.flush_cache = Mock()
        mock_manager_instance.run = Mock()
        mock_manager_instance.shutdown_cleanup = Mock()
        
        # Mock track_events to simulate quick exit
        def mock_track_events_func(*args, **kwargs):
            time.sleep(0.1)
            raise KeyboardInterrupt()
        
        mock_track_events.side_effect = mock_track_events_func
        
        # Test with adaptive management disabled
        result = self.runner.invoke(app, [
            'record',
            '--output', self.temp_dir,
            '--no-adaptive'
        ])
        
        # Verify the command completed
        assert result.exit_code == 0
        
        # Verify basic audio stream was used
        mock_basic_stream.assert_called_once()
        mock_enhanced_stream.assert_not_called()
    
    @patch('spotify_splitter.main.get_spotify_stream_info')
    @patch('spotify_splitter.main.track_events')
    @patch('spotify_splitter.main.EnhancedAudioStream')
    @patch('spotify_splitter.main.SegmentManager')
    def test_spotifyd_mode_integration(
        self, mock_segment_manager, mock_enhanced_stream, mock_track_events, mock_get_stream_info
    ):
        """Test spotifyd mode integration with headless profile."""
        # Setup mocks
        mock_get_stream_info.return_value = self.mock_stream_info
        
        mock_stream_instance = Mock()
        mock_enhanced_stream.return_value = mock_stream_instance
        mock_stream_instance.__enter__ = Mock(return_value=mock_stream_instance)
        mock_stream_instance.__exit__ = Mock(return_value=None)
        
        mock_manager_instance = Mock()
        mock_segment_manager.return_value = mock_manager_instance
        mock_manager_instance.flush_cache = Mock()
        mock_manager_instance.run = Mock()
        mock_manager_instance.shutdown_cleanup = Mock()
        
        def mock_track_events_func(*args, **kwargs):
            # Verify spotifyd player name is used
            assert kwargs.get('player_name') == 'spotifyd'
            time.sleep(0.1)
            raise KeyboardInterrupt()
        
        mock_track_events.side_effect = mock_track_events_func
        
        # Test spotifyd mode
        result = self.runner.invoke(app, [
            'record',
            '--spotifyd-mode',
            '--output', self.temp_dir
        ])
        
        assert result.exit_code == 0
        
        # Verify enhanced stream was called with appropriate settings
        mock_enhanced_stream.assert_called_once()
        call_args = mock_enhanced_stream.call_args
        
        # Should use headless profile settings (larger queue size, higher latency)
        assert call_args.kwargs['queue_size'] >= 300  # Headless profile uses larger buffers
    
    @patch('spotify_splitter.main.SystemCapabilityDetector.detect_capabilities')
    @patch('spotify_splitter.main.get_spotify_stream_info')
    @patch('spotify_splitter.main.track_events')
    @patch('spotify_splitter.main.EnhancedAudioStream')
    @patch('spotify_splitter.main.SegmentManager')
    def test_auto_profile_selection(
        self, mock_segment_manager, mock_enhanced_stream, mock_track_events,
        mock_get_stream_info, mock_detect_capabilities
    ):
        """Test automatic profile selection based on system capabilities."""
        from spotify_splitter.config_profiles import SystemCapabilities
        
        # Setup mocks
        mock_get_stream_info.return_value = self.mock_stream_info
        
        # Mock system capabilities for headless system
        mock_capabilities = SystemCapabilities(
            cpu_cores=2,
            memory_gb=4.0,
            is_headless=True,
            audio_backend="pulseaudio",
            has_gui=False,
            system_load=0.3
        )
        mock_detect_capabilities.return_value = mock_capabilities
        
        mock_stream_instance = Mock()
        mock_enhanced_stream.return_value = mock_stream_instance
        mock_stream_instance.__enter__ = Mock(return_value=mock_stream_instance)
        mock_stream_instance.__exit__ = Mock(return_value=None)
        
        mock_manager_instance = Mock()
        mock_segment_manager.return_value = mock_manager_instance
        mock_manager_instance.flush_cache = Mock()
        mock_manager_instance.run = Mock()
        mock_manager_instance.shutdown_cleanup = Mock()
        
        def mock_track_events_func(*args, **kwargs):
            time.sleep(0.1)
            raise KeyboardInterrupt()
        
        mock_track_events.side_effect = mock_track_events_func
        
        # Test auto profile selection
        result = self.runner.invoke(app, [
            '--output', self.temp_dir,
            'record',
            '--profile', 'auto'
        ])
        
        assert result.exit_code == 0
        
        # Verify capabilities detection was called
        mock_detect_capabilities.assert_called_once()
        
        # Verify enhanced stream was used with headless profile settings
        mock_enhanced_stream.assert_called_once()
        call_args = mock_enhanced_stream.call_args
        
        # Headless profile should use conservative settings
        assert call_args.kwargs['queue_size'] >= 300  # Larger buffer for stability
    
    @patch('spotify_splitter.main.get_spotify_stream_info')
    @patch('spotify_splitter.main.track_events')
    @patch('spotify_splitter.main.EnhancedAudioStream')
    @patch('spotify_splitter.main.SegmentManager')
    @patch('spotify_splitter.main.MetricsCollector')
    def test_metrics_collection_integration(
        self, mock_metrics_collector, mock_segment_manager, mock_enhanced_stream,
        mock_track_events, mock_get_stream_info
    ):
        """Test metrics collection integration."""
        # Setup mocks
        mock_get_stream_info.return_value = self.mock_stream_info
        
        mock_collector_instance = Mock()
        mock_metrics_collector.return_value = mock_collector_instance
        mock_collector_instance.start_collection = Mock()
        mock_collector_instance.stop_collection = Mock()
        mock_collector_instance.generate_diagnostic_report = Mock()
        
        mock_stream_instance = Mock()
        mock_enhanced_stream.return_value = mock_stream_instance
        mock_stream_instance.__enter__ = Mock(return_value=mock_stream_instance)
        mock_stream_instance.__exit__ = Mock(return_value=None)
        
        mock_manager_instance = Mock()
        mock_segment_manager.return_value = mock_manager_instance
        mock_manager_instance.flush_cache = Mock()
        mock_manager_instance.run = Mock()
        mock_manager_instance.shutdown_cleanup = Mock()
        
        def mock_track_events_func(*args, **kwargs):
            time.sleep(0.1)
            raise KeyboardInterrupt()
        
        mock_track_events.side_effect = mock_track_events_func
        
        # Test with metrics enabled and debug mode
        result = self.runner.invoke(app, [
            'record',
            '--metrics',
            '--debug-mode',
            '--output', self.temp_dir
        ])
        
        assert result.exit_code == 0
        
        # Verify metrics collector was initialized and started
        mock_metrics_collector.assert_called_once()
        mock_collector_instance.start_collection.assert_called_once()
        mock_collector_instance.stop_collection.assert_called_once()
        
        # Verify diagnostic report was generated in debug mode
        mock_collector_instance.generate_diagnostic_report.assert_called_once()
    
    @patch('spotify_splitter.main.get_spotify_stream_info')
    def test_cli_argument_overrides(self, mock_get_stream_info):
        """Test that CLI arguments properly override profile defaults."""
        mock_get_stream_info.return_value = self.mock_stream_info
        
        # Test with custom CLI arguments
        with patch('spotify_splitter.main.track_events') as mock_track_events, \
             patch('spotify_splitter.main.EnhancedAudioStream') as mock_enhanced_stream, \
             patch('spotify_splitter.main.SegmentManager') as mock_segment_manager:
            
            mock_stream_instance = Mock()
            mock_enhanced_stream.return_value = mock_stream_instance
            mock_stream_instance.__enter__ = Mock(return_value=mock_stream_instance)
            mock_stream_instance.__exit__ = Mock(return_value=None)
            
            mock_manager_instance = Mock()
            mock_segment_manager.return_value = mock_manager_instance
            mock_manager_instance.flush_cache = Mock()
            mock_manager_instance.run = Mock()
            mock_manager_instance.shutdown_cleanup = Mock()
            
            def mock_track_events_func(*args, **kwargs):
                time.sleep(0.1)
                raise KeyboardInterrupt()
            
            mock_track_events.side_effect = mock_track_events_func
            
            # Test with custom queue size and latency
            result = self.runner.invoke(app, [
                'record',
                '--profile', 'desktop',
                '--queue-size', '500',
                '--latency', '0.05',
                '--blocksize', '1024',
                '--output', self.temp_dir
            ])
            
            assert result.exit_code == 0
            
            # Verify custom settings were used
            call_args = mock_enhanced_stream.call_args
            assert call_args.kwargs['queue_size'] == 500
            assert call_args.kwargs['latency'] == 0.05
            assert call_args.kwargs['blocksize'] == 1024

    @patch('spotify_splitter.main.get_spotify_stream_info')
    @patch('spotify_splitter.main.track_events')
    @patch('spotify_splitter.main.EnhancedAudioStream')
    @patch('spotify_splitter.main.SegmentManager')
    def test_playlist_option(
        self, mock_segment_manager, mock_enhanced_stream, mock_track_events, mock_get_stream_info
    ):
        """Ensure playlist path is passed to SegmentManager."""
        mock_get_stream_info.return_value = self.mock_stream_info

        mock_stream_instance = Mock()
        mock_enhanced_stream.return_value = mock_stream_instance
        mock_stream_instance.__enter__ = Mock(return_value=mock_stream_instance)
        mock_stream_instance.__exit__ = Mock(return_value=None)

        mock_manager_instance = Mock()
        mock_segment_manager.return_value = mock_manager_instance
        mock_manager_instance.flush_cache = Mock()
        mock_manager_instance.run = Mock()
        mock_manager_instance.shutdown_cleanup = Mock()

        def mock_track_events_func(*args, **kwargs):
            time.sleep(0.1)
            raise KeyboardInterrupt()

        mock_track_events.side_effect = mock_track_events_func

        playlist_path = Path(self.temp_dir) / "session.m3u"
        result = self.runner.invoke(app, [
            '--output', self.temp_dir,
            'record',
            '--playlist', str(playlist_path)
        ])

        assert result.exit_code == 0
        kwargs = mock_segment_manager.call_args.kwargs
        assert kwargs['playlist_path'] == playlist_path

    @patch('spotify_splitter.main.tag_output')
    @patch('spotify_splitter.main.get_spotify_stream_info')
    @patch('spotify_splitter.main.track_events')
    @patch('spotify_splitter.main.EnhancedAudioStream')
    @patch('spotify_splitter.main.SegmentManager')
    def test_tagger_called_on_shutdown(
        self, mock_segment_manager, mock_enhanced_stream, mock_track_events, mock_get_stream_info, mock_tag_output
    ):
        """Ensure tagging API is invoked when recording stops."""
        mock_get_stream_info.return_value = self.mock_stream_info

        mock_stream_instance = Mock()
        mock_enhanced_stream.return_value = mock_stream_instance
        mock_stream_instance.__enter__ = Mock(return_value=mock_stream_instance)
        mock_stream_instance.__exit__ = Mock(return_value=None)

        mock_manager_instance = Mock()
        mock_segment_manager.return_value = mock_manager_instance
        mock_manager_instance.flush_cache = Mock()
        mock_manager_instance.run = Mock()
        mock_manager_instance.shutdown_cleanup = Mock()

        def mock_track_events_func(*args, **kwargs):
            time.sleep(0.1)
            raise KeyboardInterrupt()

        mock_track_events.side_effect = mock_track_events_func

        result = self.runner.invoke(app, [
            '--output', self.temp_dir,
            'record'
        ])

        assert result.exit_code == 0
        mock_tag_output.assert_called_once_with(Path(self.temp_dir), None)

    @patch('spotify_splitter.main.tag_output')
    @patch('spotify_splitter.main.get_spotify_stream_info')
    @patch('spotify_splitter.main.track_events')
    @patch('spotify_splitter.main.EnhancedAudioStream')
    @patch('spotify_splitter.main.SegmentManager')
    def test_tagger_called_with_playlist(
        self, mock_segment_manager, mock_enhanced_stream, mock_track_events, mock_get_stream_info, mock_tag_output
    ):
        """Ensure tagging API receives playlist when playlist option is used."""
        mock_get_stream_info.return_value = self.mock_stream_info

        mock_stream_instance = Mock()
        mock_enhanced_stream.return_value = mock_stream_instance
        mock_stream_instance.__enter__ = Mock(return_value=mock_stream_instance)
        mock_stream_instance.__exit__ = Mock(return_value=None)

        mock_manager_instance = Mock()
        mock_segment_manager.return_value = mock_manager_instance
        mock_manager_instance.flush_cache = Mock()
        mock_manager_instance.run = Mock()
        mock_manager_instance.shutdown_cleanup = Mock()

        def mock_track_events_func(*args, **kwargs):
            time.sleep(0.1)
            raise KeyboardInterrupt()

        mock_track_events.side_effect = mock_track_events_func

        playlist_path = Path(self.temp_dir) / 'playlist.m3u'
        result = self.runner.invoke(app, [
            '--output', self.temp_dir,
            'record',
            '--playlist', str(playlist_path)
        ])

        assert result.exit_code == 0
        mock_tag_output.assert_called_once_with(Path(self.temp_dir), playlist_path)
    
    def test_profiles_command(self):
        """Test the profiles command functionality."""
        with patch('spotify_splitter.main.SystemCapabilityDetector.detect_capabilities') as mock_detect:
            from spotify_splitter.config_profiles import SystemCapabilities
            
            mock_capabilities = SystemCapabilities(
                cpu_cores=4,
                memory_gb=8.0,
                is_headless=False,
                audio_backend="pipewire",
                has_gui=True,
                system_load=0.2
            )
            mock_detect.return_value = mock_capabilities
            
            result = self.runner.invoke(app, ['profiles'])
            
            assert result.exit_code == 0
            assert "System Capabilities Detection" in result.output
            assert "Available Configuration Profiles" in result.output
            assert "headless" in result.output
            assert "desktop" in result.output
            assert "high_performance" in result.output
            assert "Usage Examples" in result.output
    
    @patch('spotify_splitter.main.get_spotify_stream_info')
    def test_error_handling_stream_not_found(self, mock_get_stream_info):
        """Test error handling when Spotify stream is not found."""
        mock_get_stream_info.side_effect = RuntimeError("Spotify sink not found")
        
        result = self.runner.invoke(app, ['record'])
        
        assert result.exit_code == 1
        assert "Error finding audio source" in result.output
    
    @patch('spotify_splitter.main.get_spotify_stream_info')
    def test_error_handling_dbus_error(self, mock_get_stream_info):
        """Test error handling for D-Bus errors."""
        from spotify_splitter.main import DBusError
        mock_get_stream_info.side_effect = DBusError("D-Bus connection failed")
        
        result = self.runner.invoke(app, ['record'])
        
        assert result.exit_code == 1
        assert "D-Bus error" in result.output


class TestConfigurationProfiles:
    """Test configuration profile functionality."""
    
    def test_profile_manager_get_profile(self):
        """Test ProfileManager.get_profile functionality."""
        # Test getting specific profiles
        headless_profile = ProfileManager.get_profile(ProfileType.HEADLESS)
        assert headless_profile.name == "headless"
        assert headless_profile.buffer_strategy == BufferStrategy.CONSERVATIVE
        assert headless_profile.queue_size >= 300
        
        desktop_profile = ProfileManager.get_profile(ProfileType.DESKTOP)
        assert desktop_profile.name == "desktop"
        assert desktop_profile.buffer_strategy == BufferStrategy.BALANCED
        
        high_perf_profile = ProfileManager.get_profile(ProfileType.HIGH_PERFORMANCE)
        assert high_perf_profile.name == "high_performance"
        assert high_perf_profile.buffer_strategy == BufferStrategy.LOW_LATENCY
    
    @patch('spotify_splitter.config_profiles.SystemCapabilityDetector.detect_capabilities')
    def test_auto_profile_selection_logic(self, mock_detect):
        """Test automatic profile selection logic."""
        from spotify_splitter.config_profiles import SystemCapabilities
        
        # Test headless system selection
        headless_caps = SystemCapabilities(
            cpu_cores=2, memory_gb=4.0, is_headless=True,
            audio_backend="pulseaudio", has_gui=False, system_load=0.3
        )
        mock_detect.return_value = headless_caps
        
        profile = ProfileManager.select_optimal_profile()
        assert profile.name.startswith("headless")
        
        # Test high-performance system selection
        high_perf_caps = SystemCapabilities(
            cpu_cores=16, memory_gb=32.0, is_headless=False,
            audio_backend="pipewire", has_gui=True, system_load=0.1
        )
        mock_detect.return_value = high_perf_caps
        
        profile = ProfileManager.select_optimal_profile()
        assert profile.name.startswith("high_performance")
        
        # Test desktop system selection
        desktop_caps = SystemCapabilities(
            cpu_cores=4, memory_gb=8.0, is_headless=False,
            audio_backend="pulseaudio", has_gui=True, system_load=0.5
        )
        mock_detect.return_value = desktop_caps
        
        profile = ProfileManager.select_optimal_profile()
        assert profile.name.startswith("desktop")
    
    def test_profile_system_adjustments(self):
        """Test profile adjustments based on system characteristics."""
        from spotify_splitter.config_profiles import SystemCapabilities
        
        # Test low memory adjustment
        low_memory_caps = SystemCapabilities(
            cpu_cores=4, memory_gb=1.5, is_headless=False,
            audio_backend="pulseaudio", has_gui=True, system_load=0.3
        )
        
        base_profile = ProfileManager.get_profile(ProfileType.DESKTOP)
        adjusted_profile = ProfileManager._adjust_profile_for_system(base_profile, low_memory_caps)
        
        # Should reduce buffer size for low memory
        assert adjusted_profile.queue_size < base_profile.queue_size
        assert not adjusted_profile.enable_debug_mode
        
        # Test high memory adjustment
        high_memory_caps = SystemCapabilities(
            cpu_cores=4, memory_gb=16.0, is_headless=False,
            audio_backend="pulseaudio", has_gui=True, system_load=0.3
        )
        
        adjusted_profile = ProfileManager._adjust_profile_for_system(base_profile, high_memory_caps)
        
        # Should increase buffer size for high memory
        assert adjusted_profile.queue_size > base_profile.queue_size


class TestSystemCapabilityDetection:
    """Test system capability detection functionality."""
    
    @patch('spotify_splitter.config_profiles.psutil.cpu_count')
    @patch('spotify_splitter.config_profiles.psutil.virtual_memory')
    @patch('spotify_splitter.config_profiles.psutil.cpu_percent')
    def test_capability_detection(self, mock_cpu_percent, mock_virtual_memory, mock_cpu_count):
        """Test system capability detection."""
        # Mock system information
        mock_cpu_count.return_value = 8
        
        mock_memory = Mock()
        mock_memory.total = 16 * 1024 ** 3  # 16 GB
        mock_virtual_memory.return_value = mock_memory
        
        mock_cpu_percent.return_value = 25.0  # 25% CPU usage
        
        with patch.dict(os.environ, {'DISPLAY': ':0'}):
            capabilities = SystemCapabilityDetector.detect_capabilities()
            
            assert capabilities.cpu_cores == 8
            assert capabilities.memory_gb == 16.0
            assert capabilities.system_load == 0.25
            assert not capabilities.is_headless  # Has DISPLAY
            assert capabilities.has_gui
    
    @patch('spotify_splitter.config_profiles.subprocess.run')
    def test_audio_backend_detection(self, mock_subprocess):
        """Test audio backend detection."""
        # Test PipeWire detection
        mock_subprocess.return_value.returncode = 0
        backend = SystemCapabilityDetector._detect_audio_backend()
        assert backend == "pipewire"
        
        # Test PulseAudio detection (PipeWire not found)
        def mock_run_side_effect(cmd, **kwargs):
            result = Mock()
            if 'pipewire' in cmd:
                result.returncode = 1  # Not found
            elif 'pulseaudio' in cmd:
                result.returncode = 0  # Found
            else:
                result.returncode = 1
            return result
        
        mock_subprocess.side_effect = mock_run_side_effect
        backend = SystemCapabilityDetector._detect_audio_backend()
        assert backend == "pulseaudio"
    
    def test_headless_detection(self):
        """Test headless mode detection."""
        # Test with display environment
        with patch.dict(os.environ, {'DISPLAY': ':0'}, clear=True):
            assert not SystemCapabilityDetector._detect_headless()
        
        # Test without display environment
        with patch.dict(os.environ, {}, clear=True):
            assert SystemCapabilityDetector._detect_headless()
        
        # Test in container
        with patch.dict(os.environ, {}, clear=True), \
             patch('os.path.exists') as mock_exists:
            mock_exists.return_value = True  # /.dockerenv exists
            assert SystemCapabilityDetector._detect_headless()
        
        # Test SSH session
        with patch.dict(os.environ, {'SSH_CLIENT': '192.168.1.1'}, clear=True):
            assert SystemCapabilityDetector._detect_headless()


if __name__ == '__main__':
    pytest.main([__file__])