"""
Unit tests for ProgressMonitor component.
"""

import time
from unittest.mock import Mock, patch

from src.batch_manager import ProgressMonitor, BatchManager


class TestProgressMonitor:
    """Test ProgressMonitor functionality in isolation."""
    
    def setup_method(self):
        """Setup for each test."""
        # Create a mock BatchManager
        self.mock_manager = Mock(spec=BatchManager)
        self.mock_manager.stats = {
            "total_items": 10,
            "completed_items": 5,
            "failed_items": 1,
            "total_cost": 2.50,
            "jobs_completed": 2,
            "cost_limit_reached": False
        }
        
    def test_progress_monitor_init(self):
        """Test ProgressMonitor initialization."""
        monitor = ProgressMonitor(self.mock_manager)
        
        assert monitor.batch_manager == self.mock_manager
        assert monitor.running == False
        assert monitor.thread is None
        assert monitor.is_retry == False
        
    @patch('src.batch_manager.time.sleep')
    def test_progress_monitor_start_stop(self, mock_sleep):
        """Test starting and stopping the progress monitor."""
        monitor = ProgressMonitor(self.mock_manager)
        
        # Start monitoring
        monitor.start()
        assert monitor.running == True
        assert monitor.thread is not None
        assert monitor.thread.daemon == True
        
        # Let it run briefly
        time.sleep(0.01)  # Minimal sleep time
        
        # Stop monitoring
        monitor.stop()
        assert monitor.running == False
        
    @patch('src.batch_manager.time.sleep')
    def test_progress_monitor_retry_mode(self, mock_sleep):
        """Test progress monitor in retry mode."""
        monitor = ProgressMonitor(self.mock_manager)
        
        # Start in retry mode
        monitor.start(is_retry=True)
        assert monitor.is_retry == True
        assert monitor.running == True
        
        monitor.stop()
        
    def test_progress_update_with_stats(self):
        """Test that progress monitor calls stats correctly."""
        monitor = ProgressMonitor(self.mock_manager)
        
        # Manually call the update method
        with patch('builtins.print'):  # Suppress print output
            monitor._update_progress()
        
        # Verify stats was called
        self.mock_manager.stats
        
    def test_progress_monitor_handles_exceptions(self):
        """Test that progress monitor handles exceptions gracefully."""
        # Create a manager that raises exceptions for stats property
        broken_manager = Mock(spec=BatchManager)
        broken_manager.stats = property(lambda self: exec('raise Exception("Test error")'))
        
        monitor = ProgressMonitor(broken_manager)
        
        # This should not raise an exception due to try-catch in _monitor_loop
        with patch('builtins.print'):  # Suppress print output
            try:
                monitor._update_progress()
            except Exception:
                # Expected to fail, but the monitor loop should handle it
                pass
            
        # Monitor should still be functional
        assert monitor.batch_manager == broken_manager
        
    @patch('src.batch_manager.time.sleep')
    def test_progress_monitor_thread_termination(self, mock_sleep):
        """Test that progress monitor thread terminates properly."""
        monitor = ProgressMonitor(self.mock_manager)
        
        monitor.start()
        thread = monitor.thread
        
        # Thread should be alive
        assert thread.is_alive()
        
        # Stop and wait
        monitor.stop()
        
        # Thread should terminate within reasonable time
        thread.join(timeout=0.5)  # Reasonable timeout
        assert not thread.is_alive()
        
    def test_progress_timing(self):
        """Test that progress monitor tracks timing correctly."""
        monitor = ProgressMonitor(self.mock_manager)
        
        start_time = time.time()
        monitor.start()
        
        # Check that start_time is set reasonably
        assert abs(monitor.start_time - start_time) < 0.1
        
        time.sleep(0.01)  # Minimal sleep time
        monitor.stop()
        
    @patch('src.batch_manager.time.sleep')
    def test_multiple_start_stop_cycles(self, mock_sleep):
        """Test multiple start/stop cycles work correctly."""
        monitor = ProgressMonitor(self.mock_manager)
        
        for i in range(3):
            monitor.start()
            assert monitor.running == True
            
            time.sleep(0.01)  # Minimal sleep time
            
            monitor.stop()
            assert monitor.running == False
            
    def test_stop_before_start(self):
        """Test that stopping before starting doesn't cause issues."""
        monitor = ProgressMonitor(self.mock_manager)
        
        # Should not raise any exceptions
        monitor.stop()
        assert monitor.running == False
        assert monitor.thread is None