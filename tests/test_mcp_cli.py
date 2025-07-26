"""Tests for MCP CLI functionality."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from batchata.cli.mcp import MCPCommands


class TestMCPCommands(unittest.TestCase):
    """Test cases for MCP commands."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.mcp = MCPCommands(state_dir=self.temp_dir)
    
    def test_state_dir_creation(self):
        """Test that state directory is created."""
        self.assertTrue(self.mcp.state_dir.exists())
        self.assertTrue(self.mcp.batches_dir.exists())
    
    def test_list_empty(self):
        """Test listing when no batches exist."""
        batches = self.mcp.list()
        self.assertEqual(batches, [])
    
    @patch('batchata.cli.mcp.Batch')
    def test_create_with_messages(self, mock_batch_class):
        """Test creating a batch with messages."""
        # Mock the batch and run
        mock_batch = Mock()
        mock_run = Mock()
        mock_run.start_time = None
        mock_batch.run.return_value = mock_run
        mock_batch_class.return_value = mock_batch
        
        # Create batch
        batch_id = self.mcp.create(
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.7
        )
        
        # Verify batch was created
        self.assertIsInstance(batch_id, str)
        self.assertTrue(len(batch_id) > 0)
        
        # Check metadata file was created
        metadata_file = self.mcp.batches_dir / f"{batch_id}_metadata.json"
        self.assertTrue(metadata_file.exists())
        
        # Check metadata content
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        self.assertEqual(metadata["model"], "claude-sonnet-4-20250514")
        self.assertEqual(metadata["kwargs"]["temperature"], 0.7)
    
    @patch('batchata.cli.mcp.Batch')
    def test_create_with_file_and_prompt(self, mock_batch_class):
        """Test creating a batch with file and prompt."""
        # Mock the batch and run
        mock_batch = Mock()
        mock_run = Mock()
        mock_run.start_time = None
        mock_batch.run.return_value = mock_run
        mock_batch_class.return_value = mock_batch
        
        # Create batch
        batch_id = self.mcp.create(
            model="gpt-4o-2024-08-06",
            file_path="test.pdf",
            prompt="Summarize this document"
        )
        
        # Verify batch was created
        self.assertIsInstance(batch_id, str)
        mock_batch.add_job.assert_called_once_with(file="test.pdf", prompt="Summarize this document")
    
    def test_create_validation_error(self):
        """Test that create raises error with invalid arguments."""
        with self.assertRaises(ValueError):
            self.mcp.create(model="test-model")  # No messages or file+prompt
    
    def test_results_batch_not_found(self):
        """Test results command with non-existent batch."""
        with self.assertRaises(ValueError):
            self.mcp.results("non-existent-batch-id")
    
    def test_cancel_batch_not_found(self):
        """Test cancel command with non-existent batch."""
        with self.assertRaises(ValueError):
            self.mcp.cancel("non-existent-batch-id")


if __name__ == '__main__':
    unittest.main()