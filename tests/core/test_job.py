"""Tests for Job class.

Testing:
1. Job creation with messages and files
2. Parameter validation and merging
3. Serialization for different job types
"""

import pytest
import json
from pathlib import Path
from unittest.mock import patch, mock_open

from batchata.core.job import Job
from batchata.exceptions import ValidationError


class TestJob:
    """Test Job creation and validation."""
    
    def test_message_based_job_creation(self):
        """Test creating jobs with messages."""
        # Basic message job
        job = Job(
            id="test-1",
            model="test-model",
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
                {"role": "user", "content": "How are you?"}
            ],
            temperature=0.7
        )
        
        assert job.id == "test-1"
        assert job.model == "test-model"
        assert len(job.messages) == 3
        assert job.messages[0]["role"] == "user"
        assert job.messages[0]["content"] == "Hello"
        assert job.temperature == 0.7
        assert job.file is None
        assert job.prompt is None
        
        # Job with system message
        job2 = Job(
            id="test-2",
            model="test-model",
            messages=[
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Tell me a joke"}
            ]
        )
        
        assert len(job2.messages) == 2
        assert job2.messages[0]["role"] == "system"
    
    @pytest.mark.parametrize("file_path,prompt,should_work", [
        ("test.txt", "Summarize this", True),
        ("test.pdf", "Extract key points", True),
        ("test.txt", None, False),  # Missing prompt when file is provided
        (None, "Summarize", True),  # Prompt-only job is now allowed
    ])
    def test_file_based_job_creation(self, temp_dir, file_path, prompt, should_work):
        """Test creating jobs with file inputs."""
        if file_path:
            full_path = temp_dir / file_path
            full_path.write_text("Test content")
            file_path = Path(full_path)
        
        if should_work:
            job = Job(
                id="file-job",
                model="test-model",
                file=file_path,
                prompt=prompt
            )
            
            assert job.file == file_path
            assert job.messages is None
            assert job.prompt == prompt
        else:
            with pytest.raises(ValueError):
                Job(
                    id="file-job",
                    model="test-model",
                    file=file_path,
                    prompt=prompt
                )
    
    def test_job_serialization_and_validation(self):
        """Test job serialization and parameter handling."""
        # Message job serialization
        msg_job = Job(
            id="msg-job",
            model="test",
            messages=[{"role": "user", "content": "Test"}],
            temperature=0.5,
            max_tokens=100
        )
        
        # Should be JSON serializable
        data = msg_job.to_dict()
        json_str = json.dumps(data)
        loaded = json.loads(json_str)
        
        # Recreate from dict
        restored = Job.from_dict(loaded)
        assert restored.id == msg_job.id
        assert restored.messages == msg_job.messages
        assert restored.model == msg_job.model
        assert restored.temperature == msg_job.temperature
        assert restored.max_tokens == msg_job.max_tokens
        
        # File job with prompt
        file_job = Job(
            id="file-job",
            model="test",
            file=Path("/tmp/test.txt"),
            prompt="Analyze this"
        )
        
        file_data = file_job.to_dict()
        assert file_data["file"] == "/tmp/test.txt"
        assert file_data["prompt"] == "Analyze this"
        
        # Invalid job - both messages and file
        with pytest.raises(ValueError):
            Job(
                id="invalid",
                model="test",
                messages=[{"role": "user", "content": "Test"}],
                file=Path("/tmp/test.txt"),
                prompt="Analyze"
            )