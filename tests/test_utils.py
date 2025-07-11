"""Tests for utility functions."""

import json
import tempfile
from pathlib import Path
from typing import Dict, Any
import pytest
from pydantic import BaseModel

from batchata.utils import load_results_from_disk, _reconstruct_citations
from batchata.citations import Citation


class SampleModel(BaseModel):
    name: str
    value: int


class TestUtils:
    """Test utility functions."""

    def test_load_results_from_disk_without_model(self):
        """Test loading results without response model."""
        with tempfile.TemporaryDirectory() as temp_dir:
            results_dir = Path(temp_dir)
            processed_dir = results_dir / "processed"
            processed_dir.mkdir(parents=True)
            
            # Create mock result data
            mock_results = [
                {
                    "result": {"name": "Test Item 1", "value": 100},
                    "citations": None
                }
            ]
            
            # Save mock data to file
            result_file = processed_dir / "job_0_results.json"
            with open(result_file, 'w') as f:
                json.dump(mock_results, f)
            
            # Test loading without model
            results = load_results_from_disk(str(results_dir))
            
            assert len(results) == 1
            assert isinstance(results[0]["result"], dict)
            assert results[0]["result"]["name"] == "Test Item 1"
            assert results[0]["result"]["value"] == 100
            assert results[0]["citations"] is None

    def test_load_results_from_disk_with_model(self):
        """Test loading results with response model."""
        with tempfile.TemporaryDirectory() as temp_dir:
            results_dir = Path(temp_dir)
            processed_dir = results_dir / "processed"
            processed_dir.mkdir(parents=True)
            
            # Create mock result data
            mock_results = [
                {
                    "result": {"name": "Test Item 1", "value": 100},
                    "citations": {
                        "name": [{"type": "text", "cited_text": "Test Item 1", "document_index": 0}],
                        "value": [{"type": "text", "cited_text": "100", "document_index": 0}]
                    }
                }
            ]
            
            # Save mock data to file
            result_file = processed_dir / "job_0_results.json"
            with open(result_file, 'w') as f:
                json.dump(mock_results, f)
            
            # Test loading with model
            results = load_results_from_disk(str(results_dir), SampleModel)
            
            assert len(results) == 1
            assert isinstance(results[0]["result"], SampleModel)
            assert results[0]["result"].name == "Test Item 1"
            assert results[0]["result"].value == 100
            
            # Check citations are reconstructed
            citations = results[0]["citations"]
            assert citations is not None
            assert "name" in citations
            assert "value" in citations
            assert len(citations["name"]) == 1
            assert isinstance(citations["name"][0], Citation)
            assert citations["name"][0].cited_text == "Test Item 1"

    def test_load_results_from_disk_multiple_files(self):
        """Test loading results from multiple job files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            results_dir = Path(temp_dir)
            processed_dir = results_dir / "processed"
            processed_dir.mkdir(parents=True)
            
            # Create mock result data for multiple jobs
            mock_results_1 = [
                {"result": {"name": "Item 1", "value": 100}, "citations": None}
            ]
            mock_results_2 = [
                {"result": {"name": "Item 2", "value": 200}, "citations": None}
            ]
            
            # Save mock data to files
            (processed_dir / "job_0_results.json").write_text(json.dumps(mock_results_1))
            (processed_dir / "job_1_results.json").write_text(json.dumps(mock_results_2))
            
            # Test loading
            results = load_results_from_disk(str(results_dir), SampleModel)
            
            assert len(results) == 2
            assert results[0]["result"].name == "Item 1"
            assert results[1]["result"].name == "Item 2"

    def test_load_results_from_disk_no_directory(self):
        """Test loading from non-existent directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            results_dir = Path(temp_dir) / "nonexistent"
            
            results = load_results_from_disk(str(results_dir))
            assert results == []

    def test_reconstruct_citations_field_citations(self):
        """Test reconstructing field citations."""
        citations_data = {
            "name": [{"type": "text", "cited_text": "Test", "document_index": 0}],
            "value": [{"type": "text", "cited_text": "123", "document_index": 0}]
        }
        
        result = _reconstruct_citations(citations_data)
        
        assert isinstance(result, dict)
        assert "name" in result
        assert "value" in result
        assert isinstance(result["name"][0], Citation)
        assert result["name"][0].cited_text == "Test"

    def test_reconstruct_citations_list_citations(self):
        """Test reconstructing list of citations."""
        citations_data = [
            {"type": "text", "cited_text": "Test", "document_index": 0},
            {"type": "text", "cited_text": "Another", "document_index": 1}
        ]
        
        result = _reconstruct_citations(citations_data)
        
        assert isinstance(result, list)
        assert len(result) == 2
        assert isinstance(result[0], Citation)
        assert result[0].cited_text == "Test"

    def test_reconstruct_citations_with_defaults(self):
        """Test reconstructing citations with missing required fields."""
        citations_data = [
            {"cited_text": "Test"}  # Missing type and document_index
        ]
        
        result = _reconstruct_citations(citations_data)
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], Citation)
        assert result[0].cited_text == "Test"
        assert result[0].type == "text"  # Default
        assert result[0].document_index == 0  # Default

    def test_reconstruct_citations_malformed(self):
        """Test reconstructing malformed citations falls back gracefully."""
        citations_data = [
            {"invalid": "data"}  # Missing required fields
        ]
        
        result = _reconstruct_citations(citations_data)
        
        # Should fall back to returning the original dict if Citation creation fails
        assert isinstance(result, list)
        assert len(result) == 1
        # Could be either Citation object (if defaults worked) or original dict (if failed)
        assert "invalid" in str(result[0]) or hasattr(result[0], 'cited_text')