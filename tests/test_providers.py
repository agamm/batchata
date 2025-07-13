"""Tests for Provider base class."""

import pytest
from abc import ABC

from batchata.providers import Provider
from batchata.core import Job


class TestProvider:
    """Tests for Provider abstract base class."""
    
    def test_provider_is_abstract(self):
        """Test that Provider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Provider()