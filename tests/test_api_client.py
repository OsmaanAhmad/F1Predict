"""
Unit tests for API client
"""

import pytest
from src.data.api_client import OpenF1Client


def test_client_initialization():
    """Test API client initialization"""
    client = OpenF1Client()
    assert client.base_url == "https://api.openf1.org/v1"
    assert client.timeout == 30
    client.close()


def test_custom_config():
    """Test client with custom configuration"""
    client = OpenF1Client(
        base_url="https://custom.api.com/v1",
        timeout=60,
        max_retries=5
    )
    assert client.base_url == "https://custom.api.com/v1"
    assert client.timeout == 60
    assert client.max_retries == 5
    client.close()


# Add more tests as needed
