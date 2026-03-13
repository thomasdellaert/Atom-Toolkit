import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (require NIST network calls; skip with -m 'not slow')",
    )
