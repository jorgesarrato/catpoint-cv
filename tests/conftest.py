"""pytest fixtures and helpers for the cat-tracker test suite."""

import numpy as np
import pytest


def make_frame(height: int = 480, width: int = 640, color: tuple = (100, 150, 200)) -> np.ndarray:
    """Create a solid-color BGR frame."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = color
    return frame


def make_random_frame(height: int = 480, width: int = 640, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (height, width, 3), dtype=np.uint8)


@pytest.fixture
def solid_frame():
    return make_frame()


@pytest.fixture
def random_frame():
    return make_random_frame()
