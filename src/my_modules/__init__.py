# my_modules package initialization
# This package contains your custom modules and enhancements

# Import temporal smoothing module
from . import temporal_smoothing

# Expose key classes from temporal_smoothing
from .temporal_smoothing import (
    Base3DTracker,
    SimpleKalman3DTracker,
    AB3DMOTWrapper,
    MovingAverageLaneSmoother,
    ExponentialMovingAverageLaneSmoother,
    SmoothedVideoPipeline
)

__all__ = [
    'temporal_smoothing',
    'Base3DTracker',
    'SimpleKalman3DTracker',
    'AB3DMOTWrapper',
    'MovingAverageLaneSmoother',
    'ExponentialMovingAverageLaneSmoother',
    'SmoothedVideoPipeline'
]
