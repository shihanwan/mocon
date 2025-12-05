"""
mocon - Motion Capture Converter

An extensible tool for converting motion capture files between formats.
Currently supports: FBX → AMASS (ASAP-compatible)
"""

__version__ = "0.1.0"

from src.core import MotionData, FormatRegistry
from src.readers import FBXReader
from src.writers import AMASSWriter

__all__ = [
    "MotionData",
    "FormatRegistry",
    "FBXReader",
    "AMASSWriter",
]
