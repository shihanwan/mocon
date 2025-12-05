"""Base class for all motion file readers."""

from abc import ABC, abstractmethod
from pathlib import Path

from src.core.motion import MotionData


class BaseReader(ABC):
    """Abstract base class for motion file readers."""

    @abstractmethod
    def read(self, filepath: str | Path) -> MotionData:
        """
        Read motion data from file.

        Args:
            filepath: Path to the motion file

        Returns:
            MotionData object with the loaded motion
        """
        pass

    @abstractmethod
    def can_read(self, filepath: str | Path) -> bool:
        """
        Check if this reader can handle the given file.

        Args:
            filepath: Path to check

        Returns:
            True if this reader can handle the file
        """
        pass
