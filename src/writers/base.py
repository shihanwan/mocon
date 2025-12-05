from abc import ABC, abstractmethod
from pathlib import Path

from src.core.motion import MotionData


class BaseWriter(ABC):
    @abstractmethod
    def write(self, motion: MotionData, filepath: str | Path, **kwargs) -> Path:
        """
        Write motion data to file.

        Args:
            motion: MotionData to write
            filepath: Output file path
            **kwargs: Format-specific options

        Returns:
            Path to the written file
        """
        pass

    @abstractmethod
    def get_extension(self) -> str:
        """Return the file extension this writer produces."""
        pass
