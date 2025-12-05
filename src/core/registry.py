from typing import Dict, List


class FormatRegistry:
    _readers: Dict[str, Dict] = {}
    _writers: Dict[str, Dict] = {}

    @classmethod
    def register_reader(cls, format_name: str, extensions: List[str]):
        """Decorator to register a reader class."""

        def decorator(reader_class):
            cls._readers[format_name.lower()] = {
                "class": reader_class,
                "extensions": [ext.lower().lstrip(".") for ext in extensions],
            }
            return reader_class

        return decorator

    @classmethod
    def register_writer(cls, format_name: str, extensions: List[str]):
        """Decorator to register a writer class."""

        def decorator(writer_class):
            cls._writers[format_name.lower()] = {
                "class": writer_class,
                "extensions": [ext.lower().lstrip(".") for ext in extensions],
            }
            return writer_class

        return decorator

    @classmethod
    def get_reader(cls, format_name: str = None, extension: str = None):
        """Get reader class by format name or file extension."""
        if format_name:
            entry = cls._readers.get(format_name.lower())
            if entry:
                return entry["class"]

        if extension:
            ext = extension.lower().lstrip(".")
            for entry in cls._readers.values():
                if ext in entry["extensions"]:
                    return entry["class"]

        return None

    @classmethod
    def get_writer(cls, format_name: str = None, extension: str = None):
        """Get writer class by format name or file extension."""
        if format_name:
            entry = cls._writers.get(format_name.lower())
            if entry:
                return entry["class"]

        if extension:
            ext = extension.lower().lstrip(".")
            for entry in cls._writers.values():
                if ext in entry["extensions"]:
                    return entry["class"]

        return None

    @classmethod
    def list_readers(cls) -> List[str]:
        """List all registered reader formats."""
        return list(cls._readers.keys())

    @classmethod
    def list_writers(cls) -> List[str]:
        """List all registered writer formats."""
        return list(cls._writers.keys())

    @classmethod
    def get_reader_extensions(cls, format_name: str) -> List[str]:
        """Get file extensions for a reader format."""
        entry = cls._readers.get(format_name.lower())
        return entry["extensions"] if entry else []

    @classmethod
    def get_writer_extensions(cls, format_name: str) -> List[str]:
        """Get file extensions for a writer format."""
        entry = cls._writers.get(format_name.lower())
        return entry["extensions"] if entry else []
