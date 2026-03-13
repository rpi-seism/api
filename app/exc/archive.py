class ArchiveNotFound(Exception):
    """Raised when an SDS archive is not found at the configured path."""


class InventoryNotFound(Exception):
    """Raised when station.xml inventory file is not found at the configured path."""


class InvalidTimeFormat(Exception):
    """Raised when a time string cannot be parsed as ISO-8601."""
