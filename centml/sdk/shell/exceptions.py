"""SDK exceptions for shell operations (no Click dependency)."""


class ShellError(Exception):
    """Base exception for shell operations."""


class NoPodAvailableError(ShellError):
    """No running pods found for the deployment."""


class PodNotFoundError(ShellError):
    """Specified pod not found among running pods."""
