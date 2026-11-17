"""Actionable errors; pipeline failures must never silently change the protocol."""


class PreprocessingError(RuntimeError):
    """Base error for a failed preprocessing operation."""


class ConfigurationError(PreprocessingError):
    """Invalid settings or an unsafe output location."""


class DataValidationError(PreprocessingError):
    """Source data do not satisfy the documented experiment contract."""


class ResumeMismatchError(PreprocessingError):
    """Existing output belongs to different inputs, code, or configuration."""
