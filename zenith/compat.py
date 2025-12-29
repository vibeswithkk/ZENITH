# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Backward Compatibility Utilities for Zenith

Provides deprecation warnings and version compatibility tools.

Usage:
    from zenith.compat import deprecated, warn_deprecated, VersionInfo

    @deprecated(since="0.3.0", removal="0.5.0", alternative="new_function()")
    def old_function():
        pass
"""

import functools
import warnings
from dataclasses import dataclass
from typing import Optional, Callable, Any


@dataclass
class VersionInfo:
    """
    Semantic version representation.

    Supports comparison operators for version checks.

    Example:
        v = VersionInfo.parse("0.3.0")
        if v >= VersionInfo(0, 3, 0):
            print("Feature available")
    """

    major: int
    minor: int
    patch: int

    @classmethod
    def parse(cls, version_str: str) -> "VersionInfo":
        """
        Parse version string to VersionInfo.

        Args:
            version_str: Version string like "0.3.0" or "1.2.3"

        Returns:
            VersionInfo instance
        """
        parts = version_str.split(".")
        if len(parts) < 2:
            raise ValueError(f"Invalid version string: {version_str}")

        major = int(parts[0])
        minor = int(parts[1])
        patch = int(parts[2]) if len(parts) > 2 else 0

        return cls(major=major, minor=minor, patch=patch)

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    def __lt__(self, other: "VersionInfo") -> bool:
        return (self.major, self.minor, self.patch) < (
            other.major,
            other.minor,
            other.patch,
        )

    def __le__(self, other: "VersionInfo") -> bool:
        return (self.major, self.minor, self.patch) <= (
            other.major,
            other.minor,
            other.patch,
        )

    def __gt__(self, other: "VersionInfo") -> bool:
        return (self.major, self.minor, self.patch) > (
            other.major,
            other.minor,
            other.patch,
        )

    def __ge__(self, other: "VersionInfo") -> bool:
        return (self.major, self.minor, self.patch) >= (
            other.major,
            other.minor,
            other.patch,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, VersionInfo):
            return False
        return (self.major, self.minor, self.patch) == (
            other.major,
            other.minor,
            other.patch,
        )


def get_current_version() -> VersionInfo:
    """Get current Zenith version."""
    try:
        from zenith import __version__

        return VersionInfo.parse(__version__)
    except (ImportError, ValueError):
        return VersionInfo(0, 0, 0)


class ZenithDeprecationWarning(DeprecationWarning):
    """Custom deprecation warning for Zenith."""

    pass


# Ensure our warnings are shown by default
warnings.filterwarnings("default", category=ZenithDeprecationWarning)


def warn_deprecated(
    name: str,
    since: str,
    removal: Optional[str] = None,
    alternative: Optional[str] = None,
    stacklevel: int = 2,
) -> None:
    """
    Emit a deprecation warning.

    Args:
        name: Name of deprecated feature
        since: Version when deprecation was introduced
        removal: Version when feature will be removed
        alternative: Suggested alternative
        stacklevel: Stack level for warning (default: 2)
    """
    msg_parts = [f"'{name}' is deprecated since version {since}."]

    if removal:
        msg_parts.append(f"It will be removed in version {removal}.")

    if alternative:
        msg_parts.append(f"Use {alternative} instead.")

    message = " ".join(msg_parts)
    warnings.warn(message, ZenithDeprecationWarning, stacklevel=stacklevel)


def deprecated(
    since: str,
    removal: Optional[str] = None,
    alternative: Optional[str] = None,
) -> Callable:
    """
    Decorator to mark functions/methods as deprecated.

    Args:
        since: Version when deprecation was introduced
        removal: Version when feature will be removed (optional)
        alternative: Suggested alternative (optional)

    Returns:
        Decorator function

    Example:
        @deprecated(since="0.3.0", removal="0.5.0", alternative="new_api()")
        def old_api():
            pass
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            warn_deprecated(
                name=func.__qualname__,
                since=since,
                removal=removal,
                alternative=alternative,
                stacklevel=3,
            )
            return func(*args, **kwargs)

        # Update docstring
        deprecation_note = f"\n\n.. deprecated:: {since}\n"
        if removal:
            deprecation_note += f"   Will be removed in {removal}.\n"
        if alternative:
            deprecation_note += f"   Use {alternative} instead.\n"

        if wrapper.__doc__:
            wrapper.__doc__ += deprecation_note
        else:
            wrapper.__doc__ = deprecation_note.strip()

        return wrapper

    return decorator


def check_version_compatibility(
    min_version: Optional[str] = None,
    max_version: Optional[str] = None,
) -> bool:
    """
    Check if current version is within specified range.

    Args:
        min_version: Minimum required version (inclusive)
        max_version: Maximum supported version (inclusive)

    Returns:
        True if current version is compatible
    """
    current = get_current_version()

    if min_version:
        min_v = VersionInfo.parse(min_version)
        if current < min_v:
            return False

    if max_version:
        max_v = VersionInfo.parse(max_version)
        if current > max_v:
            return False

    return True


# Registry of deprecated features for documentation
_DEPRECATION_REGISTRY: list[dict] = []


def register_deprecation(
    name: str,
    since: str,
    removal: Optional[str] = None,
    reason: Optional[str] = None,
    alternative: Optional[str] = None,
) -> None:
    """
    Register a deprecation for tracking and documentation.

    Args:
        name: Name of deprecated feature
        since: Version when deprecated
        removal: Planned removal version
        reason: Why it was deprecated
        alternative: Suggested alternative
    """
    _DEPRECATION_REGISTRY.append(
        {
            "name": name,
            "since": since,
            "removal": removal,
            "reason": reason,
            "alternative": alternative,
        }
    )


def get_deprecation_list() -> list[dict]:
    """Get list of all registered deprecations."""
    return _DEPRECATION_REGISTRY.copy()


def print_deprecation_report() -> None:
    """Print a formatted deprecation report."""
    if not _DEPRECATION_REGISTRY:
        print("No deprecations registered.")
        return

    print("=" * 60)
    print("ZENITH DEPRECATION REPORT")
    print("=" * 60)

    for dep in _DEPRECATION_REGISTRY:
        print(f"\nFeature: {dep['name']}")
        print(f"  Deprecated since: {dep['since']}")
        if dep.get("removal"):
            print(f"  Removal planned: {dep['removal']}")
        if dep.get("reason"):
            print(f"  Reason: {dep['reason']}")
        if dep.get("alternative"):
            print(f"  Alternative: {dep['alternative']}")

    print("\n" + "=" * 60)
