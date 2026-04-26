"""
Lightweight scheduling state module.

This package intentionally exposes a very small API surface:
- request lifecycle hooks used by the request entrypoint / middleware
- one unified function that returns the scheduling snapshot
"""

from gpustack.runtime_state.service import get_current_state
from gpustack.runtime_state.tracker import (
    extract_request_meta,
    request_bound,
    request_finished,
    request_started,
)

__all__ = [
    "extract_request_meta",
    "get_current_state",
    "request_bound",
    "request_finished",
    "request_started",
]
