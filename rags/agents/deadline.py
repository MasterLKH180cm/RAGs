"""A cooperative runtime deadline shared with trusted agent tool providers.

The deadline is scoped to one execution context, never to an HTTP request's
tenant or a global process. Providers must use it to bound external I/O;
untrusted/side-effecting callbacks cannot be forcibly stopped safely.
"""
from contextvars import ContextVar
import math
from time import monotonic

from rags.registry import CapabilityError


execution_deadline: ContextVar[float | None] = ContextVar("rags_agent_deadline", default=None)


def remaining_timeout(max_seconds: float) -> float:
    """Return a provider timeout no larger than the remaining tool budget."""
    if not isinstance(max_seconds, (int, float)) or not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("Provider deadline must be positive and finite")
    deadline = execution_deadline.get()
    if deadline is None:
        return float(max_seconds)
    remaining = deadline - monotonic()
    if remaining <= 0.05:
        raise CapabilityError("tool_timeout", "Tool's execution deadline was exhausted", 504)
    return max(0.05, min(float(max_seconds), remaining))
