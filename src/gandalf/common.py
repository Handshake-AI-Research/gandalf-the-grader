"""Shared helpers used by both the judge and orchestrator layers."""

from gandalf.config import Verdict


def fail_verdicts(n: int, reason: str) -> list[Verdict]:
    """Return *n* fail verdicts that all share the same reason."""
    return [Verdict(met=None, reasoning=reason) for _ in range(n)]
