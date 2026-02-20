"""
ARIA — Agentic Runtime for Intelligent Autonomy.

Safety infrastructure for AI agents: sandboxed capabilities, immutable audit
ledger, rollback, trust-scoped sub-agent delegation, and human checkpoint gates.
"""

from aria.models import (
    AgentManifest,
    AuditEvent,
    CapabilityPolicy,
    Checkpoint,
    HumanGateRule,
    TrustToken,
)
from aria.orchestrator import ARIASession

__version__ = "0.1.0"
__all__ = [
    "ARIASession",
    "AgentManifest",
    "AuditEvent",
    "CapabilityPolicy",
    "Checkpoint",
    "HumanGateRule",
    "TrustToken",
]
