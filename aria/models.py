"""Core Pydantic v2 schemas shared across all ARIA layers."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TrustLevel(int, Enum):
    UNTRUSTED = 0
    LOW = 3
    MEDIUM = 5
    HIGH = 8
    SYSTEM = 10


class EventType(str, Enum):
    AGENT_START = "agent_start"
    AGENT_STOP = "agent_stop"
    THINK = "think"
    ACT = "act"
    OBSERVE = "observe"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    SANDBOX_VIOLATION = "sandbox_violation"
    CHECKPOINT_PENDING = "checkpoint_pending"
    CHECKPOINT_APPROVED = "checkpoint_approved"
    CHECKPOINT_DENIED = "checkpoint_denied"
    TRUST_TOKEN_ISSUED = "trust_token_issued"
    TRUST_TOKEN_VIOLATED = "trust_token_violated"
    ROLLBACK = "rollback"
    ERROR = "error"


class PolicyAction(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"


class GateCondition(str, Enum):
    CONFIDENCE_BELOW = "confidence_below"
    COST_ABOVE = "cost_above"
    EXPLICIT = "explicit"


# ---------------------------------------------------------------------------
# Agent Manifest
# ---------------------------------------------------------------------------


class AgentManifest(BaseModel):
    """Declares an agent's identity and the capabilities it is permitted to use."""

    agent_id: str = Field(default_factory=lambda: f"agent_{uuid.uuid4().hex[:8]}")
    name: str
    version: str = "0.1.0"
    description: str = ""
    allowed_capabilities: list[str] = Field(default_factory=list)
    trust_level: TrustLevel = TrustLevel.MEDIUM
    tags: list[str] = Field(default_factory=list)

    @field_validator("allowed_capabilities")
    @classmethod
    def capabilities_must_be_non_empty_strings(cls, v: list[str]) -> list[str]:
        for cap in v:
            if not cap.strip():
                raise ValueError("Capability names must be non-empty strings")
        return [c.strip().lower() for c in v]


# ---------------------------------------------------------------------------
# Capability Policy
# ---------------------------------------------------------------------------


class PolicyRule(BaseModel):
    """A single allow/deny/require_approval rule for a specific capability."""

    capability: str
    action: PolicyAction
    agent_id: str | None = None  # None = applies to all agents


class CapabilityPolicy(BaseModel):
    """Policy set evaluated by the SandboxExecutor before every tool call."""

    rules: list[PolicyRule] = Field(default_factory=list)
    default_action: PolicyAction = PolicyAction.DENY

    def evaluate(self, agent_id: str, capability: str) -> PolicyAction:
        """Return the action for a given agent + capability pair."""
        for rule in self.rules:
            if rule.capability == capability:
                if rule.agent_id is None or rule.agent_id == agent_id:
                    return rule.action
        return self.default_action


# ---------------------------------------------------------------------------
# Audit Event
# ---------------------------------------------------------------------------


class AuditEvent(BaseModel):
    """Immutable record of a single agent action or system event."""

    event_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    session_id: str
    agent_id: str
    event_type: EventType
    input_data: dict[str, Any] = Field(default_factory=dict)
    output_data: dict[str, Any] = Field(default_factory=dict)
    confidence: float | None = None
    estimated_cost: float | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# Trust Token
# ---------------------------------------------------------------------------


class TrustToken(BaseModel):
    """A cryptographically signed, capability-scoped delegation token."""

    token_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    issuer_agent_id: str
    grantee_agent_id: str
    scoped_capabilities: list[str]
    issued_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime
    signature: str = ""  # set by TrustTokenManager.issue()

    def is_expired(self) -> bool:
        return datetime.now(timezone.utc) > self.expires_at

    def permits(self, capability: str) -> bool:
        return capability.lower() in [c.lower() for c in self.scoped_capabilities]


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------


class Checkpoint(BaseModel):
    """A named point-in-time snapshot within a session, used for rollback."""

    checkpoint_id: str = Field(default_factory=lambda: f"ckpt_{uuid.uuid4().hex[:8]}")
    session_id: str
    label: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    snapshot_refs: list[str] = Field(default_factory=list)  # keys into RollbackEngine
    reversible: bool = True


# ---------------------------------------------------------------------------
# Human Gate Rule
# ---------------------------------------------------------------------------


class HumanGateRule(BaseModel):
    """Describes when execution should pause and wait for human approval."""

    rule_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    condition: GateCondition
    threshold: float | None = None  # used for CONFIDENCE_BELOW / COST_ABOVE
    applies_to: list[str] = Field(default_factory=list)  # capability names; empty = all
    description: str = ""
