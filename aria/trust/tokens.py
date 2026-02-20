"""Trust delegation — HMAC-signed, capability-scoped tokens for sub-agents."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import secrets
from datetime import datetime, timedelta, timezone

from aria.models import AgentManifest, AuditEvent, EventType, TrustToken

logger = logging.getLogger(__name__)

# Session secret used to sign tokens. In production, inject via env or KMS.
_SESSION_SECRET: str = os.environ.get("ARIA_SESSION_SECRET", secrets.token_hex(32))


class TrustTokenManager:
    """
    Issues and verifies cryptographically signed trust tokens.

    A parent agent can only grant capabilities that are a **strict subset**
    of its own — privilege escalation is impossible at the issuance layer.
    """

    def __init__(self, ledger: object, secret: str | None = None) -> None:
        self.ledger = ledger
        self._secret = (secret or _SESSION_SECRET).encode()

    def issue(
        self,
        parent_manifest: AgentManifest,
        grantee_agent_id: str,
        requested_capabilities: list[str],
        ttl_seconds: int = 3600,
    ) -> TrustToken:
        """
        Issue a trust token from parent to grantee.

        Only capabilities present in ``parent_manifest.allowed_capabilities``
        are granted — extras are silently dropped.
        """
        parent_caps = set(parent_manifest.allowed_capabilities)
        granted = sorted(set(c.lower() for c in requested_capabilities) & parent_caps)

        if not granted:
            raise ValueError(
                f"No overlapping capabilities between requested {requested_capabilities} "
                f"and parent manifest {list(parent_caps)}"
            )

        token = TrustToken(
            issuer_agent_id=parent_manifest.agent_id,
            grantee_agent_id=grantee_agent_id,
            scoped_capabilities=granted,
            expires_at=datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds),
        )
        token = token.model_copy(update={"signature": self._sign(token)})
        logger.info(
            "TrustToken issued: %s -> %s caps=%s",
            parent_manifest.agent_id,
            grantee_agent_id,
            granted,
        )
        return token

    def verify(self, token: TrustToken, capability: str, session_id: str = "") -> bool:
        """
        Verify a token is authentic, unexpired, and permits the capability.
        Returns True if all checks pass.
        """
        if token.is_expired():
            logger.warning("TrustToken %s is expired", token.token_id)
            return False

        expected_sig = self._sign(token)
        if not hmac.compare_digest(token.signature, expected_sig):
            logger.error("TrustToken %s has invalid signature", token.token_id)
            return False

        if not token.permits(capability):
            logger.warning(
                "TrustToken %s does not permit capability %s", token.token_id, capability
            )
            return False

        return True

    def _sign(self, token: TrustToken) -> str:
        payload = json.dumps(
            {
                "token_id": token.token_id,
                "issuer": token.issuer_agent_id,
                "grantee": token.grantee_agent_id,
                "caps": sorted(token.scoped_capabilities),
                "expires_at": token.expires_at.isoformat(),
            },
            sort_keys=True,
        ).encode()
        return hmac.new(self._secret, payload, hashlib.sha256).hexdigest()
