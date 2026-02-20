"""Tests for trust delegation — token issuance, verification, scope enforcement."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from aria.ledger.store import LedgerStore
from aria.models import AgentManifest, TrustLevel, TrustToken
from aria.trust.tokens import TrustTokenManager


@pytest.fixture
async def ledger(tmp_path):
    store = LedgerStore(db_path=tmp_path / "trust.db")
    await store.connect()
    yield store
    await store.close()


@pytest.fixture
def manager(ledger):
    return TrustTokenManager(ledger=ledger, secret="test_secret_key_for_tests")


@pytest.fixture
def parent_manifest():
    return AgentManifest(
        agent_id="parent_001",
        name="ParentAgent",
        allowed_capabilities=["search", "summarize", "write_file"],
    )


class TestTrustTokenIssuance:
    def test_issue_grants_only_subset_of_parent_caps(self, manager, parent_manifest):
        token = manager.issue(
            parent_manifest=parent_manifest,
            grantee_agent_id="child_001",
            requested_capabilities=["search", "summarize"],
        )
        assert set(token.scoped_capabilities) == {"search", "summarize"}

    def test_issue_drops_capabilities_not_in_parent(self, manager, parent_manifest):
        token = manager.issue(
            parent_manifest=parent_manifest,
            grantee_agent_id="child_001",
            requested_capabilities=["search", "launch_rockets"],  # launch_rockets not in parent
        )
        assert "launch_rockets" not in token.scoped_capabilities
        assert "search" in token.scoped_capabilities

    def test_issue_with_no_overlap_raises(self, manager, parent_manifest):
        with pytest.raises(ValueError, match="No overlapping capabilities"):
            manager.issue(
                parent_manifest=parent_manifest,
                grantee_agent_id="child_001",
                requested_capabilities=["hack_db", "steal_data"],
            )

    def test_token_is_signed(self, manager, parent_manifest):
        token = manager.issue(
            parent_manifest=parent_manifest,
            grantee_agent_id="child_001",
            requested_capabilities=["search"],
        )
        assert len(token.signature) == 64  # SHA-256 hex


class TestTrustTokenVerification:
    def test_valid_token_verifies(self, manager, parent_manifest):
        token = manager.issue(
            parent_manifest=parent_manifest,
            grantee_agent_id="child_001",
            requested_capabilities=["search"],
        )
        assert manager.verify(token, "search") is True

    def test_unpermitted_capability_fails_verification(self, manager, parent_manifest):
        token = manager.issue(
            parent_manifest=parent_manifest,
            grantee_agent_id="child_001",
            requested_capabilities=["search"],
        )
        assert manager.verify(token, "write_file") is False

    def test_tampered_signature_fails_verification(self, manager, parent_manifest):
        token = manager.issue(
            parent_manifest=parent_manifest,
            grantee_agent_id="child_001",
            requested_capabilities=["search"],
        )
        tampered = token.model_copy(update={"signature": "deadbeef" * 8})
        assert manager.verify(tampered, "search") is False

    def test_expired_token_fails_verification(self, manager, parent_manifest):
        token = manager.issue(
            parent_manifest=parent_manifest,
            grantee_agent_id="child_001",
            requested_capabilities=["search"],
            ttl_seconds=1,
        )
        # Manually expire it
        expired = token.model_copy(
            update={"expires_at": datetime.now(timezone.utc) - timedelta(seconds=10)}
        )
        assert manager.verify(expired, "search") is False

    def test_token_permits_case_insensitive(self, manager, parent_manifest):
        token = manager.issue(
            parent_manifest=parent_manifest,
            grantee_agent_id="child_001",
            requested_capabilities=["Search"],  # mixed case
        )
        assert token.permits("search") is True
        assert token.permits("SEARCH") is True
