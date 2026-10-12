"""Approving a Lens/MCP actor proposal from the Guard Approvals inbox must run it.

Regression: the inbox called apply_decision directly. The row flipped to
"approved" but spec.execute never ran (a deactivate_agent_identity left the
identity active), and dispatch_confirm then refused it as "action is approved".
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

from app.modules.glens.actor import helpers  # noqa: E402
from app.modules.guard.routers import approvals  # noqa: E402

WS = "fd4b6608-f320-44b8-af22-fc579bd53600"


def _row(rule_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        id="1778ced2-2d80-4907-8e5d-2495b1ac0d26", rule_id=rule_id, status="pending",
        workspace_id=WS, tool_name="deactivate_agent_identity",
    )


ACTOR = _row(f"{helpers.LENS_ACTOR_RULE_ID_PREFIX}.deactivate_agent_identity")
POLICY = _row("no-rm-rf")


def test_is_actor_row():
    assert helpers.is_actor_row(ACTOR)
    assert not helpers.is_actor_row(POLICY)
    assert not helpers.is_actor_row(_row(helpers.LENS_ACTOR_RULE_ID_PREFIX))  # prefix alone, no tool
    assert not helpers.is_actor_row(_row(None))


@pytest.mark.parametrize("decision,expected", [("approved", "confirm"), ("rejected", "cancel")])
def test_decide_actor_row_routes_to_dispatch(monkeypatch, decision, expected):
    calls = []
    monkeypatch.setattr(helpers, "dispatch_confirm", lambda db, **kw: calls.append(("confirm", kw)) or {"executed": True})
    monkeypatch.setattr(helpers, "dispatch_cancel", lambda db, **kw: calls.append(("cancel", kw)) or {"executed": False})
    helpers.decide_actor_row(
        None, ACTOR, decision=decision, workspace_id=WS,
        clerk_user_id="user_1", user_email="a@b.c", role="admin", reason="tidy",
    )
    assert [c[0] for c in calls] == [expected]
    assert calls[0][1]["action_id"] == ACTOR.id


class _DB:
    def refresh(self, row):
        row.status = "approved"


@pytest.fixture
def decide(monkeypatch):
    applied, dispatched = [], []
    monkeypatch.setattr(approvals, "get_clerk_user_email", lambda uid: "sudhi@example.com")
    monkeypatch.setattr(approvals, "sweep_if_timed_out", lambda db, row: row)
    monkeypatch.setattr(approvals, "_serialize", lambda row: {"id": row.id, "status": row.status})
    monkeypatch.setattr(approvals, "DecisionOut", lambda **kw: kw)
    monkeypatch.setattr(approvals, "can_decide", lambda *a, **k: (True, None))
    monkeypatch.setattr(approvals, "apply_decision", lambda db, row, **kw: applied.append(row.rule_id) or row)
    monkeypatch.setattr(approvals, "_resume_workflow_run", lambda *a, **k: None)
    monkeypatch.setattr(helpers, "decide_actor_row", lambda db, row, **kw: dispatched.append((row.rule_id, kw)) or {"executed": True})

    def call(row, decision="approved", reason=None):
        monkeypatch.setattr(approvals, "_get_row", lambda *a, **k: SimpleNamespace(**vars(row)))
        out = approvals.decide_approval(
            request_id=row.id, body=SimpleNamespace(decision=decision, reason=reason),
            workspace_id=WS, user_id="user_1", role="admin", db=_DB(),
        )
        return out, applied, dispatched
    return call


def test_inbox_approve_of_actor_row_executes_it(decide):
    out, applied, dispatched = decide(ACTOR)
    assert applied == []  # never the bare status flip
    assert dispatched[0][0] == ACTOR.rule_id
    assert dispatched[0][1]["decision"] == "approved" and dispatched[0][1]["role"] == "admin"
    assert out["request"]["status"] == "approved"


def test_inbox_reject_of_actor_row_cancels_it(decide):
    _, applied, dispatched = decide(ACTOR, decision="rejected", reason="not now")
    assert applied == [] and dispatched[0][1]["decision"] == "rejected"
    assert dispatched[0][1]["reason"] == "not now"


def test_inbox_policy_approval_unchanged(decide):
    _, applied, dispatched = decide(POLICY)
    assert applied == ["no-rm-rf"] and dispatched == []


def test_inbox_maps_confirm_errors_to_http(decide, monkeypatch):
    def boom(*a, **k):
        raise helpers.ConfirmError(403, "only the proposer can decide this action")
    monkeypatch.setattr(helpers, "decide_actor_row", boom)
    with pytest.raises(approvals.HTTPException) as e:
        decide(ACTOR)
    assert e.value.status_code == 403
