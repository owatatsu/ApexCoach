from apexcoach.action_arbiter import ActionArbiter
from apexcoach.config import ArbiterConfig
from apexcoach.models import Action, Decision


def test_same_action_cooldown() -> None:
    arbiter = ActionArbiter(ArbiterConfig(same_action_cooldown_seconds=2.0))
    d = Decision(action=Action.HEAL, reason="heal")
    r1 = arbiter.arbitrate(d, timestamp=0.0)
    r2 = arbiter.arbitrate(d, timestamp=1.0)
    r3 = arbiter.arbitrate(d, timestamp=2.1)
    assert r1.emitted is True
    assert r2.emitted is False
    assert r3.emitted is True


def test_retreat_immediate_override_and_push_block() -> None:
    config = ArbiterConfig(
        same_action_cooldown_seconds=2.0,
        retreat_hold_seconds=1.2,
        push_block_after_retreat_seconds=3.0,
    )
    arbiter = ActionArbiter(config)

    r1 = arbiter.arbitrate(
        Decision(action=Action.RETREAT, reason="danger"),
        timestamp=0.0,
    )
    assert r1.action == Action.RETREAT
    assert r1.emitted is True

    # Within retreat hold window, action should stay RETREAT.
    r2 = arbiter.arbitrate(
        Decision(action=Action.PUSH, reason="advantage"),
        timestamp=0.5,
    )
    assert r2.action == Action.RETREAT

    # Hold ended, but PUSH still blocked by post-retreat suppression.
    r3 = arbiter.arbitrate(
        Decision(action=Action.PUSH, reason="advantage"),
        timestamp=2.0,
    )
    assert r3.action == Action.NONE
