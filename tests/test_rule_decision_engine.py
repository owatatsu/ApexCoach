from apexcoach.config import ThresholdConfig
from apexcoach.models import Action, GameState
from apexcoach.rule_decision_engine import RuleDecisionEngine


def test_retreat_when_low_and_under_fire() -> None:
    engine = RuleDecisionEngine(ThresholdConfig())
    state = GameState(
        timestamp=10.0,
        hp_pct=0.2,
        shield_pct=0.2,
        allies_alive=2,
        allies_down=1,
        recent_damage_1s=0.15,
        recent_damage_3s=0.4,
        under_fire=True,
        enemy_knock_recent=False,
        ally_knock_recent=False,
    )
    decision = engine.decide(state)
    assert decision.action == Action.RETREAT


def test_heal_when_safe_and_low_total() -> None:
    engine = RuleDecisionEngine(ThresholdConfig())
    state = GameState(
        timestamp=11.0,
        hp_pct=0.35,
        shield_pct=0.2,
        allies_alive=3,
        allies_down=0,
        recent_damage_1s=0.0,
        recent_damage_3s=0.1,
        under_fire=False,
        enemy_knock_recent=False,
        ally_knock_recent=False,
    )
    decision = engine.decide(state)
    assert decision.action == Action.HEAL


def test_push_on_enemy_knock_with_good_hp() -> None:
    engine = RuleDecisionEngine(ThresholdConfig())
    state = GameState(
        timestamp=12.0,
        hp_pct=0.95,
        shield_pct=0.85,
        allies_alive=3,
        allies_down=0,
        recent_damage_1s=0.0,
        recent_damage_3s=0.1,
        under_fire=False,
        enemy_knock_recent=True,
        ally_knock_recent=False,
    )
    decision = engine.decide(state)
    assert decision.action == Action.PUSH
