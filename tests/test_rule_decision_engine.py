from apexcoach.config import ThresholdConfig
from apexcoach.models import Action, GameState
from apexcoach.rule_decision_engine import RuleDecisionEngine


def test_retreat_when_low_and_under_fire() -> None:
    engine = RuleDecisionEngine(
        ThresholdConfig(
            low_hp_consecutive_frames=1,
            retreat_lowhp_stationary_frames=0,
        )
    )
    state = GameState(
        timestamp=10.0,
        hp_pct=0.2,
        shield_pct=0.2,
        retreat_low_hp_streak=1,
        stationary_frames=1,
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
    engine = RuleDecisionEngine(
        ThresholdConfig(
            low_hp_consecutive_frames=1,
            heal_stationary_frames=0,
        )
    )
    state = GameState(
        timestamp=11.0,
        hp_pct=0.35,
        shield_pct=0.2,
        heal_low_hp_streak=1,
        stationary_frames=1,
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


def test_take_cover_when_under_fire_and_exposed() -> None:
    engine = RuleDecisionEngine(
        ThresholdConfig(exposed_confidence_min=0.6, low_hp_consecutive_frames=1)
    )
    state = GameState(
        timestamp=13.0,
        hp_pct=0.7,
        shield_pct=0.7,
        allies_alive=3,
        allies_down=0,
        recent_damage_1s=0.2,
        recent_damage_3s=0.3,
        under_fire=True,
        enemy_knock_recent=False,
        ally_knock_recent=False,
        exposed_no_cover=True,
        exposed_confidence=0.8,
    )
    decision = engine.decide(state)
    assert decision.action == Action.TAKE_COVER


def test_take_high_ground_when_under_fire_and_low_ground() -> None:
    engine = RuleDecisionEngine(
        ThresholdConfig(low_ground_confidence_min=0.6, low_hp_consecutive_frames=1)
    )
    state = GameState(
        timestamp=14.0,
        hp_pct=0.9,
        shield_pct=0.8,
        allies_alive=3,
        allies_down=0,
        recent_damage_1s=0.12,
        recent_damage_3s=0.25,
        under_fire=True,
        enemy_knock_recent=False,
        ally_knock_recent=False,
        low_ground_disadvantage=True,
        low_ground_confidence=0.82,
        exposed_no_cover=False,
        exposed_confidence=0.2,
    )
    decision = engine.decide(state)
    assert decision.action == Action.TAKE_HIGH_GROUND


def test_no_heal_or_retreat_when_vitals_unreliable() -> None:
    engine = RuleDecisionEngine(ThresholdConfig(vitals_confidence_min=0.7))
    state = GameState(
        timestamp=15.0,
        hp_pct=0.05,
        shield_pct=0.05,
        vitals_confidence=0.2,
        allies_alive=3,
        allies_down=0,
        recent_damage_1s=0.5,
        recent_damage_3s=0.7,
        under_fire=True,
        enemy_knock_recent=False,
        ally_knock_recent=False,
    )
    decision = engine.decide(state)
    assert decision.action == Action.TAKE_COVER or decision.action == Action.NONE


def test_heal_requires_low_hp_streak() -> None:
    engine = RuleDecisionEngine(
        ThresholdConfig(low_hp_consecutive_frames=3, heal_stationary_frames=0)
    )
    base_state = dict(
        timestamp=16.0,
        hp_pct=0.25,
        shield_pct=0.25,
        vitals_confidence=0.9,
        allies_alive=3,
        allies_down=0,
        recent_damage_1s=0.0,
        recent_damage_3s=0.0,
        under_fire=False,
        enemy_knock_recent=False,
        ally_knock_recent=False,
    )
    early = engine.decide(GameState(**base_state, heal_low_hp_streak=2))
    ready = engine.decide(GameState(**base_state, heal_low_hp_streak=3))
    assert early.action == Action.NONE
    assert ready.action == Action.HEAL


def test_heal_blocked_while_moving() -> None:
    engine = RuleDecisionEngine(
        ThresholdConfig(
            low_hp_consecutive_frames=1,
            heal_stationary_frames=4,
        )
    )
    state = GameState(
        timestamp=17.0,
        hp_pct=0.25,
        shield_pct=0.2,
        vitals_confidence=0.9,
        heal_low_hp_streak=6,
        stationary_frames=1,
        is_moving=True,
        allies_alive=3,
        allies_down=0,
        recent_damage_1s=0.0,
        recent_damage_3s=0.0,
        under_fire=False,
        enemy_knock_recent=False,
        ally_knock_recent=False,
    )
    decision = engine.decide(state)
    assert decision.action == Action.NONE


def test_candidates_return_multiple_actions() -> None:
    engine = RuleDecisionEngine(
        ThresholdConfig(
            low_hp_consecutive_frames=1,
            retreat_lowhp_stationary_frames=0,
            exposed_confidence_min=0.6,
            low_ground_confidence_min=0.6,
        )
    )
    state = GameState(
        timestamp=18.0,
        hp_pct=0.2,
        shield_pct=0.2,
        vitals_confidence=0.9,
        retreat_low_hp_streak=5,
        stationary_frames=8,
        allies_alive=3,
        allies_down=0,
        recent_damage_1s=0.2,
        recent_damage_3s=0.4,
        under_fire=True,
        enemy_knock_recent=False,
        ally_knock_recent=False,
        exposed_no_cover=True,
        exposed_confidence=0.8,
        low_ground_disadvantage=True,
        low_ground_confidence=0.8,
    )
    candidates = engine.decide_candidates(state)
    assert [c.action for c in candidates][:3] == [
        Action.RETREAT,
        Action.TAKE_COVER,
        Action.TAKE_HIGH_GROUND,
    ]


def test_decide_uses_first_candidate_as_primary() -> None:
    engine = RuleDecisionEngine(
        ThresholdConfig(low_hp_consecutive_frames=1, retreat_lowhp_stationary_frames=0)
    )
    state = GameState(
        timestamp=19.0,
        hp_pct=0.2,
        shield_pct=0.2,
        vitals_confidence=0.9,
        retreat_low_hp_streak=2,
        stationary_frames=3,
        allies_alive=3,
        allies_down=0,
        recent_damage_1s=0.2,
        recent_damage_3s=0.2,
        under_fire=True,
        enemy_knock_recent=False,
        ally_knock_recent=False,
    )
    decision = engine.decide(state)
    assert decision.action == Action.RETREAT
