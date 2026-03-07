from __future__ import annotations

from apexcoach.models import Action


_ACTION_LABELS: dict[Action, str] = {
    Action.NONE: "待機",
    Action.HEAL: "回復",
    Action.RETREAT: "退避",
    Action.TAKE_COVER: "遮蔽へ移動",
    Action.TAKE_HIGH_GROUND: "高所を取る",
    Action.PUSH: "攻める",
}

_REASON_LABELS: dict[str, str] = {
    "No strong signal.": "強いシグナルはありません。",
    "Ally down while taking fire.": "味方がダウンし、さらに被弾中です。",
    "Isolated or almost isolated.": "孤立しているか、孤立寸前です。",
    "High incoming damage in last 1s.": "直近1秒の被ダメージが大きいです。",
    "Low HP+Shield while under fire.": "被弾中で、HPとシールドが低いです。",
    "Under fire in exposed area. Move behind cover.": "開けた場所で撃たれています。遮蔽へ移動してください。",
    "Under fire from lower position. Take higher ground.": "不利な低所で撃ち合っています。高所を取りましょう。",
    "Safe window and low HP+Shield.": "安全な隙があり、HPとシールドが低いです。",
    "Enemy knock advantage with healthy team.": "敵ノックの有利があり、味方の体力状況も良好です。",
}


def action_label(action: Action) -> str:
    return _ACTION_LABELS.get(action, action.value)


def localize_reason(reason: str) -> str:
    text = (reason or "").strip()
    if not text:
        return ""
    return _REASON_LABELS.get(text, text)


def format_instruction_line(action: Action, reason: str) -> str:
    label = action_label(action)
    localized_reason = localize_reason(reason)
    if localized_reason:
        return f"{label} | {localized_reason}"
    return label


def is_llm_label(label: str) -> bool:
    return label in {"LLM", "補足"}
