# 04. ラベル → 練習メニュー対応表

## 基本ルール

- 出現頻度上位 2 ラベルを採用
- 合計 5〜10 分に収める
- 1 回の提案は 2〜3 ドリルまで

## 対応表

### `slow_initial_adjustment`

- Drill: 反応 + 小フリック
- 目安: 3 分
- チェックポイント: 視認後に一度で寄せすぎない

### `overflick`

- Drill: 小さく止めるマイクロフリック
- 目安: 3 分
- チェックポイント: 初動の勢いより停止精度を優先する

### `tracking_delay`

- Drill: 中距離トラッキング
- 目安: 3 分
- チェックポイント: 敵の少し先をなぞる

### `recoil_breakdown`

- Drill: 継続射撃のリコイル維持
- 目安: 3 分
- チェックポイント: 後半のブレを抑える

### `close_range_instability`

- Drill: 近距離腰撃ちの視点安定
- 目安: 2〜3 分
- チェックポイント: 近距離で振り回されない

### `ads_judgment_issue`

- Drill: 近距離での ADS / 腰撃ち切り替え確認
- 目安: 2 分
- チェックポイント: 近距離は腰撃ち優先を試す

## 生成ロジック

### 例 1

入力:

```json
{
  "label_counts": {
    "tracking_delay": 3,
    "close_range_instability": 2,
    "overflick": 1
  }
}
```

出力:

```json
{
  "priority_labels": [
    "tracking_delay",
    "close_range_instability"
  ],
  "drills": [
    {
      "title": "中距離トラッキング",
      "minutes": 3
    },
    {
      "title": "近距離腰撃ちの視点安定",
      "minutes": 3
    }
  ],
  "total_minutes": 6
}
```

### 例 2

入力:

```json
{
  "label_counts": {
    "slow_initial_adjustment": 2,
    "overflick": 2
  }
}
```

出力:

```json
{
  "priority_labels": [
    "slow_initial_adjustment",
    "overflick"
  ],
  "drills": [
    {
      "title": "反応 + 小フリック",
      "minutes": 3
    },
    {
      "title": "小さく止めるマイクロフリック",
      "minutes": 3
    }
  ],
  "total_minutes": 6
}
```
