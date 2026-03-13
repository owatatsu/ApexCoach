# 02. クリップ分析 API 入出力仕様

## 目的

手動で切り出した 1 つ以上のクリップを受け取り、クリップごとの診断結果と全体練習メニューを返す。

## API 一覧

### 1. 動画登録

`POST /api/aim/videos`

用途:

- 動画メタデータを作成する
- ファイル保存先と ID を返す

リクエスト例:

```json
{
  "filename": "rank_match_001.mp4",
  "duration_sec": 1185.4
}
```

レスポンス例:

```json
{
  "video_id": "vid_01JXYZ...",
  "status": "uploaded",
  "duration_sec": 1185.4
}
```

### 2. クリップ登録

`POST /api/aim/videos/{video_id}/clips`

用途:

- 手動指定したクリップ群を保存する

リクエスト例:

```json
{
  "clips": [
    {
      "clip_id": "clip_001",
      "start_sec": 35.2,
      "end_sec": 43.6,
      "note": "建物内の 1v1"
    },
    {
      "clip_id": "clip_002",
      "start_sec": 122.0,
      "end_sec": 130.4
    }
  ]
}
```

レスポンス例:

```json
{
  "video_id": "vid_01JXYZ...",
  "clip_count": 2,
  "status": "clips_saved"
}
```

### 3. 分析実行

`POST /api/aim/videos/{video_id}/analyze`

用途:

- 保存済みクリップを分析する

リクエスト例:

```json
{
  "mode": "manual_clips",
  "labels": [
    "slow_initial_adjustment",
    "overflick",
    "tracking_delay",
    "recoil_breakdown",
    "close_range_instability",
    "ads_judgment_issue"
  ]
}
```

レスポンス例:

```json
{
  "video_id": "vid_01JXYZ...",
  "analysis_job_id": "job_01JXYZ...",
  "status": "queued"
}
```

### 4. 分析結果取得

`GET /api/aim/videos/{video_id}/results`

レスポンス例:

```json
{
  "video_id": "vid_01JXYZ...",
  "status": "completed",
  "clips": [
    {
      "clip_id": "clip_001",
      "start_sec": 35.2,
      "end_sec": 43.6,
      "summary": "初動は良いが、追いエイムで少し遅れが出ている",
      "strengths": [
        "視認から射撃開始までは速い"
      ],
      "weaknesses": [
        "敵の横移動に対して視点が後追いになっている"
      ],
      "labels": [
        "tracking_delay",
        "close_range_instability"
      ],
      "confidence": 0.71,
      "next_focus": "敵の少し先をなぞる意識を持つ"
    }
  ],
  "training_plan": {
    "priority_labels": [
      "tracking_delay",
      "close_range_instability"
    ],
    "drills": [
      {
        "title": "中距離トラッキング",
        "minutes": 3,
        "instruction": "一定速度の横移動を視点で置き去りにしない"
      },
      {
        "title": "近距離腰撃ちの安定",
        "minutes": 3,
        "instruction": "最初の数発が当たった後に振り回されない"
      }
    ],
    "total_minutes": 6
  }
}
```

## サーバー内部の分析入力モデル

```json
{
  "video_id": "vid_01JXYZ...",
  "clip": {
    "clip_id": "clip_001",
    "start_sec": 35.2,
    "end_sec": 43.6
  },
  "analysis_context": {
    "fps": 30,
    "sample_stride": 2,
    "center_region": [0.35, 0.25, 0.65, 0.75]
  }
}
```

## クリップ分析結果の内部モデル

```json
{
  "clip_id": "clip_001",
  "summary": "初動は良いが、追いエイムで少し遅れが出ている",
  "strengths": [
    "視認から射撃開始までは速い"
  ],
  "weaknesses": [
    "敵の横移動に対して視点が後追いになっている"
  ],
  "labels": [
    "tracking_delay"
  ],
  "confidence": 0.71,
  "next_focus": "敵の少し先をなぞる意識を持つ",
  "metrics": {
    "time_to_first_shot_ms": 280,
    "aim_path_variance": 0.41,
    "tracking_error_score": 0.66,
    "close_range_score": 0.72
  }
}
```

## 設計メモ

- MVP では同期 API よりも job 方式が安全
- 結果レスポンスは UI がそのまま描画できる shape を優先する
- `metrics` は内部向けで、UI には直接出さなくてもよい
