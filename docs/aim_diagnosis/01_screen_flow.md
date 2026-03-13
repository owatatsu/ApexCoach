# 01. 画面遷移図

## 全体フロー

```text
Upload
  ↓
Clip Editor
  ↓
Analysis List
  ↓
Clip Detail
  ↓
Training Plan
```

## 画面ごとの役割

### 1. Upload

目的:

- 録画ファイルを受け取る
- サムネイル、長さ、形式を確認させる

主な UI:

- mp4 ファイル選択
- サムネイル
- 動画長
- アップロード進捗
- `解析を開始` ボタン

状態:

- `idle`
- `uploading`
- `uploaded`
- `failed`

### 2. Clip Editor

目的:

- 手動で交戦クリップを切り出す

主な UI:

- 動画プレイヤー
- 再生 / 一時停止
- 5 秒戻る / 進む
- 現在時刻
- `開始点を追加`
- `終了点を追加`
- クリップ一覧
- 削除

主要ルール:

- 1 動画あたり 3〜10 クリップ程度
- 1 クリップは `start_sec < end_sec`
- 1 クリップは 2〜20 秒程度を推奨

### 3. Analysis List

目的:

- クリップごとの診断結果を一覧で見せる

主な UI:

- クリップカード
- 時間帯
- 一言サマリー
- 弱点ラベル
- 優先度
- `詳細を見る`

表示順:

- デフォルトはクリップ順
- 将来的に優先度順へ切り替え可能

### 4. Clip Detail

目的:

- 単一クリップの診断を具体的に示す

主な UI:

- 該当クリップ再生
- 一言評価
- 良かった点
- 改善点
- ラベル
- 次に意識すること

### 5. Training Plan

目的:

- 全体の弱点を練習メニューに変換する

主な UI:

- 上位ラベル 1〜3 個
- 優先度
- 5〜10 分の練習メニュー
- チェックポイント

## 推奨ルーティング

```text
/aim/upload
/aim/videos/:video_id/clips
/aim/videos/:video_id/results
/aim/clips/:clip_id
/aim/videos/:video_id/training
```

## 状態遷移

```text
video_uploaded
  -> clips_defined
  -> analysis_requested
  -> analysis_completed
  -> training_plan_ready
```
