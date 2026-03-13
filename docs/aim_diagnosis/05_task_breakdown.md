# 05. 実装タスク分解

## P0: 体験成立に必要な最小構成

### Backend / Core

- `Video` / `Clip` / `AnalysisResult` / `TrainingPlan` のモデル追加
- クリップ分析エントリポイント追加
- 1 クリップ単位のサンプリング処理追加
- ラベル推定のルールベース実装
- コメントテンプレートの適用
- 練習メニュー生成

### Frontend / UI

- Upload 画面
- Clip Editor 画面
- Analysis List 画面
- Clip Detail 画面
- Training Plan 画面

### Storage

- 動画メタデータ保存
- クリップ時刻保存
- 分析結果保存

## P1: 先に作る順番

### Step 1. データモデル

- `src/apexcoach/aim_diagnosis/models.py`
- `VideoRef`
- `ClipRange`
- `ClipAnalysisMetrics`
- `ClipDiagnosis`
- `TrainingDrill`
- `TrainingPlan`

### Step 2. 診断ロジック

- `src/apexcoach/aim_diagnosis/labeler.py`
- `src/apexcoach/aim_diagnosis/comment_builder.py`
- `src/apexcoach/aim_diagnosis/training_planner.py`

### Step 3. クリップ処理

- `src/apexcoach/aim_diagnosis/clip_analyzer.py`
- 動画読み込み
- フレームサンプリング
- 簡易メトリクス抽出

### Step 4. API / CLI

- `src/apexcoach/aim_diagnosis/service.py`
- クリップ登録 API
- 分析実行 API
- 結果取得 API

### Step 5. UI

- 手動クリップ選択 UI
- 分析結果一覧 UI
- クリップ詳細 UI
- 練習メニュー UI

## 推奨ディレクトリ構成

```text
src/apexcoach/aim_diagnosis/
  __init__.py
  models.py
  clip_analyzer.py
  metrics.py
  labeler.py
  comment_builder.py
  training_planner.py
  service.py
```

## 最初の実装単位

### Milestone A

- `models.py`
- `training_planner.py`
- ラベル → 練習変換
- テンプレート出力

成功条件:

- ダミー分析結果から練習メニューを返せる

### Milestone B

- 手動クリップ JSON 入力
- クリップ単位分析のスタブ
- 結果 JSON 生成

成功条件:

- API から clip ごとの診断レスポンスを返せる

### Milestone C

- 動画プレイヤー UI
- クリップ編集 UI
- 結果一覧 UI

成功条件:

- 手動クリップ指定から結果閲覧まで一連で通る

## テスト方針

- ルールベースのラベル判定は unit test で固定
- コメント生成は snapshot 的に文字列検証
- 練習メニュー生成は label count ごとに期待値を固定
- 動画処理は最初は synthetic clip / mock metrics で十分

## リスク管理

- 自動抽出は後回しにする
- 敵骨格推定は MVP に入れない
- UI を先に作り込みすぎず、API shape を先に固める
