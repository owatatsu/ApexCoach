# ApexCoach Aim Diagnosis MVP

このディレクトリは、録画ベースのエイム診断 MVP を実装へ落とすための設計資料をまとめる。

含まれる資料:

- `01_screen_flow.md`: 画面遷移と各画面の責務
- `02_clip_analysis_api.md`: クリップ分析 API の入出力仕様
- `03_comment_templates.md`: 診断コメントのテンプレート
- `04_label_drill_mapping.md`: ラベルから練習メニューへの対応表
- `05_task_breakdown.md`: 実装タスク分解

MVP の基本方針:

- 最初は手動クリップ指定を採用する
- 分析は 1 クリップ単位で行う
- 出力は長文レビューより短い診断カードを優先する
- 練習提案まで返して「次に何をやるか」を明確にする
