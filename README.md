# ApexCoach

Apex Legends for coaching and analysis only.

The MVP is intentionally small and rule-first:

- Capture gameplay video or screen
- Parse minimal UI signals (HP/Shield/teammate state)
- Detect lightweight events (damage, ally/enemy knock)
- Aggregate temporal state (1-5 seconds)
- Decide one action (`HEAL`, `RETREAT`, `PUSH`, `TAKE_COVER`, `TAKE_HIGH_GROUND`) with rules
- Stabilize output with cooldown/hysteresis arbiter
- Render overlay and save structured logs

## Safety Policy

This project does not implement:

- input automation
- memory reading
- packet sniffing
- bot gameplay

It is advice-only.

## Install

```bash
pip install -e .
pip install -e .[dev]
```

## Offline MVP Run

```bash
apexcoach --video path/to/apex_recording.mp4 --config config/apexcoach.example.yaml
```

With log and overlay video output:

```bash
apexcoach ^
  --video path/to/apex_recording.mp4 ^
  --config config/apexcoach.example.yaml ^
  --log logs/session.jsonl ^
  --output-video output/overlay.mp4
```

## Offline Local LLM Coaching (LM Studio)

LM Studio local server example:

```bash
# Start Local Server in LM Studio
# API endpoint: http://127.0.0.1:1234/v1
# Model: qwen3.5-9b-instruct (quantized recommended)
```

Enable in config:

```yaml
llm:
  enabled: true
  provider: "lmstudio"
  model_name: "qwen3.5-9b-instruct"
  base_url: "http://127.0.0.1:1234/v1"
  api_key: "lm-studio"
  timeout_ms: 300
  temperature: 0.1
  max_reason_chars: 48
  min_request_interval_ms: 1000
  parse_retry_count: 1
  failure_threshold: 5
  disable_seconds: 10
  request_log_path: "logs/llm_requests.jsonl"
  llm_max_tokens: 64
  advice_enabled: true
  offline_review_enabled: true
  offline_review_output: "logs/coach_review.md"
```

Then run normal offline/realtime pipeline.
- Rule engine stays primary.
- LLM is event-driven, low-frequency helper.
- Provider failures always fall back to rule decisions.
- Per-request logs are saved to `logs/llm_requests.jsonl`.
- Offline review markdown is saved to `logs/coach_review.md`.

CLI override example:

```bash
apexcoach ^
  --video path/to/apex_recording.mp4 ^
  --config config/apexcoach.example.yaml ^
  --llm-enable ^
  --llm-provider lmstudio ^
  --llm-model qwen3.5-9b-instruct ^
  --llm-base-url http://127.0.0.1:1234/v1 ^
  --llm-review-output logs/coach_review.md
```

Ollama can still be used by setting `llm.provider: "ollama"` and `llm.base_url: "http://127.0.0.1:11434"`.

## Optional Telemetry JSONL

For faster rule tuning, you can inject UI state from JSONL per frame.

Example line:

```json
{"frame_index": 42, "hp_pct": 0.62, "shield_pct": 0.30, "allies_alive": 2, "allies_down": 1, "enemy_knock": false, "ally_knock": true, "low_ground_disadvantage": true, "low_ground_confidence": 0.82, "exposed_no_cover": false, "exposed_confidence": 0.35}
```

Run with:

```bash
apexcoach --video path/to/apex_recording.mp4 --telemetry path/to/telemetry.jsonl
```

## Realtime Run

Capture live game screen and show guidance in realtime:

```bash
apexcoach --realtime --config config/apexcoach.example.yaml
```

Monitor/region example:

```bash
apexcoach --realtime --monitor 1 --region 0,0,1920,1080 --duration 300
```

- Stop with `Ctrl+C` (or after `--duration` seconds)
- `--output-video` can record overlay output while running
- Use `overlay.window_mode: "hud"` for instruction-only overlay (no full capture window)

Recommended overlay settings:

- `overlay.display_hold_seconds: 3.0`
- `overlay.max_lines: 3`
- `overlay.background_alpha: 0.5`
- `overlay.text_scale: 0.85`

ROI debug overlay (to verify scan regions):

- `overlay.debug_show_rois: true`
- `overlay.debug_roi_alpha: 0.35`
- `overlay.debug_show_roi_labels: true`
- For realtime visual check, use `overlay.window_mode: "frame"` during debugging

Recommended local models:

- Text coaching: `qwen3.5-9b-instruct` (LM Studio)
- Vision extension later: `qwen2.5vl:7b`

## Tests

```bash
pytest -q -p no:cacheprovider
```

## Performance Tuning

Use `performance` in config for faster offline conversion:

- `parallel_io`: enable threaded read/write pipeline
- `read_prefetch_queue_size`: frame prefetch buffer
- `write_queue_size`: async writer buffer
- `opencv_threads`: OpenCV internal threads (`0` keeps default)

## License

Apache-2.0. See [LICENSE](LICENSE).
