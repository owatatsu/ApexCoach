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
