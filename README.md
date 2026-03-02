# ApexCoach

Apex Legends for coaching and analysis only.

The MVP is intentionally small and rule-first:

- Capture gameplay video or screen
- Parse minimal UI signals (HP/Shield/teammate state)
- Detect lightweight events (damage, ally/enemy knock)
- Aggregate temporal state (1-5 seconds)
- Decide one action (`HEAL`, `RETREAT`, `PUSH`) with rules
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
{"frame_index": 42, "hp_pct": 0.62, "shield_pct": 0.30, "allies_alive": 2, "allies_down": 1, "enemy_knock": false, "ally_knock": true}
```

Run with:

```bash
apexcoach --video path/to/apex_recording.mp4 --telemetry path/to/telemetry.jsonl
```

## Tests

```bash
pytest -q -p no:cacheprovider
```

## License

Apache-2.0. See [LICENSE](LICENSE).
